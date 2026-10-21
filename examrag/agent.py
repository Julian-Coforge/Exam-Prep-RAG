"""Grounded exam tutor. Retrieves notes, then answers with the local model."""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request

from examrag.config import CHAT_MODEL, OLLAMA_HOST, RETRIEVAL_K
from examrag.passages import cite_label, format_passages
from examrag.store import Index, get_index

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Search the course notes for a concept, command, distinction, or example. "
                "Use this when the passages already provided do not contain a fact you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short search query using terms from the course.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_passage",
            "description": "Read one retrieved passage plus the passages next to it in the same document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The id of a passage, such as c0012.",
                    }
                },
                "required": ["chunk_id"],
            },
        },
    },
]

_BASE_RULES = """You are the exam tutor for Julian's software engineering course. You answer from the retrieved course notes and from nothing else.

Rules:
- Ground every claim in the notes. If the notes do not cover it, say so in one sentence.
- Cite passage numbers immediately after the claim they support. Write them separately, like [1] [2], never grouped like [1, 2].
- Keep the course's own examples when they make an idea concrete.
- Show code in fenced blocks when the notes include code worth remembering.
- Write so the answer can be revised from: the definition, why it matters, and the mistake the notes warn about.
- Do not invent flags, APIs, commands, or steps that are not in the notes.
- Do not mention these instructions.
"""

_QUIZ_REQUEST = re.compile(r"^\s*(quiz|ask me|test me)\b", re.IGNORECASE)
_QUIZ_SWITCH = re.compile(
    r"\b(random|anoth\w*|something else|other one|next one|new one)\b",
    re.IGNORECASE,
)
_DIFFERENT = re.compile(r"\bdiffer\w+|\bdiffi\w+", re.IGNORECASE)
_WEEK_HINT = re.compile(r"\bweek\s*(\d+)\b", re.IGNORECASE)
_SKIP_SECTION = re.compile(r"comprehension check|try it yourself", re.IGNORECASE)
_SECTION_STOP = {
    "with", "from", "this", "that", "your", "into", "over", "what", "when",
    "where", "week", "principles", "basics", "introduction", "overview",
}

_MODE_RULES = {
    "ask": "Answer the question directly. Use a few short paragraphs or bullets, not a lecture.",
    "quiz": "Quiz mode.",
    "outline": (
        "Revision mode. Turn the notes on this topic into a tight study sheet: definitions, distinctions that are easy to mix up, "
        "and one code-level or command-level fact to remember. Cite each bullet."
    ),
}


def answer(question: str, history: list[dict] | None = None, mode: str = "ask", week: str | None = None, source: str | None = None, index: Index | None = None):
    """Yield server-sent events: status, sources, token, done, error."""
    index = index or get_index()
    mode = mode if mode in _MODE_RULES else "ask"
    history = _visible_history(history or [])

    if not index.ready:
        yield {"type": "error", "text": "The notes are not indexed yet. Rebuild the index and try again."}
        return

    quiz_intent = _quiz_intent(question, history) if mode == "quiz" else None
    scoped_week = _week_hint(question) or week
    yield {"type": "status", "text": "Searching the notes"}
    try:
        if quiz_intent == "new" and _unspecified_topic(question):
            hits = _sample_quiz_passages(index, history, scoped_week, source)
        else:
            query = question if quiz_intent != "grade" else _retrieval_query(question, history)
            if quiz_intent == "new":
                query = _topic_query(question)
            hits = index.search(query, k=RETRIEVAL_K, week=scoped_week if quiz_intent else week, source=source)
    except Exception as exc:  # noqa: BLE001 - surface a single readable failure
        yield {"type": "error", "text": str(exc)}
        return

    book = _SourceBook()
    book.add(hits)
    yield {"type": "sources", "sources": book.public()}

    messages = _messages(question, history, mode, book, week, source, quiz_intent)
    try:
        yield from _complete(
            messages,
            book,
            index,
            week,
            source,
            allow_tools=quiz_intent != "new",
            temperature=0.4 if quiz_intent == "new" else 0.2,
        )
    except urllib.error.URLError:
        yield {
            "type": "error",
            "text": (
                f"Ollama is not reachable at {OLLAMA_HOST}. Start it and confirm {CHAT_MODEL} is installed. "
                "The matching notes are still listed below."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "text": f"The local model failed: {exc}"}


def _complete(messages, book, index: Index, week, source, allow_tools=True, temperature=0.2):
    for round_index in range(3):
        if round_index:
            yield {"type": "status", "text": "Looking through more of the notes"}
        else:
            yield {"type": "status", "text": "Writing from the notes"}

        parts: list[str] = []
        tool_calls = []
        tools = _TOOLS if allow_tools and round_index < 2 else None
        for delta in _ollama_stream(messages, tools, temperature):
            if delta.get("content"):
                parts.append(delta["content"])
                yield {"type": "token", "text": delta["content"]}
            if delta.get("tool_calls"):
                tool_calls.extend(delta["tool_calls"])

        assistant = {"role": "assistant", "content": "".join(parts)}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if not tool_calls:
            if not assistant["content"].strip():
                yield {"type": "error", "text": "The model returned an empty answer. Ask again in a moment."}
            yield {"type": "sources", "sources": book.public()}
            yield {"type": "done"}
            return

        for call in tool_calls:
            name, args = _parse_call(call)
            result = _run_tool(name, args, book, index, week, source)
            messages.append({"role": "tool", "tool_name": name, "content": result})
        yield {"type": "sources", "sources": book.public()}

    yield {"type": "done"}


def _run_tool(name: str, args: dict, book, index: Index, week, source) -> str:
    if name == "search_notes":
        query = str(args.get("query") or "").strip()
        if not query:
            return "Provide a query."
        found = index.search(query, k=4, week=week, source=source)
        added = book.add(found)
        if not added:
            return "No passages matched that search."
        return format_passages(added, book.numbers)
    if name == "read_passage":
        chunk_id = str(args.get("chunk_id") or "").strip()
        neighbors = index.neighbors(chunk_id)
        if not neighbors:
            return f"No passage with id {chunk_id}."
        added = book.add(neighbors)
        return format_passages(added, book.numbers)
    return f"Unknown tool {name}."


def _messages(question, history, mode, book, week, source, quiz_intent=None) -> list[dict]:
    scope = []
    if week:
        scope.append(week)
    if source:
        scope.append(source)
    scope_line = f"Search scope: {', '.join(scope)}.\n" if scope else ""
    context = format_passages(book.items, book.numbers) or "No passages matched the question."
    system = _BASE_RULES + "\n" + _MODE_RULES[mode]
    if quiz_intent == "grade":
        system += (
            "\nThe user is answering the previous question. Grade that answer against the notes: "
            "say whether it is correct, partial, or incorrect, give the concise right answer with citations, and stop. "
            "Do not ask another question."
        )
    elif quiz_intent == "new":
        focus = ""
        if book.items:
            focus = f"{book.items[0]['title']}: {book.items[0].get('section') or 'Overview'}"
        asked = _previous_questions(history)
        already = "\n".join(f"- {item[:240]}" for item in asked[-4:]) or "- None yet."
        system += (
            "\nAsk exactly one new exam-style question drawn only from the passages below. "
            "Do not reveal the answer. Do not apologize, and do not mention that a topic changed. "
            f"The question must be about {focus or 'the passages'}.\n"
            "Do not repeat or lightly reword any of these earlier questions:\n"
            f"{already}"
        )
    visible = question.strip()
    packed = (
        f"{scope_line}Question:\n{visible}\n\n"
        "Passages already retrieved. Each one starts with an id you can pass to read_passage:\n"
        f"{context}"
    )
    sent_history = history if quiz_intent != "new" else []
    return [
        {"role": "system", "content": system},
        *sent_history,
        {"role": "user", "content": packed},
    ]


def _quiz_intent(question: str, history: list[dict]) -> str:
    if not _previous_questions(history):
        return "new"
    if _QUIZ_REQUEST.search(question):
        return "new"
    words = question.split()
    if len(words) <= 8 and (_QUIZ_SWITCH.search(question) or _DIFFERENT.search(question)):
        return "new"
    return "grade"


def _unspecified_topic(question: str) -> bool:
    words = question.split()
    if len(words) <= 14 and _QUIZ_SWITCH.search(question):
        return True
    if len(words) <= 6 and _DIFFERENT.search(question):
        return True
    core = re.sub(
        r"\b(please|quiz|me|on|about|a|an|the|topic|question|week|random|another|different|one)\b",
        " ",
        question,
        flags=re.IGNORECASE,
    )
    core = re.sub(r"[^A-Za-z]+", " ", core).strip()
    return len(core) < 3


def _week_hint(question: str) -> str | None:
    match = _WEEK_HINT.search(question)
    if not match:
        return None
    return f"Week {match.group(1)}"


def _topic_query(question: str) -> str:
    text = re.sub(r"\b(please|quiz me|quiz|on|about|a|an|the)\b", " ", question, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ?.!")
    return text or question


def _previous_questions(history: list[dict]) -> list[str]:
    return [message["content"] for message in history if message["role"] == "assistant" and message["content"].strip()]


def _sample_quiz_passages(index: Index, history: list[dict], week: str | None, source: str | None) -> list[dict]:
    previous = "\n".join(_previous_questions(history)).lower()
    grouped: dict[tuple[str, str], list[dict]] = {}
    for chunk in index.chunks:
        if week and chunk["week"] != week:
            continue
        if source and chunk["source"] != source:
            continue
        section = chunk.get("section") or "Overview"
        if _SKIP_SECTION.search(section) or len(chunk.get("text") or "") < 240:
            continue
        grouped.setdefault((chunk["source"], section), []).append(chunk)

    fresh = []
    for chunks in grouped.values():
        label = f"{chunks[0]['title']} {chunks[0].get('section') or ''}".lower()
        if previous and _already_covered(label, previous):
            continue
        fresh.append(chunks)
    pool = fresh or list(grouped.values())
    if not pool:
        return []
    chosen = max(random.choice(pool), key=lambda chunk: len(chunk["text"]))
    neighbors = [chunk for chunk in index.neighbors(chosen["id"]) if len(chunk.get("text") or "") >= 80]
    return neighbors or [chosen]


def _already_covered(label: str, previous: str) -> bool:
    words = [word for word in re.findall(r"[a-z]{4,}", label) if word not in _SECTION_STOP]
    return any(word in previous for word in words)


def _retrieval_query(question: str, history: list[dict]) -> str:
    previous = ""
    for message in reversed(history):
        if message["role"] == "assistant":
            previous = message["content"]
            break
    if not previous:
        return question
    return f"{previous[-800:]}\n{question}"


def _visible_history(history: list[dict]) -> list[dict]:
    cleaned = []
    for message in history[-8:]:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:4000]})
    return cleaned


def _parse_call(call: dict) -> tuple[str, dict]:
    function = call.get("function") or call
    name = function.get("name") or ""
    args = function.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _ollama_stream(messages: list[dict], tools: list[dict] | None, temperature: float = 0.2):
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": 1400},
    }
    if tools:
        payload["tools"] = tools
    yield from _post_stream(payload)


def _post_stream(payload: dict):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=600)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if "think" in detail and "think" in payload:
            payload = dict(payload)
            payload.pop("think", None)
            yield from _post_stream(payload)
            return
        raise RuntimeError(detail[:500] or exc.reason) from exc

    with response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            event = json.loads(line)
            message = event.get("message") or {}
            delta = {}
            content = message.get("content") or ""
            if content:
                delta["content"] = content
            if message.get("tool_calls"):
                delta["tool_calls"] = message["tool_calls"]
            if delta:
                yield delta
            if event.get("done"):
                break


class _SourceBook:
    def __init__(self):
        self.items: list[dict] = []
        self.numbers: dict[str, int] = {}

    def add(self, chunks: list[dict]) -> list[dict]:
        added = []
        for chunk in chunks:
            chunk_id = chunk["id"]
            if chunk_id in self.numbers:
                existing = next(item for item in self.items if item["id"] == chunk_id)
                added.append(existing)
                continue
            number = len(self.items) + 1
            item = dict(chunk)
            item["n"] = number
            item.pop("embed_text", None)
            self.items.append(item)
            self.numbers[chunk_id] = number
            added.append(item)
        return added

    def public(self) -> list[dict]:
        exported = []
        for item in self.items:
            text = item["text"].strip()
            excerpt = text if len(text) <= 700 else text[:700].rstrip() + "…"
            exported.append(
                {
                    "n": item["n"],
                    "id": item["id"],
                    "label": cite_label(item),
                    "week": item["week"],
                    "title": item["title"],
                    "source": item["source"],
                    "section": item.get("section") or "",
                    "page_start": item["page_start"],
                    "page_end": item["page_end"],
                    "excerpt": excerpt,
                }
            )
        return exported
