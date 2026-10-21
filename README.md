# Exam prep

Local question-answering over the Week 1 and Week 2 course PDFs.

The pipeline:

1. Read each PDF in reading order, keeping headings and code blocks.
2. Split the text into passages tied to a document, section, and page.
3. Index every passage two ways: keyword search (BM25) and local embeddings (`BAAI/bge-small-en-v1.5`).
4. A tutor retrieves the best passages, can search again, and answers with the local Ollama model. Claims are cited back to the page.

## Run it

From this folder, with Ollama running and `gemma4:26b` pulled:

```bash
source .venv/bin/activate
python -m examrag ingest
python -m examrag serve
```

Then open http://127.0.0.1:8765

Other commands:

```bash
python -m examrag search "what is polymorphism"
python -m examrag ask "what is the difference between inheritance and polymorphism"
python -m examrag ask --mode outline "SDLC phases"
```

Drop new PDFs anywhere in this folder (a `Week 3` folder works) and run `python -m examrag ingest` again, or use Rebuild index in the study desk.

Set `EXAMRAG_MODEL` if you want a different Ollama model. Set `OLLAMA_HOST` if Ollama is not on `http://127.0.0.1:11434`.
