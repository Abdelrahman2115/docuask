# DocuAsk

A Retrieval-Augmented Generation (RAG) assistant that answers questions about
U.S. FCC telecom consumer guides and cites the exact source passages as
`[file, page]`. It says "I don't know" when the documents don't contain the answer.

> Work in progress. Architecture, results and demo are added in later phases.

## Quick start (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
copy .env.example .env      # then add your GROQ_API_KEY
python scripts/check_setup.py
pytest
pip freeze > requirements.lock.txt
```

## Project layout

```
app/          Gradio UI (Phase 7)
config.yaml   All non-secret settings (models, chunking, top-k)
data/raw/     Source documents (public-domain FCC guides)
data/index/   Vector index (generated, not committed)
notebooks/    Exploration and results analysis
scripts/      CLI entry points (setup check, ingest, build index)
src/docuask/  Library code
tests/        Unit tests
```
