# Knowledge Graph RAG

## What the project does

This is a basic Knowledge Graph RAG application. It stores embedded text chunks in Neo4j, retrieves up to five relevant chunks, keeps conversations separate with conversation IDs, and displays grounded answers through Streamlit.

Users paste text, then ask questions about the stored information.

## Assignment requirements

The project meets the main requirements by:

1. Storing documents, chunks, and embeddings in Neo4j.
2. Returning up to five relevant chunks with similarity scores.
3. Filtering retrieval with `conversation_id` so conversations remain separate.
4. Combining chunk evidence with connected graph facts to generate a grounded answer.

## Architecture

```text
Text
        |
        v
     Chunking
        |
        v
    Embeddings
        |
        v
      Neo4j
        |
        v
Top-five vector retrieval
        |
        v
Connected graph facts
        |
        v
  Grounded answer
```

The graph connects conversations, documents, chunks, and entities using relationships such as `HAS_DOCUMENT`, `HAS_CHUNK`, `MENTIONS`, and `RELATED_TO`.

## Technologies used

- Python
- FastAPI
- Streamlit
- Neo4j AuraDB
- Groq
- OpenRouter
- Sentence Transformers

## Installation

```powershell
git clone https://github.com/Aaditya-Tripathi/GraphRAG.git
cd GraphRAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your Neo4j AuraDB credentials. Add a Groq key,
an OpenRouter key, or both. The provider can be selected in the UI.
OpenRouter uses the free `openai/gpt-oss-20b:free` model.
Then create the required Neo4j constraints and vector index:

```powershell
python -m scripts.setup_database
```

## Running the backend and UI

Start FastAPI in the first terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

Start Streamlit in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run ui\streamlit_app.py
```

Open:

- Streamlit: http://localhost:8501
- API documentation: http://127.0.0.1:8000/docs

## Basic usage

1. Open the Streamlit interface.
2. Use the generated conversation ID or enter your own.
3. Enter a document title and paste the text to ingest.
4. Click **Ingest information**.
5. Open **Ask questions** and enter a question.
6. Review the answer, supporting chunks, similarity scores, graph visualization, and connected facts.

Example source text and questions are available in `sample_data/`.

Run the focused tests with:

```powershell
python -m pytest -q
```

## Limitations

- File uploads are not currently supported; documents must be pasted as text.
- Free-model availability and rate limits can vary by provider.
