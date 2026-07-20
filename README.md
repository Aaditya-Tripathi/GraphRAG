# Knowledge Graph RAG

## What the project does

Knowledge Graph RAG is a basic question-answering application that combines vector search with a knowledge graph.

Users add text to a conversation and ask questions about it. The application divides the text into chunks, creates embeddings, stores the data in Neo4j, extracts entities and relationships, and generates an answer supported by the stored information.

The Streamlit interface displays the answer, relevant chunks, similarity scores, connected facts, and an interactive graph visualization.

## How it meets the project requirements

1. Documents, chunks, and vector embeddings are stored in Neo4j.
2. A question retrieves up to five relevant chunks and their similarity scores.
3. Every operation is filtered by `conversation_id`, keeping conversations separate.
4. Retrieved chunks and connected graph facts are combined to generate a grounded answer.

## Architecture

```text
Text
  → cleaning and chunking
  → sentence-transformer embeddings
  → Neo4j vector and graph storage
  → top-five vector retrieval
  → connected entities and graph facts
  → grounded LLM answer
```

The stored graph uses the following main structure:

```text
Conversation → Document → Chunk → Entity → Entity
```

Relationships include `HAS_DOCUMENT`, `HAS_CHUNK`, `NEXT_CHUNK`, `MENTIONS`, and `RELATED_TO`.

## Technologies used

- Python
- FastAPI
- Streamlit
- Neo4j AuraDB
- Sentence Transformers
- Groq and OpenRouter
- Pytest

## Installation

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/Aaditya-Tripathi/GraphRAG.git
cd GraphRAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and enter your Neo4j AuraDB credentials. Add a Groq API key, an OpenRouter API key, or both. Never commit the completed `.env` file.

Create the Neo4j constraints and vector index:

```powershell
python -m scripts.setup_database
```

The focused test suite can be run with:

```powershell
python -m pytest -q
```

## Running the backend and UI

Start the FastAPI backend in the first terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

Start Streamlit in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run ui\streamlit_app.py
```

Open the Streamlit application at `http://localhost:8501`.

FastAPI documentation is available at `http://127.0.0.1:8000/docs`.

## Basic usage

1. Start the backend and Streamlit interface.
2. Keep the generated conversation ID or enter your own.
3. Select Groq or OpenRouter as the language-model provider.
4. Enter a document title and paste the source text.
5. Select **Ingest information** to build the vector and knowledge graph data.
6. Open **Ask questions** and enter a question about the stored text.
7. Review the grounded answer, supporting chunks, similarity scores, graph facts, and interactive graph.
8. Create a different conversation ID when you want an isolated collection of information.

Example text and questions are provided in `sample_data/`.

## Limitations

- The current version accepts pasted text only; file uploads are not included.
- Free LLM models can be slower or temporarily unavailable because of provider rate limits.
