import os
from typing import Any
from uuid import uuid4

import httpx
import streamlit as st

from app.constants import (
    MAX_CONVERSATION_ID_LENGTH,
    MAX_DOCUMENT_CHARACTERS,
    MAX_DOCUMENT_TITLE_LENGTH,
    MAX_QUESTION_LENGTH,
    MAX_TOP_K,
)
from ui.graph_visualization import render_graph_visualization


API_URL = os.getenv(
    "GRAPHRAG_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")
REQUEST_TIMEOUT = httpx.Timeout(300.0, connect=5.0)


st.set_page_config(
    page_title="Knowledge Graph RAG",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --kg-bg: #0b0e0c;
            --kg-panel: #131815;
            --kg-line: #2a342d;
            --kg-text: #f1f5ed;
            --kg-muted: #99a69d;
            --kg-accent: #c7ff4a;
            --kg-cyan: #6fe7dc;
        }
        .stApp {
            background:
                radial-gradient(circle at 85% 8%, rgba(111,231,220,.08), transparent 28rem),
                linear-gradient(rgba(199,255,74,.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(199,255,74,.018) 1px, transparent 1px),
                var(--kg-bg);
            background-size: auto, 32px 32px, 32px 32px, auto;
        }
        [data-testid="stHeader"] {
            background: rgba(11,14,12,.76);
            backdrop-filter: blur(12px);
        }
        [data-testid="stSidebar"] {
            background: #0e120f;
            border-right: 1px solid var(--kg-line);
        }
        h1, h2, h3 {
            font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
            letter-spacing: -.025em;
            text-wrap: balance;
        }
        h1 {
            max-width: 11ch;
            font-size: clamp(2.4rem, 5vw, 4.8rem) !important;
            line-height: .95 !important;
        }
        p, label, button, input, textarea {
            font-family: Aptos, "Segoe UI", sans-serif;
        }
        .kg-kicker {
            display: inline-flex;
            align-items: center;
            gap: .55rem;
            margin-bottom: .9rem;
            color: var(--kg-accent);
            font-size: .72rem;
            font-weight: 750;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .kg-kicker::before {
            width: 1.8rem;
            height: 1px;
            background: var(--kg-accent);
            content: "";
        }
        .kg-intro {
            max-width: 48rem;
            margin: -.35rem 0 2rem;
            color: var(--kg-muted);
            font-size: 1.05rem;
            line-height: 1.7;
            text-wrap: pretty;
        }
        .kg-live-dot {
            display: inline-block;
            width: .52rem;
            height: .52rem;
            margin-right: .45rem;
            border-radius: 50%;
            background: var(--kg-accent);
            box-shadow: 0 0 .8rem rgba(199,255,74,.65);
        }
        [data-testid="stForm"] {
            padding: 1.4rem 1.4rem .5rem;
            border: 1px solid var(--kg-line);
            border-radius: .45rem;
            background: rgba(19,24,21,.82);
        }
        [data-testid="stMetric"] {
            min-height: 7.2rem;
            padding: 1rem 1.1rem;
            border: 1px solid var(--kg-line);
            border-radius: .35rem;
            background: var(--kg-panel);
        }
        [data-testid="stMetricValue"] {
            color: var(--kg-accent);
            font-family: "Iowan Old Style", Georgia, serif;
            font-variant-numeric: tabular-nums;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
            border-bottom: 1px solid var(--kg-line);
        }
        .stTabs [data-baseweb="tab"] {
            padding: .75rem 1rem;
            border-radius: .3rem .3rem 0 0;
            color: var(--kg-muted);
        }
        .stTabs [aria-selected="true"] {
            color: var(--kg-text) !important;
            background: var(--kg-panel) !important;
        }
        .stButton > button,
        [data-testid="stFormSubmitButton"] > button {
            border-radius: .25rem;
            border-color: #3a463e;
            font-weight: 700;
            transition: transform .2s ease, border-color .2s ease;
        }
        .stButton > button:active,
        [data-testid="stFormSubmitButton"] > button:active {
            transform: translateY(1px);
        }
        [data-testid="stFormSubmitButton"] > button[kind="primary"] {
            color: #10140f;
            background: var(--kg-accent);
            border-color: var(--kg-accent);
        }
        [data-testid="stExpander"] {
            border-color: var(--kg-line);
            border-radius: .35rem;
            background: rgba(19,24,21,.62);
        }
        code { color: var(--kg-cyan) !important; }
        hr { border-color: var(--kg-line) !important; }
        @media (max-width: 760px) {
            h1 { font-size: 2.7rem !important; }
            .kg-intro { font-size: .96rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_conversation_id() -> str:
    return f"conversation-{uuid4().hex[:12]}"


def initialize_state() -> None:
    defaults = {
        "conversation_id": create_conversation_id(),
        "llm_provider": "openrouter",
        "last_ingestion": None,
        "last_query": None,
        "backend_health": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation_results() -> None:
    st.session_state.last_ingestion = None
    st.session_state.last_query = None


def parse_api_response(response: httpx.Response) -> dict[str, Any]:
    """Return JSON from FastAPI or raise a readable UI error."""

    if response.is_error:
        try:
            error_data = response.json()
        except ValueError:
            detail: Any = response.text or response.reason_phrase
        else:
            detail = (
                error_data.get("detail", error_data)
                if isinstance(error_data, dict)
                else error_data
            )

        raise RuntimeError(
            f"API returned HTTP {response.status_code}: {detail}"
        )

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            "The API returned an invalid JSON response."
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError("The API returned an unexpected response.")

    return data


def call_api(
    method: str,
    endpoint: str,
    json_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.request(
                method=method,
                url=f"{API_URL}{endpoint}",
                json=json_data,
            )
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "The request timed out while waiting for the backend."
        ) from error
    except httpx.RequestError as error:
        raise RuntimeError(
            f"Could not reach the backend at {API_URL}."
        ) from error

    return parse_api_response(response)


def render_ingestion_result(data: dict[str, Any]) -> None:
    st.success("Information was stored successfully.")

    graph_summary = data.get("knowledge_graph", {})
    columns = st.columns(4)
    columns[0].metric("Chunks", data.get("chunks_created", 0))
    columns[1].metric(
        "Embedding dimensions",
        data.get("embedding_dimension", 0),
    )
    columns[2].metric(
        "Entities extracted",
        graph_summary.get("entities_extracted", 0),
    )
    columns[3].metric(
        "Graph facts",
        graph_summary.get("facts_processed", 0),
    )


def render_supporting_chunks(
    results: list[dict[str, Any]],
) -> None:
    st.subheader(f"Supporting chunks ({len(results)})")

    if not results:
        st.info("No supporting chunks were returned.")
        return

    for index, result in enumerate(results, start=1):
        score = float(result.get("score", 0.0))
        document_title = (
            result.get("document_title") or "Untitled document"
        )
        heading = (
            f"Chunk {index} · score {score:.4f} · {document_title}"
        )

        with st.expander(heading, expanded=index == 1):
            metadata_columns = st.columns(3)
            metadata_columns[0].write(
                f"**Stored chunk index:** "
                f"{result.get('chunk_index', 'unknown')}"
            )
            metadata_columns[1].write(
                f"**Similarity score:** {score:.4f}"
            )
            metadata_columns[2].write(
                f"**Document:** {document_title}"
            )
            st.markdown("**Chunk text**")
            st.write(result.get("text", ""))
            st.caption(f"Chunk ID: {result.get('chunk_id', '')}")


def render_graph_facts(facts: list[dict[str, Any]]) -> None:
    st.subheader(f"Connected graph facts ({len(facts)})")

    if not facts:
        st.info("No connected graph facts were returned.")
        return

    for index, fact in enumerate(facts, start=1):
        confidence = float(fact.get("confidence", 0.0))
        st.markdown(
            f"**Fact {index}:** `{fact.get('source', 'Unknown')}` "
            f"— **{fact.get('predicate', 'RELATED_TO')}** → "
            f"`{fact.get('target', 'Unknown')}`"
        )
        st.caption(
            f"Confidence: {confidence:.2f} · Source chunk: "
            f"{fact.get('source_chunk_id', '')}"
        )
        if fact.get("evidence"):
            with st.expander(
                f"Evidence for Fact {index}",
                expanded=False,
            ):
                st.write(fact["evidence"])


def render_entities(entities: list[dict[str, Any]]) -> None:
    with st.expander(
        f"Connected entities ({len(entities)})",
        expanded=False,
    ):
        if not entities:
            st.write("No entities were returned.")
            return

        st.dataframe(
            [
                {
                    "Name": entity.get("name", ""),
                    "Type": entity.get("type", ""),
                    "Normalized name": entity.get(
                        "normalized_name",
                        "",
                    ),
                }
                for entity in entities
            ],
            width="stretch",
            hide_index=True,
        )


initialize_state()
inject_styles()

st.markdown(
    '<div class="kg-kicker">Evidence graph console</div>',
    unsafe_allow_html=True,
)
st.title("Knowledge Graph RAG")
st.markdown(
    """
    <div class="kg-intro">
        Store source material in Neo4j, retrieve its most relevant
        passages, traverse connected facts, and generate answers that
        point back to their evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Conversation")

    if st.button("Create new conversation", width="stretch"):
        st.session_state.conversation_id = create_conversation_id()
        clear_conversation_results()
        st.rerun()

    conversation_id = st.text_input(
        "Conversation ID",
        key="conversation_id",
        max_chars=MAX_CONVERSATION_ID_LENGTH,
        help=(
            "Documents and questions using the same ID belong "
            "to the same conversation."
        ),
        on_change=clear_conversation_results,
    )
    st.caption(
        "Keep this ID unchanged when ingesting and querying "
        "related information."
    )

    st.selectbox(
        "AI provider",
        options=["openrouter", "groq"],
        format_func=lambda value: (
            "OpenRouter (GPT-OSS 20B Free)"
            if value == "openrouter"
            else "Groq"
        ),
        key="llm_provider",
        help=(
            "The selected provider is used for knowledge "
            "extraction and answer generation."
        ),
        on_change=clear_conversation_results,
    )
    llm_provider = st.session_state.llm_provider
    st.divider()

    if st.button("Check backend connection", width="stretch"):
        st.session_state.backend_health = None
        try:
            with st.spinner("Checking FastAPI and Neo4j..."):
                st.session_state.backend_health = call_api(
                    "GET",
                    "/health",
                )
        except RuntimeError as error:
            st.error(str(error))

    if st.session_state.backend_health:
        health = st.session_state.backend_health
        st.success("Backend is connected.")
        st.markdown(
            '<span class="kg-live-dot"></span>Live services',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Neo4j: {health.get('neo4j')}  \n"
            f"Embedding: {health.get('embedding_model')}  \n"
            f"Groq: {health.get('groq_model')}  \n"
            f"OpenRouter: {health.get('openrouter_model')}"
        )


ingestion_tab, query_tab = st.tabs(
    ["Ingest information", "Ask questions"]
)


with ingestion_tab:
    st.header("Add information to the knowledge graph")

    with st.form("information_ingestion_form"):
        document_title = st.text_input(
            "Document title",
            max_chars=MAX_DOCUMENT_TITLE_LENGTH,
            placeholder="Example: Neo4j GraphRAG Notes",
        )
        document_text = st.text_area(
            "Document text",
            height=320,
            max_chars=MAX_DOCUMENT_CHARACTERS,
            placeholder=(
                "Paste information that should be stored "
                "and queried..."
            ),
        )

        ingest_submitted = st.form_submit_button(
            "Ingest information",
            type="primary",
            width="stretch",
        )

    if ingest_submitted:
        st.session_state.last_ingestion = None

        if not conversation_id.strip():
            st.error("A conversation ID is required.")
        elif not document_title.strip():
            st.error("A document title is required.")
        elif not document_text.strip():
            st.error("Document text is required.")
        else:
            try:
                with st.spinner(
                    "Chunking, embedding, and building "
                    "the knowledge graph..."
                ):
                    result = call_api(
                        "POST",
                        "/api/ingest/text",
                        {
                            "conversation_id": conversation_id,
                            "document_title": document_title,
                            "text": document_text,
                            "provider": llm_provider,
                        },
                    )

                st.session_state.last_ingestion = result
                st.session_state.last_query = None
            except RuntimeError as error:
                st.error(str(error))

    ingestion_result = st.session_state.last_ingestion
    if (
        ingestion_result
        and ingestion_result.get("conversation_id") == conversation_id
    ):
        render_ingestion_result(ingestion_result)


with query_tab:
    st.header("Query the stored information")

    with st.form("query_form"):
        question = st.text_area(
            "Question",
            height=120,
            max_chars=MAX_QUESTION_LENGTH,
            placeholder=(
                "Ask a question about information stored in this "
                "conversation..."
            ),
        )
        top_k = st.slider(
            "Maximum supporting chunks",
            min_value=1,
            max_value=MAX_TOP_K,
            value=MAX_TOP_K,
        )
        query_submitted = st.form_submit_button(
            "Generate answer",
            type="primary",
            width="stretch",
        )

    if query_submitted:
        st.session_state.last_query = None

        if not conversation_id.strip():
            st.error("A conversation ID is required.")
        elif not question.strip():
            st.error("Enter a question.")
        else:
            try:
                with st.spinner(
                    "Retrieving chunks, traversing the graph, and "
                    "generating an answer..."
                ):
                    st.session_state.last_query = call_api(
                        "POST",
                        "/api/query",
                        {
                            "conversation_id": conversation_id,
                            "question": question,
                            "top_k": top_k,
                            "provider": llm_provider,
                        },
                    )
            except RuntimeError as error:
                st.error(str(error))

    response = st.session_state.last_query
    if response and response.get("conversation_id") == conversation_id:
        st.subheader("Grounded answer")
        st.markdown(response.get("answer", "No answer was returned."))
        st.divider()
        render_supporting_chunks(response.get("results", []))
        st.divider()
        st.subheader("Interactive knowledge graph")
        render_graph_visualization(
            response.get("entities", []),
            response.get("facts", []),
        )
        st.divider()
        render_graph_facts(response.get("facts", []))
        render_entities(response.get("entities", []))
