# app.py
# Author: Amegah Sewenam Kwame Bill | Index: 10022200079
# Academic City RAG Chatbot — CS4241 Exam Project
# Streamlit application entry point

import os
import json
import glob
import streamlit as st
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# ── Path helpers ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH    = os.path.join(DATA_DIR, "Ghana_Election_Result.csv")


def _resolve_budget_pdf_path(data_dir: str) -> str | None:
    """Pick a budget PDF: env override, legacy fixed name, or first *Budget*.pdf in data/."""
    env = os.environ.get("BUDGET_PDF_PATH", "").strip()
    if env and os.path.isfile(env):
        return env
    legacy = os.path.join(data_dir, "2025_Budget_Statement.pdf")
    if os.path.isfile(legacy):
        return legacy
    matches = sorted(
        glob.glob(os.path.join(data_dir, "*Budget*.pdf"))
        + glob.glob(os.path.join(data_dir, "*budget*.pdf"))
    )
    seen = set()
    for p in matches:
        if p not in seen:
            seen.add(p)
            return p
    return None


PDF_PATH = _resolve_budget_pdf_path(DATA_DIR)
CHUNKS_PATH = os.path.join(OUTPUT_DIR, "chunks.json")


# ── Imports (after path setup) ────────────────────────────────────────────────
from src.ingestion.load_csv    import load_csv
from src.ingestion.load_pdf    import load_pdf
from src.preprocessing.clean_csv  import clean_dataframe, rows_to_chunks
from src.preprocessing.clean_pdf  import clean_pages
from src.preprocessing.chunking   import fixed_size_chunks, paragraph_aware_chunks
from src.retrieval.embedder        import embed_texts
from src.retrieval.vector_store    import VectorStore
from src.retrieval.bm25_retriever  import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pipeline.rag_pipeline     import run_pipeline
from src.utils.logger              import get_logs


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACity RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

st.markdown(
    """
    <style>
    .app-shell {
        border: 1px solid #20263a;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: linear-gradient(180deg, #0d1322 0%, #0b111e 100%);
        margin-bottom: 0.75rem;
    }
    .app-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
    .title-wrap { display: flex; align-items: center; gap: 0.65rem; }
    .assistant-avatar {
        width: 34px; height: 34px; border-radius: 999px;
        background: radial-gradient(circle at 30% 30%, #5ca8ff, #2f59d4);
        display: flex; align-items: center; justify-content: center; font-size: 18px;
        box-shadow: 0 0 0 1px #304a96;
    }
    .assistant-name { color: #f5f7ff; font-weight: 700; font-size: 1.05rem; }
    .assistant-sub { color: #9da8c8; font-size: 0.86rem; margin-top: 0.1rem; }
    .online-pill {
        border: 1px solid #2a3f2f; background: #101a14; color: #b8f5c6;
        border-radius: 999px; padding: 0.2rem 0.6rem; font-size: 0.8rem;
    }
    .welcome-card {
        border: 1px solid #1f2638; border-radius: 12px; padding: 0.8rem 0.9rem;
        background: #0a1020; margin-bottom: 0.4rem;
    }
    .welcome-title { color: #edf1ff; font-weight: 650; margin-bottom: 0.15rem; }
    .welcome-sub { color: #a5b0d0; font-size: 0.92rem; }
    .chip-note { color: #8793b8; font-size: 0.84rem; margin: 0.2rem 0 0.45rem; }
    div[data-testid="stPills"] [role="radiogroup"] {
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 0.25rem;
    }
    div[data-testid="stPills"] button {
        min-height: 1.8rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.82rem;
        border-color: #334160;
        color: #e7ecff;
        background: #121a2e;
    }
    .empty-state {
        border: 1px dashed #27314b; border-radius: 12px; min-height: 180px;
        display: flex; align-items: center; justify-content: center; text-align: center;
        background: #0a1121; color: #aab6d7; margin-top: 0.75rem;
    }
    .empty-state .icon { font-size: 1.5rem; margin-bottom: 0.25rem; }
    .empty-state .title { color: #e7ebff; font-weight: 650; }
    div[data-testid="stTextInput"] input {
        border: 1px solid #2b3550 !important;
        background: #0e1527 !important;
        color: #f3f6ff !important;
        border-radius: 10px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #4f7fff !important;
        box-shadow: 0 0 0 0.18rem rgba(79, 127, 255, 0.18) !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(180deg, #4b7dff, #3f66d8);
        border: 1px solid #567dff;
        color: #ffffff;
        font-weight: 600;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        border-color: #7ea0ff;
        box-shadow: 0 0 0 0.1rem rgba(110, 151, 255, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-shell">
      <div class="app-header">
        <div class="title-wrap">
          <div class="assistant-avatar">🎓</div>
          <div>
            <div class="assistant-name">Academic City Assistant — Ghana Data Chat</div>
            <div class="assistant-sub">Election trends + 2025 budget Q&A</div>
          </div>
        </div>
        <div class="online-pill">● Online</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=8, value=4)
    prompt_version = st.selectbox(
        "Prompt version",
        options=["v3", "v2", "v1"],
        index=0,
        help="v1=basic, v2=hallucination-controlled, v3=structured (recommended)"
    )
    chunking_method = st.selectbox(
        "PDF Chunking method",
        options=["paragraph_aware", "fixed_size"],
        index=0
    )
    compare_pure_llm = st.checkbox("Compare with pure LLM (no retrieval)", value=False)
    st.markdown("---")
    show_debug = st.checkbox("Show debug panel", value=False)
    run_eval_btn = st.button("🧪 Run Evaluation Suite")


# ── System initialisation (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading and indexing documents…")
def init_system(chunking_method: str):
    """
    Full setup pipeline:
      1. Load & clean CSV + PDF
      2. Chunk documents
      3. Save chunks.json
      4. Generate embeddings
      5. Build FAISS index
      6. Build BM25 index
    """
    all_chunks = []

    # ── CSV ──────────────────────────────────────────────────────────────────
    if os.path.exists(CSV_PATH):
        df, _ = load_csv(CSV_PATH)
        df     = clean_dataframe(df)
        csv_chunks = rows_to_chunks(df, source="election")
        all_chunks.extend(csv_chunks)
    else:
        st.warning(f"CSV not found at {CSV_PATH}. Skipping election data.")

    # ── PDF ──────────────────────────────────────────────────────────────────
    if PDF_PATH and os.path.isfile(PDF_PATH):
        raw_pages     = load_pdf(PDF_PATH)
        cleaned_pages = clean_pages(raw_pages)
        if chunking_method == "fixed_size":
            pdf_chunks = fixed_size_chunks(cleaned_pages)
        else:
            pdf_chunks = paragraph_aware_chunks(cleaned_pages)
        all_chunks.extend(pdf_chunks)
    else:
        st.warning(
            "No budget PDF found. Put a file matching *Budget*.pdf in data/, rename to "
            "2025_Budget_Statement.pdf, or set BUDGET_PDF_PATH in .env to the full path."
        )

    if not all_chunks:
        st.error("No data loaded. Please add your data files to the /data directory.")
        st.stop()

    # Save chunks
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2, default=str)

    # ── Embeddings ────────────────────────────────────────────────────────────
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # ── FAISS ─────────────────────────────────────────────────────────────────
    dim = embeddings.shape[1]
    vs = VectorStore(dim=dim)
    vs.add(embeddings, all_chunks)

    # ── BM25 ──────────────────────────────────────────────────────────────────
    bm25 = BM25Retriever()
    bm25.build(all_chunks)

    retriever = HybridRetriever(vector_store=vs, bm25=bm25)
    return retriever, len(all_chunks)


retriever, n_chunks = init_system(chunking_method)

st.sidebar.success(f"✅ {n_chunks} chunks indexed")

GHANA_REGIONS = [
    "Ahafo", "Ashanti", "Bono", "Bono East", "Central", "Eastern", "Greater Accra",
    "North East", "Northern", "Oti", "Savannah", "Upper East", "Upper West",
    "Volta", "Western", "Western North",
]

# Region helper for scoped quick prompts
@st.cache_data(show_spinner=False)
def get_election_regions() -> list[str]:
    csv_regions = set()
    if os.path.isfile(CSV_PATH):
        df, _ = load_csv(CSV_PATH)
        df = clean_dataframe(df)
        if "region" in df.columns:
            vals = [str(v).strip() for v in df["region"].dropna().tolist()]
            vals = [v for v in vals if v and v.lower() not in ("nan", "none")]
            csv_regions = set(vals)
    return sorted(set(GHANA_REGIONS).union(csv_regions))


# ── Query interface ────────────────────────────────────────────────────────────
regions = get_election_regions()
default_region = regions[0] if regions else "Greater Accra"

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

st.markdown(
    """
    <div class="welcome-card">
      <div class="welcome-title">Hi! Ask me about Ghana's election and budget data.</div>
      <div class="welcome-sub">Use a quick suggestion or type your own question below.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

region_choice = st.selectbox("Region filter", options=regions or [default_region], index=0)

quick_suggestions = [
    "Who won the most parliamentary seats overall?",
    "Summarize the top 3 priorities in the 2025 budget.",
    f"Which party performed best in {region_choice}?",
    f"What was the voter turnout trend in {region_choice}?",
    "Compare budget allocations for education and health.",
    "List key election insights from the dataset.",
]

st.markdown('<div class="chip-note">Quick suggestions</div>', unsafe_allow_html=True)
if hasattr(st, "pills"):
    picked = st.pills(
        "Suggestion chips",
        options=quick_suggestions,
        label_visibility="collapsed",
    )
else:
    # Streamlit fallback for versions without st.pills (e.g., some 1.x builds).
    picked = st.selectbox(
        "Suggestion chips",
        options=[""] + quick_suggestions,
        index=0,
        format_func=lambda x: "Select a quick suggestion..." if x == "" else x,
        label_visibility="collapsed",
    )
if picked:
    st.session_state.query_input = picked

input_col, send_col = st.columns([7, 1.2], vertical_alignment="bottom")
with input_col:
    st.text_input(
        "Ask your question",
        key="query_input",
        placeholder="e.g. How many seats did the NDC win in parliament?",
        label_visibility="collapsed",
    )
with send_col:
    submit = st.button("Send", type="primary", use_container_width=True)

query = st.session_state.query_input.strip()

if submit and query.strip():
    with st.spinner("Running RAG pipeline…"):
        result = run_pipeline(
            query=query.strip(),
            retriever=retriever,
            top_k=top_k,
            prompt_version=prompt_version,
            compare_pure_llm=compare_pure_llm
        )

    st.subheader("🤖 Final Answer")
    st.success(result["response"])

    if compare_pure_llm and result.get("pure_llm_response"):
        st.subheader("🔄 Pure LLM Response (no retrieval)")
        st.info(result["pure_llm_response"])
        st.caption(
            "Compare the two responses above. The RAG response is grounded "
            "in the provided documents; the pure LLM response uses only model knowledge."
        )

    with st.expander("ℹ️ Additional Information", expanded=False):
        st.markdown(
            f"**Query type detected:** `{result['query_type']}` | "
            f"**Prompt version:** `{prompt_version}`"
        )

        st.subheader("📄 Retrieved Chunks & Scores")
        for i, (chunk, scores) in enumerate(result["selected"]):
            with st.expander(
                f"Chunk {i+1}: {chunk['chunk_id']} | Source: {chunk['source']} | "
                f"Final Score: {scores['final_score']:.4f}"
            ):
                st.markdown(f"**Text preview:**\n\n{chunk['text'][:500]}…")
                st.markdown("**Score breakdown:**")
                score_cols = st.columns(4)
                score_cols[0].metric("Vector", f"{scores['vector_score']:.4f}")
                score_cols[1].metric("BM25",   f"{scores['bm25_score']:.4f}")
                score_cols[2].metric("Source bonus", f"{scores['source_bonus']:.4f}")
                score_cols[3].metric("KW bonus", f"{scores['keyword_bonus']:.4f}")
                if chunk.get("year"):
                    st.caption(f"Year detected: {chunk['year']}")
                if chunk.get("keywords"):
                    st.caption(f"Keywords: {', '.join(chunk['keywords'][:6])}")

        with st.expander("🔍 View Final Prompt Sent to LLM", expanded=False):
            st.code(result["prompt"], language="text")

        if show_debug:
            with st.expander("🛠 Debug Info", expanded=False):
                st.json({
                    "query_type":        result["query_type"],
                    "prompt_version":    prompt_version,
                    "chunking_method":   chunking_method,
                    "top_k":             top_k,
                    "total_retrieved":   len(result["retrieved"]),
                    "total_selected":    len(result["selected"]),
                })
else:
    st.markdown(
        """
        <div class="empty-state">
          <div>
            <div class="icon">💬</div>
            <div class="title">No conversation yet</div>
            <div>Pick a quick suggestion or enter a question to start.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Evaluation mode ────────────────────────────────────────────────────────────
if run_eval_btn:
    st.subheader("🧪 Evaluation Results")
    with st.spinner("Running evaluation suite (this may take a few minutes)…"):
        from src.evaluation.run_evaluation import run_evaluation
        eval_results = run_evaluation(retriever)

    st.success(
        f"Evaluation complete! "
        f"Avg keyword hit rate: {eval_results['summary']['avg_keyword_hit_rate']} | "
        f"Hallucination risk: {eval_results['summary']['hallucination_rate']}"
    )

    tab1, tab2 = st.tabs(["Factual Queries", "Adversarial Queries"])
    with tab1:
        for r in eval_results["factual_results"]:
            with st.expander(f"Q: {r['query']}"):
                st.markdown(f"**RAG Answer:** {r['rag_response']}")
                if r.get("pure_llm_response"):
                    st.markdown(f"**Pure LLM:** {r['pure_llm_response']}")
                st.json(r["rag_score"])

    with tab2:
        for r in eval_results["adversarial_results"]:
            with st.expander(f"[{r['type'].upper()}] {r['query']}"):
                st.markdown(f"**Expected behavior:** _{r['expected_behavior']}_")
                st.markdown(f"**Response:** {r['rag_response']}")
                if r["insufficient"]:
                    st.success("✅ System correctly returned 'insufficient information' response")
                else:
                    st.warning("⚠️ System did not return fallback phrase — review for hallucination")

# ── Log viewer ─────────────────────────────────────────────────────────────────
with st.expander("📋 View Query Logs"):
    logs = get_logs()
    if logs:
        st.write(f"{len(logs)} logged queries.")
        for log in reversed(logs[-5:]):   # Show 5 most recent
            st.json({
                "timestamp":    log.get("timestamp"),
                "query":        log.get("query"),
                "query_type":   log.get("query_type"),
                "response":     log.get("response", "")[:300]
            })
    else:
        st.info("No queries logged yet.")

st.markdown("---")
st.caption("CS4241 Introduction to Artificial Intelligence | Academic City University | [Your Name] | [Index Number]")
