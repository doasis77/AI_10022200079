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
    .block-container { padding-top: 1.1rem; }
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
    .section-kicker {
        color: #8ea0cf;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0.1rem 0 0.35rem;
    }
    .answer-card {
        border: 1px solid #24423b;
        background: linear-gradient(180deg, #0f1e1a 0%, #0d1916 100%);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.55rem;
    }
    .answer-title {
        color: #baf7d7;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .result-meta {
        color: #9fb0d8;
        font-size: 0.86rem;
        margin-bottom: 0.45rem;
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
    st.caption("Tune retrieval, prompting, and analysis controls.")
    with st.expander("Retrieval", expanded=True):
        top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=8, value=4)
        chunking_method = st.selectbox(
            "PDF chunking method",
            options=["paragraph_aware", "fixed_size"],
            index=0
        )
    with st.expander("Generation", expanded=True):
        prompt_version = st.selectbox(
            "Prompt version",
            options=["v3", "v2", "v1"],
            index=0,
            help="v1=basic, v2=hallucination-controlled, v3=structured (recommended)"
        )
        compare_pure_llm = st.checkbox("Compare with pure LLM (no retrieval)", value=False)
    with st.expander("Diagnostics", expanded=False):
        show_debug = st.checkbox("Show debug panel", value=False)
        run_eval_btn = st.button("🧪 Run Evaluation Suite", use_container_width=True)


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


# ── Home with Get Started flow ─────────────────────────────────────────────────
if "started" not in st.session_state:
    st.session_state.started = False
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
regions = get_election_regions()
default_region = regions[0] if regions else "Greater Accra"

if not st.session_state.started:
    st.markdown(
        """
        <div class="welcome-card">
          <div class="welcome-title">Welcome to Academic City Assistant</div>
          <div class="welcome-sub">Ask grounded questions about Ghana election and budget datasets.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Indexed Chunks", n_chunks)
    c2.metric("Regions Available", len(regions))
    c3.metric("Chunking Mode", chunking_method.replace("_", " ").title())
    st.markdown("### What you can do")
    st.markdown(
        "- Explore election performance by region and year.\n"
        "- Compare policy priorities in the budget statement.\n"
        "- Run RAG vs pure-LLM comparisons from the sidebar."
    )
    st.markdown("### Try these starter questions")
    st.info(
        "• Which party performed best in Greater Accra?\n\n"
        "• Summarize the top 3 priorities in the 2025 budget.\n\n"
        "• Compare education and health budget allocations."
    )
    if st.button("Get Started", type="primary"):
        st.session_state.started = True
        st.session_state.query_input = "Which party performed best in Greater Accra?"
        st.rerun()
else:
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
        st.session_state.last_prompt_version = None
        st.session_state.last_compare_pure = False

    # Keep final answer pinned at the top when available.
    action_left, action_right = st.columns([8, 2])
    with action_left:
        st.markdown('<div class="section-kicker">Conversation</div>', unsafe_allow_html=True)
    with action_right:
        if st.button("Clear answer", use_container_width=True):
            st.session_state.last_result = None
            st.session_state.query_input = ""
            st.rerun()

    if st.session_state.last_result:
        r = st.session_state.last_result
        st.markdown(
            """
            <div class="answer-card">
              <div class="answer-title">Final Answer</div>
              <div class="result-meta">Grounded using retrieved context and current prompt settings.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success(r["response"])
        if st.session_state.last_compare_pure and r.get("pure_llm_response"):
            st.markdown('<div class="section-kicker">Baseline Comparison</div>', unsafe_allow_html=True)
            st.subheader("🔄 Pure LLM Response (no retrieval)")
            st.info(r["pure_llm_response"])
            st.caption(
                "Compare the two responses above. The RAG response is grounded "
                "in the provided documents; the pure LLM response uses only model knowledge."
            )

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
            placeholder="Type a grounded question and press Enter...",
            label_visibility="collapsed",
        )
    with send_col:
        submit = st.button(
            "Send",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.query_input.strip()
        )

    query = st.session_state.query_input.strip()
    if submit and query:
        with st.spinner("Running RAG pipeline…"):
            result = run_pipeline(
                query=query,
                retriever=retriever,
                top_k=top_k,
                prompt_version=prompt_version,
                compare_pure_llm=compare_pure_llm
            )
        st.session_state.last_result = result
        st.session_state.last_prompt_version = prompt_version
        st.session_state.last_compare_pure = compare_pure_llm
        st.rerun()

    if st.session_state.last_result:
        r = st.session_state.last_result
        with st.expander("ℹ️ Additional Information", expanded=False):
            st.markdown(
                f"**Query type detected:** `{r['query_type']}` | "
                f"**Prompt version:** `{st.session_state.last_prompt_version}`"
            )
            st.caption(
                f"Retrieved {len(r['retrieved'])} candidates and selected {len(r['selected'])} chunks for final context."
            )
            st.subheader("📄 Retrieved Chunks & Scores")
            for i, (chunk, scores) in enumerate(r["selected"]):
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
                st.code(r["prompt"], language="text")
            if show_debug:
                with st.expander("🛠 Debug Info", expanded=False):
                    st.json({
                        "query_type":        r["query_type"],
                        "prompt_version":    st.session_state.last_prompt_version,
                        "chunking_method":   chunking_method,
                        "top_k":             top_k,
                        "total_retrieved":   len(r["retrieved"]),
                        "total_selected":    len(r["selected"]),
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

if st.session_state.started:
    # ── Evaluation mode ────────────────────────────────────────────────────────
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

    # ── Log viewer ─────────────────────────────────────────────────────────────
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
st.caption(
    "CS4241 Introduction to Artificial Intelligence | Academic City University | "
    "Amegah Sewenam Kwame Bill | 10022200079"
)
