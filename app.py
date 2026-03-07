"""
Enterprise Document Intelligence Platform
-------------------------------------------
Streamlit UI Application

Layout:
  Left Sidebar  → Platform branding + Clear chat
  Right Panel   → Chatbot interface only

Run:
  streamlit run app.py
"""

import streamlit as st
from ui.styles import get_styles
from ui.chat import run_rag_pipeline, stream_answer

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Enterprise Document Intelligence Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ────────────────────────────────────────────────
st.markdown(get_styles(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_app" not in st.session_state:
    st.session_state.rag_app = None


# ══════════════════════════════════════════════════════════════════════
# LOAD RAG APP ONCE
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_rag_app():
    """Builds and caches the LangGraph RAG graph — loaded once."""
    import sys
    import os
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    from rag.graph import build_rag_graph
    app = build_rag_graph()
    sys.stdout = old_stdout
    return app


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — LEFT PANEL
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Logo + Title ─────────────────────────────────────────────────
    st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
            <div style="width:44px; height:44px; background:#1a73e8; border-radius:10px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:22px;">🏢</div>
            <div class="platform-title">Enterprise<br>Document<br>Intelligence<br>Platform</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Platform Info ─────────────────────────────────────────────────
    st.markdown("""
        <div style="font-size:13px; color:#666; padding: 0 4px; line-height:1.8;">
            <b>Supported Domains:</b><br>
            🟢 HR &nbsp;&nbsp; 🔵 IT &nbsp;&nbsp; 🟡 Finance &nbsp;&nbsp; 🟠 Operations
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Clear Chat Button ─────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Guardrail Notice ──────────────────────────────────────────────
    st.markdown("""
        <div style="font-size:12px; color:#999; padding: 0 4px; line-height:1.7;">
            🔒 <b>Guardrails Active</b><br>
            PII detection & profanity filter enabled on all queries.
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN PANEL — CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════

# ── Chat Header ──────────────────────────────────────────────────────
st.markdown(
    "<div class='chat-header'>Hi, How Can I Help? 👋</div>",
    unsafe_allow_html=True,
)

# ── Render Chat History ───────────────────────────────────────────────
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    citations = message.get("citations", [])

    if role == "user":
        st.markdown(
            f"""
            <div class="user-bubble">
                <div class="user-bubble-inner">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif role == "assistant":
        st.markdown(
            f"""
            <div class="bot-bubble">
                <div class="bot-bubble-inner">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Render Citations ──────────────────────────────────────────
        if citations:
            citation_text = " &nbsp;|&nbsp; ".join(
                [f"📄 {c}" for c in citations]
            )
            st.markdown(
                f'<div class="citation-box">📚 Sources: {citation_text}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════
# CHAT INPUT
# ══════════════════════════════════════════════════════════════════════
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])

    with col1:
        user_input = st.text_input(
            label="Message",
            placeholder="Type your message here...",
            label_visibility="collapsed",
        )

    with col2:
        send_clicked = st.form_submit_button("Send →")


# ══════════════════════════════════════════════════════════════════════
# HANDLE SEND
# ══════════════════════════════════════════════════════════════════════
if send_clicked and user_input.strip():

    # ── Add user message to history ───────────────────────────────────
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip(),
    })

    # ── Render user bubble immediately ────────────────────────────────
    st.markdown(
        f"""
        <div class="user-bubble">
            <div class="user-bubble-inner">{user_input.strip()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load RAG app if not already loaded ────────────────────────────
    if st.session_state.rag_app is None:
        with st.spinner("Initializing AI engine..."):
            st.session_state.rag_app = load_rag_app()

    # ── Run RAG pipeline ──────────────────────────────────────────────
    with st.spinner("Thinking..."):
        result = run_rag_pipeline(
            question=user_input.strip(),
            rag_app=st.session_state.rag_app,
        )

    answer = result["answer"]
    citations = result.get("citations", [])

    # ── Stream bot answer ─────────────────────────────────────────────
    with st.container():
        st.markdown(
            '<div class="bot-bubble"><div class="bot-bubble-inner">',
            unsafe_allow_html=True,
        )
        st.write_stream(stream_answer(answer))
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Render citations (only for non-guardrail responses) ───────────
    if citations and not result.get("guardrail_triggered"):
        citation_text = " &nbsp;|&nbsp; ".join([f"📄 {c}" for c in citations])
        st.markdown(
            f'<div class="citation-box">📚 Sources: {citation_text}</div>',
            unsafe_allow_html=True,
        )

    # ── Save to history ───────────────────────────────────────────────
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })

    st.rerun()


# ══════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; margin-top:60px; color:#9aa0a6;">
            <div style="font-size:48px;">💬</div>
            <div style="font-size:16px; margin-top:12px;">
                Ask a question about your enterprise documents.
            </div>
            <div style="font-size:13px; margin-top:6px; color:#bbb;">
                Supports HR, IT, Finance and Operations domains.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
