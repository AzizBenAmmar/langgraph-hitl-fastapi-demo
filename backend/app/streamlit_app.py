"""
streamlit_app.py — HITL Assistant, Option B (tout embarqué).
Lance depuis backend/ : streamlit run streamlit_app.py
"""

import sys, os, csv, io
from uuid import uuid4
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from app.graph import graph
from app.evaluator import EvalScores
from app.feedback_store import get_store

# ---------------------------------------------------------------------------
st.set_page_config(page_title="HITL Assistant", page_icon="🔁", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "thread_id": None,
    "history": [],
    "eval_scores": None,
    "eval_rationale": None,
    "revision_count": 0,
    "ui_state": "idle",
    "assistant_response": "",
    "pending_question": "",
    "pending_feedback": "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# LangGraph helpers
# ---------------------------------------------------------------------------
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _is_waiting_feedback(thread_id: str) -> bool:
    state = graph.get_state(_config(thread_id))
    return bool(state.next and "human_feedback" in state.next)


def _start(question: str):
    thread_id = str(uuid4())
    st.session_state.thread_id = thread_id
    st.session_state.history = [{"role": "user", "content": question}]
    st.session_state.eval_scores = None
    st.session_state.revision_count = 0

    result = graph.invoke(
        {"human_request": question, "thread_id": thread_id, "revision_count": 0},
        _config(thread_id),
    )
    _apply(result)


def _resume(action: str, comment: str | None = None):
    thread_id = st.session_state.thread_id
    update = {"status": action}
    if comment:
        update["human_comment"] = comment
        st.session_state.history.append({"role": "user", "content": comment})

    graph.update_state(_config(thread_id), update)
    result = graph.invoke(None, _config(thread_id))
    _apply(result)

    if action == "approved" and not _is_waiting_feedback(thread_id):
        st.session_state.ui_state = "finished"


def _apply(result: dict):
    """Met à jour le session state depuis le résultat direct du graph.invoke()."""
    response = result.get("assistant_response", "")
    st.session_state.assistant_response = response

    if response:
        h = st.session_state.history
        if h and h[-1]["role"] == "assistant":
            h[-1]["content"] = response
        else:
            h.append({"role": "assistant", "content": response})

    if result.get("eval_scores"):
        st.session_state.eval_scores = result["eval_scores"]
        st.session_state.eval_rationale = result.get("eval_rationale", "")
        st.session_state.revision_count = result.get("revision_count", 0)

    if st.session_state.ui_state != "finished":
        if _is_waiting_feedback(st.session_state.thread_id):
            st.session_state.ui_state = "idle"


def _reset():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v, list) else []


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------
def score_bar(label: str, value: float | None):
    if value is None:
        return
    color = "#2e7d32" if value >= 7 else "#e65100" if value >= 4 else "#c62828"
    pct = int(value / 10 * 100)
    st.markdown(f"""
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
        <span style="color:#aaa">{label}</span>
        <span style="font-weight:600;color:{color}">{value:.1f}/10</span>
      </div>
      <div style="background:#333;border-radius:4px;height:7px">
        <div style="width:{pct}%;background:{color};height:100%;border-radius:4px"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def scorecard():
    s = st.session_state.eval_scores
    if not s:
        return
    rev = st.session_state.revision_count
    label = f"⚖️ LLM-as-judge {'— révision #' + str(rev) if rev > 0 else '— 1er tour'}"
    with st.expander(label, expanded=True):
        score_bar("Cohérence avec la demande", s.get("coherence"))
        score_bar("Ton et clarté", s.get("tone_clarity"))
        if s.get("feedback_respect") is not None:
            score_bar("Respect du feedback", s.get("feedback_respect"))
        score_bar("Score de confiance global", s.get("confidence"))
        if st.session_state.eval_rationale:
            st.markdown(
                f"<div style='font-size:12px;color:#888;font-style:italic;"
                f"border-top:1px solid #333;padding-top:8px;margin-top:8px'>"
                f"{st.session_state.eval_rationale}</div>",
                unsafe_allow_html=True,
            )


def history():
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def sidebar():
    store = get_store()
    stats = store.get_session_stats()
    st.sidebar.title("📊 Statistiques HITL")
    st.sidebar.metric("Sessions totales", stats["total_sessions"])
    st.sidebar.metric("Taux d'approbation", f"{stats['approval_rate']}%")
    st.sidebar.metric("Tours moyens", f"{stats['avg_turns_before_approval']:.1f}")
    st.sidebar.metric("Confiance moyenne", f"{stats['avg_confidence_score']:.1f}/10")
    st.sidebar.metric("Approuvé au 1er tour", stats["first_turn_approval_count"])
    st.sidebar.divider()

    dataset = store.get_dataset_for_finetuning()
    if dataset:
        buf = io.StringIO()
        csv.DictWriter(buf, fieldnames=dataset[0].keys()).writeheader() or \
            csv.DictWriter(buf, fieldnames=dataset[0].keys()).writerows(dataset)
        # réécrire proprement
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=dataset[0].keys())
        w.writeheader()
        w.writerows(dataset)
        st.sidebar.download_button(
            "⬇️ Exporter dataset (CSV)",
            data=buf.getvalue(),
            file_name="hitl_feedback.csv",
            mime="text/csv",
        )
    else:
        st.sidebar.caption("Aucune donnée à exporter.")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("🔁 Human-in-the-Loop Assistant")
st.caption("LangGraph · HuggingFace Qwen2.5-72B · LLM-as-judge · SQLite")

sidebar()

_, col_reset = st.columns([8, 1])
with col_reset:
    if st.button("🔄 Reset"):
        _reset()
        st.rerun()

# ---------------------------------------------------------------------------
# STATE MACHINE
# ---------------------------------------------------------------------------
state = st.session_state.ui_state

# — Idle sans historique : input initial —
if state == "idle" and not st.session_state.history:
    q = st.chat_input("Posez une question...")
    if q:
        st.session_state.pending_question = q
        st.session_state.ui_state = "waiting_start"
        st.rerun()

# — Lancer le graph (1er tour) —
elif state == "waiting_start":
    history()
    with st.spinner("L'assistant réfléchit..."):
        try:
            _start(st.session_state.pending_question)
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.session_state.ui_state = "idle"
    st.rerun()

# — Reprendre après feedback —
elif state == "waiting_feedback":
    history()
    with st.spinner("Révision en cours..."):
        try:
            _resume("feedback", st.session_state.pending_feedback)
            st.session_state.pending_feedback = ""
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.session_state.ui_state = "idle"
    st.rerun()

# — Reprendre après approbation —
elif state == "waiting_approve":
    history()
    with st.spinner("Finalisation..."):
        try:
            _resume("approved")
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.session_state.ui_state = "idle"
    st.rerun()

# — Idle avec historique : afficher réponse + actions —
elif state == "idle" and st.session_state.history:
    history()
    scorecard()
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approuver", use_container_width=True, type="primary"):
            st.session_state.ui_state = "waiting_approve"
            st.rerun()
    with c2:
        if st.button("✏️ Donner un feedback", use_container_width=True):
            st.session_state.ui_state = "feedback_form"
            st.rerun()

# — Formulaire de feedback —
elif state == "feedback_form":
    history()
    scorecard()
    st.divider()
    fb = st.text_area(
        "Votre feedback :",
        placeholder="Ex : Rends la réponse plus courte, max 3 lignes...",
        height=100,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📨 Envoyer", use_container_width=True, type="primary"):
            if fb.strip():
                st.session_state.pending_feedback = fb.strip()
                st.session_state.ui_state = "waiting_feedback"
                st.rerun()
            else:
                st.warning("Écris quelque chose avant d'envoyer.")
    with c2:
        if st.button("Annuler", use_container_width=True):
            st.session_state.ui_state = "idle"
            st.rerun()

# — Finished —
elif state == "finished":
    for msg in st.session_state.history[:-1]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    st.success("✨ Version finale approuvée")
    with st.container(border=True):
        st.markdown(st.session_state.assistant_response)
    st.divider()
    if st.button("🔄 Nouvelle conversation", type="primary"):
        _reset()
        st.rerun()
