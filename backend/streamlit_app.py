"""
streamlit_app.py
Interface Streamlit pour le HITL assistant — Option B (tout embarqué).

Lance avec :
    streamlit run streamlit_app.py

depuis le dossier backend/.
"""

import sys
import os
import csv
import io
from uuid import uuid4
from pathlib import Path

import streamlit as st

# --- Path setup : permet d'importer app.* depuis backend/ ---
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

from app.graph import graph, DraftReviewState
from app.evaluator import EvalScores
from app.feedback_store import get_store

# ---------------------------------------------------------------------------
# Config page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HITL Assistant",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init():
    defaults = {
        "thread_id": None,
        "history": [],           # list of {"role": "user"|"assistant", "content": str}
        "eval_scores": None,
        "eval_rationale": None,
        "revision_count": 0,
        "ui_state": "idle",      # idle | waiting | feedback_form | finished
        "assistant_response": "",
        "question": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ---------------------------------------------------------------------------
# Helpers — LangGraph calls
# ---------------------------------------------------------------------------

def _get_state_values(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    return state.values, state.next


def _run_graph(input_state: dict | None, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(input_state, config)
    return result


def start_conversation(question: str):
    thread_id = str(uuid4())
    st.session_state.thread_id = thread_id
    st.session_state.history = [{"role": "user", "content": question}]
    st.session_state.eval_scores = None
    st.session_state.revision_count = 0

    initial_state = {
        "human_request": question,
        "thread_id": thread_id,
        "revision_count": 0,
    }
    _run_graph(initial_state, thread_id)

    values, next_nodes = _get_state_values(thread_id)
    _update_session_from_state(values)


def resume_conversation(action: str, comment: str | None = None):
    thread_id = st.session_state.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    update = {"status": action}
    if comment:
        update["human_comment"] = comment
    graph.update_state(config, update)

    if comment:
        st.session_state.history.append({"role": "user", "content": comment})

    _run_graph(None, thread_id)
    values, next_nodes = _get_state_values(thread_id)
    _update_session_from_state(values)

    if action == "approved" and not next_nodes:
        st.session_state.ui_state = "finished"


def _update_session_from_state(values: dict):
    st.session_state.assistant_response = values.get("assistant_response", "")
    st.session_state.revision_count = values.get("revision_count", 0)

    if values.get("eval_scores"):
        st.session_state.eval_scores = values["eval_scores"]
        st.session_state.eval_rationale = values.get("eval_rationale", "")

    # Mettre à jour l'historique avec la dernière réponse assistant
    history = st.session_state.history
    response = values.get("assistant_response", "")
    if response:
        if history and history[-1]["role"] == "assistant":
            history[-1]["content"] = response
        else:
            history.append({"role": "assistant", "content": response})

    _, next_nodes = _get_state_values(st.session_state.thread_id)
    if next_nodes and "human_feedback" in next_nodes:
        st.session_state.ui_state = "idle"
    elif not next_nodes:
        st.session_state.ui_state = "finished"


def reset():
    for k in ["thread_id", "history", "eval_scores", "eval_rationale",
              "revision_count", "assistant_response", "question"]:
        st.session_state[k] = None if k in ["thread_id", "eval_scores", "eval_rationale"] else \
                               [] if k == "history" else \
                               0 if k == "revision_count" else ""
    st.session_state.ui_state = "idle"


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_score_bar(label: str, value: float | None, key: str):
    if value is None:
        return
    color = "#2e7d32" if value >= 7 else "#e65100" if value >= 4 else "#c62828"
    pct = int(value / 10 * 100)
    st.markdown(
        f"""
        <div style="margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
            <span style="color:#555">{label}</span>
            <span style="font-weight:600;color:{color}">{value:.1f}/10</span>
          </div>
          <div style="background:#eee;border-radius:4px;height:7px;overflow:hidden">
            <div style="width:{pct}%;background:{color};height:100%;border-radius:4px;
                        transition:width 0.6s ease"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scorecard():
    scores = st.session_state.eval_scores
    if not scores:
        return

    rev = st.session_state.revision_count
    label = f"⚖️ LLM-as-judge {'— révision #' + str(rev) if rev > 0 else '— 1er tour'}"

    with st.expander(label, expanded=True):
        render_score_bar("Cohérence avec la demande", scores.get("coherence"), "coh")
        render_score_bar("Ton et clarté", scores.get("tone_clarity"), "tone")
        if scores.get("feedback_respect") is not None:
            render_score_bar("Respect du feedback", scores.get("feedback_respect"), "fb")
        render_score_bar("Score de confiance global", scores.get("confidence"), "conf")

        rationale = st.session_state.eval_rationale
        if rationale:
            st.markdown(
                f"<div style='font-size:12px;color:#666;font-style:italic;"
                f"border-top:1px solid #eee;padding-top:8px;margin-top:8px'>"
                f"{rationale}</div>",
                unsafe_allow_html=True,
            )


def render_history():
    for msg in st.session_state.history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])


def render_stats_sidebar():
    store = get_store()
    stats = store.get_session_stats()

    st.sidebar.title("📊 Statistiques HITL")
    st.sidebar.metric("Sessions totales", stats["total_sessions"])
    st.sidebar.metric("Taux d'approbation", f"{stats['approval_rate']}%")
    st.sidebar.metric("Tours moyens", f"{stats['avg_turns_before_approval']:.1f}")
    st.sidebar.metric("Confiance moyenne", f"{stats['avg_confidence_score']:.1f}/10")
    st.sidebar.metric("Approuvé au 1er tour", stats["first_turn_approval_count"])

    st.sidebar.divider()

    # Export CSV
    dataset = store.get_dataset_for_finetuning()
    if dataset:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=dataset[0].keys())
        writer.writeheader()
        writer.writerows(dataset)
        st.sidebar.download_button(
            label="⬇️ Exporter le dataset (CSV)",
            data=output.getvalue(),
            file_name="hitl_feedback_dataset.csv",
            mime="text/csv",
        )
    else:
        st.sidebar.caption("Aucune donnée à exporter pour l'instant.")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🔁 Human-in-the-Loop Assistant")
st.caption("LangGraph · LLM-as-judge · Feedback store SQLite")

# Sidebar stats
render_stats_sidebar()

# Bouton reset
col_title, col_reset = st.columns([6, 1])
with col_reset:
    if st.button("🔄 Reset", use_container_width=True):
        reset()
        st.rerun()

# ---------------------------------------------------------------------------
# STATE : idle — pas encore de conversation
# ---------------------------------------------------------------------------
if st.session_state.ui_state == "idle" and not st.session_state.history:
    question = st.chat_input("Posez une question...")
    if question:
        st.session_state.question = question
        st.session_state.ui_state = "waiting"
        st.rerun()

# ---------------------------------------------------------------------------
# STATE : waiting — on appelle le graph
# ---------------------------------------------------------------------------
elif st.session_state.ui_state == "waiting":
    render_history()
    with st.spinner("L'assistant réfléchit..."):
        try:
            if not st.session_state.thread_id:
                start_conversation(st.session_state.question)
            else:
                resume_conversation("feedback", st.session_state.get("pending_feedback"))
                st.session_state.pending_feedback = None
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.session_state.ui_state = "idle"
    st.rerun()

# ---------------------------------------------------------------------------
# STATE : idle — en attente de décision humaine (après un draft)
# ---------------------------------------------------------------------------
elif st.session_state.ui_state == "idle" and st.session_state.history:
    render_history()
    render_scorecard()

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approuver", use_container_width=True, type="primary"):
            with st.spinner("Finalisation..."):
                try:
                    resume_conversation("approved")
                except Exception as e:
                    st.error(f"Erreur : {e}")
            st.rerun()

    with col2:
        if st.button("✏️ Donner un feedback", use_container_width=True):
            st.session_state.ui_state = "feedback_form"
            st.rerun()

# ---------------------------------------------------------------------------
# STATE : feedback_form
# ---------------------------------------------------------------------------
elif st.session_state.ui_state == "feedback_form":
    render_history()
    render_scorecard()

    st.divider()
    feedback = st.text_area(
        "Votre feedback pour améliorer la réponse :",
        placeholder="Ex : Rends la réponse plus courte, max 3 lignes...",
        height=100,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📨 Envoyer le feedback", use_container_width=True, type="primary"):
            if feedback.strip():
                st.session_state.pending_feedback = feedback.strip()
                st.session_state.ui_state = "waiting"
                st.rerun()
            else:
                st.warning("Écris quelque chose avant d'envoyer.")
    with col2:
        if st.button("Annuler", use_container_width=True):
            st.session_state.ui_state = "idle"
            st.rerun()

# ---------------------------------------------------------------------------
# STATE : finished
# ---------------------------------------------------------------------------
elif st.session_state.ui_state == "finished":
    # Afficher tout l'historique sauf le dernier assistant (affiché séparément)
    for msg in st.session_state.history[:-1]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    st.success("✨ Version finale approuvée")
    with st.container(border=True):
        st.markdown(st.session_state.assistant_response)

    st.divider()
    if st.button("🔄 Nouvelle conversation", type="primary"):
        reset()
        st.rerun()
