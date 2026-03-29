"""
graph.py — HITL graph avec LLM-as-judge et feedback store.
Provider : HuggingFace Inference API (langchain-huggingface)
"""

from typing import Literal, Optional
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from app.evaluator import evaluate_draft, EvalScores
from app.feedback_store import get_store

# --- Model HuggingFace ---
_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.4,
)
model = ChatHuggingFace(llm=_endpoint)


# --- Extended State ---
class DraftReviewState(MessagesState):
    human_request: str
    human_comment: Optional[str]
    status: Literal["approved", "feedback"]
    assistant_response: str
    eval_scores: Optional[dict]
    eval_rationale: Optional[str]
    revision_count: int
    thread_id: Optional[str]


# --- Node 1 : assistant_draft ---
def assistant_draft(state: DraftReviewState) -> DraftReviewState:
    user_message = HumanMessage(content=state["human_request"])
    status = state.get("status", "approved")
    revision_count = state.get("revision_count", 0)

    if status == "feedback" and state.get("human_comment"):
        system_message = SystemMessage(content=(
            f"""Tu es un assistant IA révisant ton brouillon précédent.

FEEDBACK REÇU : \"{state['human_comment']}\"

Intègre soigneusement ce feedback. NE répète PAS le feedback dans ta réponse.
"""
        ))
        messages = [user_message] + state["messages"] + [system_message]
        all_messages = state["messages"]
    else:
        system_message = SystemMessage(content=(
            "Tu es un assistant IA. Réponds de façon claire, pertinente et complète."
        ))
        messages = [system_message, user_message]
        all_messages = state["messages"]

    response = model.invoke(messages)
    all_messages = all_messages + [response]

    return {
        **state,
        "messages": all_messages,
        "assistant_response": response.content,
        "revision_count": revision_count + (1 if status == "feedback" else 0),
    }


# --- Node 2 : evaluator ---
def evaluator(state: DraftReviewState) -> DraftReviewState:
    scores: EvalScores = evaluate_draft(
        human_request=state["human_request"],
        draft=state["assistant_response"],
        human_comment=state.get("human_comment"),
        revision_count=state.get("revision_count", 0),
    )
    return {
        **state,
        "eval_scores": scores.model_dump(),
        "eval_rationale": scores.rationale,
    }


# --- Node 3 : human_feedback (pause) ---
def human_feedback(state: DraftReviewState):
    pass


# --- Node 4 : feedback_logger ---
def feedback_logger(state: DraftReviewState) -> DraftReviewState:
    store = get_store()
    thread_id = state.get("thread_id", "unknown")
    revision_count = state.get("revision_count", 0)
    status = state.get("status", "approved")

    store.create_session(thread_id, state["human_request"])

    if state.get("eval_scores"):
        scores = EvalScores(**state["eval_scores"])
        store.log_turn(
            thread_id=thread_id,
            turn_number=revision_count,
            draft=state["assistant_response"],
            scores=scores,
            human_comment=state.get("human_comment"),
        )
        store.record_human_action(
            thread_id=thread_id,
            turn_number=revision_count,
            action=status,
            human_comment=state.get("human_comment"),
        )
    return state


# --- Node 5 : assistant_finalize ---
def assistant_finalize(state: DraftReviewState) -> DraftReviewState:
    latest_response = state["assistant_response"]
    system_message = SystemMessage(content=(
        "L'utilisateur a approuvé ton brouillon. "
        "Apporte de légers ajustements de clarté et de ton. "
        "Ne l'étends pas significativement."
    ))
    user_message = HumanMessage(content=state["human_request"])
    assistant_message = HumanMessage(content=f"Mon brouillon : {latest_response}")
    response = model.invoke([system_message, user_message, assistant_message])
    all_messages = state["messages"] + [response]

    store = get_store()
    store.finalize_session(
        thread_id=state.get("thread_id", "unknown"),
        outcome="approved",
        final_response=response.content,
    )
    return {**state, "messages": all_messages, "assistant_response": response.content}


# --- Router ---
def feedback_router(state: DraftReviewState) -> str:
    return "assistant_finalize" if state["status"] == "approved" else "assistant_draft"


# --- Graph ---
builder = StateGraph(DraftReviewState)
builder.add_node("assistant_draft", assistant_draft)
builder.add_node("evaluator", evaluator)
builder.add_node("human_feedback", human_feedback)
builder.add_node("feedback_logger", feedback_logger)
builder.add_node("assistant_finalize", assistant_finalize)

builder.add_edge(START, "assistant_draft")
builder.add_edge("assistant_draft", "evaluator")
builder.add_edge("evaluator", "human_feedback")
builder.add_edge("human_feedback", "feedback_logger")
builder.add_conditional_edges("feedback_logger", feedback_router, {
    "assistant_finalize": "assistant_finalize",
    "assistant_draft": "assistant_draft",
})
builder.add_edge("assistant_finalize", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

__all__ = ["graph", "DraftReviewState"]
