# Lesson 1: Basic Blocking API — enrichie avec LLM-as-judge + feedback store

from fastapi import APIRouter
from uuid import uuid4
from app.models import StartRequest, GraphResponse, ResumeRequest, EvalScoresResponse, StatsResponse
from app.graph import graph
from app.feedback_store import get_store

router = APIRouter()


def _build_response(result, config) -> GraphResponse:
    """Construit la GraphResponse en incluant les scores du judge si disponibles."""
    state = graph.get_state(config)
    next_nodes = state.next
    thread_id = config["configurable"]["thread_id"]

    run_status = "user_feedback" if (next_nodes and "human_feedback" in next_nodes) else "finished"

    # Extraire les scores du judge depuis le state
    eval_scores = None
    values = state.values
    if values.get("eval_scores"):
        eval_scores = EvalScoresResponse(**values["eval_scores"])

    return GraphResponse(
        thread_id=thread_id,
        run_status=run_status,
        assistant_response=result.get("assistant_response") if result else values.get("assistant_response"),
        eval_scores=eval_scores,
        revision_count=values.get("revision_count", 0),
    )


@router.post("/graph/start", response_model=GraphResponse)
def start_graph(request: StartRequest):
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    # On injecte thread_id dans le state pour que feedback_logger puisse l'utiliser
    initial_state = {
        "human_request": request.human_request,
        "thread_id": thread_id,
        "revision_count": 0,
    }
    result = graph.invoke(initial_state, config)
    return _build_response(result, config)


@router.post("/graph/resume", response_model=GraphResponse)
def resume_graph(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    state_update = {"status": request.review_action}
    if request.human_comment is not None:
        state_update["human_comment"] = request.human_comment

    graph.update_state(config, state_update)
    result = graph.invoke(None, config)
    return _build_response(result, config)


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """Statistiques globales HITL — taux d'approbation, nb tours moyen, etc."""
    store = get_store()
    return StatsResponse(**store.get_session_stats())


@router.get("/dataset")
def get_dataset():
    """Export du dataset de feedback pour fine-tuning / analyse."""
    store = get_store()
    return {"records": store.get_dataset_for_finetuning()}
