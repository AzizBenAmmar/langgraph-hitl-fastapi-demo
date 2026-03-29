# Lesson 2: Advanced Streaming API with SSE — enrichie avec scores du judge

from fastapi import APIRouter, Request
from uuid import uuid4
from app.models import StartRequest, GraphResponse, ResumeRequest
from app.graph import graph
from sse_starlette.sse import EventSourceResponse
import json

router = APIRouter()

# In-memory storage for run configurations
run_configs = {}


@router.post("/graph/stream/create", response_model=GraphResponse)
def create_graph_streaming(request: StartRequest):
    thread_id = str(uuid4())

    run_configs[thread_id] = {
        "type": "start",
        "human_request": request.human_request,
        "thread_id": thread_id,
    }

    return GraphResponse(
        thread_id=thread_id,
        run_status="pending",
        assistant_response=None,
    )


@router.post("/graph/stream/resume", response_model=GraphResponse)
def resume_graph_streaming(request: ResumeRequest):
    thread_id = request.thread_id

    run_configs[thread_id] = {
        "type": "resume",
        "review_action": request.review_action,
        "human_comment": request.human_comment,
    }

    return GraphResponse(
        thread_id=thread_id,
        run_status="pending",
        assistant_response=None,
    )


@router.get("/graph/stream/{thread_id}")
async def stream_graph(request: Request, thread_id: str):
    if thread_id not in run_configs:
        return {"error": "Thread ID not found. Call /graph/stream/create or /graph/stream/resume first."}

    run_data = run_configs[thread_id]
    config = {"configurable": {"thread_id": thread_id}}

    input_state = None
    if run_data["type"] == "start":
        event_type = "start"
        input_state = {
            "human_request": run_data["human_request"],
            "thread_id": thread_id,
            "revision_count": 0,
        }
    else:
        event_type = "resume"
        state_update = {"status": run_data["review_action"]}
        if run_data.get("human_comment") is not None:
            state_update["human_comment"] = run_data["human_comment"]
        graph.update_state(config, state_update)

    async def event_generator():
        initial_data = json.dumps({"thread_id": thread_id})
        yield {"event": event_type, "data": initial_data}

        try:
            for msg, metadata in graph.stream(input_state, config, stream_mode="messages"):
                if await request.is_disconnected():
                    break

                node = metadata.get("langgraph_node")

                # Streamer les tokens des nodes de génération
                if node in ["assistant_draft", "assistant_finalize"]:
                    if msg.content:
                        token_data = json.dumps({"content": msg.content})
                        yield {"event": "token", "data": token_data}

                # Le node evaluator ne génère pas de tokens publics
                # (il utilise structured output, pas de streaming)

            # Après le stream, récupérer les scores du judge depuis le state
            state = graph.get_state(config)
            values = state.values

            eval_payload = None
            if values.get("eval_scores"):
                eval_payload = {
                    "eval_scores": values["eval_scores"],
                    "eval_rationale": values.get("eval_rationale", ""),
                    "revision_count": values.get("revision_count", 0),
                }
                yield {"event": "eval", "data": json.dumps(eval_payload)}

            # Status final
            if state.next and "human_feedback" in state.next:
                yield {"event": "status", "data": json.dumps({"status": "user_feedback"})}
            else:
                yield {"event": "status", "data": json.dumps({"status": "finished"})}

            if thread_id in run_configs:
                del run_configs[thread_id]

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
            if thread_id in run_configs:
                del run_configs[thread_id]

    return EventSourceResponse(event_generator())
