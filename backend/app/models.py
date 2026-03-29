from pydantic import BaseModel
from typing import Optional, Literal


# --- Start Graph Run ---
class StartRequest(BaseModel):
    human_request: str

# --- Resume Paused Graph Run ---
class ResumeRequest(BaseModel):
    thread_id: str
    review_action: Literal["approved", "feedback"]
    human_comment: Optional[str] = None

# --- Eval scores exposés dans la réponse ---
class EvalScoresResponse(BaseModel):
    coherence: float
    tone_clarity: float
    feedback_respect: Optional[float] = None
    confidence: float
    rationale: str

# --- API Response (enrichie avec les scores du judge) ---
class GraphResponse(BaseModel):
    thread_id: str
    run_status: Literal["finished", "user_feedback", "pending"]
    assistant_response: Optional[str] = None
    eval_scores: Optional[EvalScoresResponse] = None   # nouveau
    revision_count: int = 0                            # nouveau

# --- MCP approval ---
class ApproveRequest(BaseModel):
    thread_id: str
    approve_action: Literal["approved", "rejected"]

# --- Stats endpoint ---
class StatsResponse(BaseModel):
    total_sessions: int
    approved: int
    approval_rate: float
    avg_turns_before_approval: float
    avg_confidence_score: float
    first_turn_approval_count: int
