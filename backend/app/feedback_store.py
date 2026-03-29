"""
feedback_store.py
Persistance SQLite des sessions HITL avec scores LLM-as-judge.

Schéma :
  - sessions       : une ligne par conversation (thread_id)
  - feedback_turns : une ligne par tour draft/feedback/eval

Usage :
    store = FeedbackStore()
    store.init_db()
    store.log_turn(thread_id, turn_data)
    store.finalize_session(thread_id, "approved", final_response)
    df = store.to_dataframe()   # pour analyse / fine-tuning
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from app.evaluator import EvalScores


# DB path — à côté du dossier backend, configurable via env
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "hitl_feedback.db"
DB_PATH = Path(os.getenv("HITL_DB_PATH", str(_DEFAULT_DB_PATH)))


# --- Schema SQL ---

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id       TEXT PRIMARY KEY,
    human_request   TEXT NOT NULL,
    final_response  TEXT,
    outcome         TEXT,          -- 'approved' | 'abandoned'
    total_turns     INTEGER DEFAULT 0,
    avg_confidence  REAL,
    created_at      TEXT NOT NULL,
    finalized_at    TEXT
);
"""

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS feedback_turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id       TEXT NOT NULL,
    turn_number     INTEGER NOT NULL,
    draft           TEXT NOT NULL,
    human_comment   TEXT,          -- NULL si premier tour ou approbation
    human_action    TEXT,          -- 'feedback' | 'approved' | NULL (en attente)
    score_coherence REAL,
    score_tone      REAL,
    score_feedback  REAL,          -- NULL si pas de feedback humain
    score_confidence REAL,
    eval_rationale  TEXT,
    eval_scores_json TEXT,         -- JSON complet pour archivage
    created_at      TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES sessions(thread_id)
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_turns_thread ON feedback_turns(thread_id);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_outcome ON sessions(outcome);",
    "CREATE INDEX IF NOT EXISTS idx_turns_action ON feedback_turns(human_action);",
]


# --- Store class ---

class FeedbackStore:
    """Gère la persistance SQLite des sessions HITL."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")  # safe pour accès concurrent
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        """Crée les tables si elles n'existent pas."""
        with self._connect() as conn:
            conn.execute(_CREATE_SESSIONS)
            conn.execute(_CREATE_TURNS)
            for idx in _CREATE_INDEXES:
                conn.execute(idx)

    def create_session(self, thread_id: str, human_request: str):
        """Ouvre une nouvelle session au démarrage d'une conversation."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (thread_id, human_request, created_at)
                VALUES (?, ?, ?)
                """,
                (thread_id, human_request, _now()),
            )

    def log_turn(
        self,
        thread_id: str,
        turn_number: int,
        draft: str,
        scores: EvalScores,
        human_comment: Optional[str] = None,
    ) -> int:
        """
        Enregistre un tour : draft + scores du judge.
        Retourne l'id du tour créé.
        """
        scores_dict = scores.model_dump()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback_turns
                    (thread_id, turn_number, draft, human_comment,
                     score_coherence, score_tone, score_feedback, score_confidence,
                     eval_rationale, eval_scores_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    turn_number,
                    draft,
                    human_comment,
                    scores.coherence,
                    scores.tone_clarity,
                    scores.feedback_respect,
                    scores.confidence,
                    scores.rationale,
                    json.dumps(scores_dict),
                    _now(),
                ),
            )
            return cursor.lastrowid

    def record_human_action(
        self,
        thread_id: str,
        turn_number: int,
        action: str,                    # 'approved' | 'feedback'
        human_comment: Optional[str] = None,
    ):
        """Met à jour le tour avec la décision humaine."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE feedback_turns
                SET human_action = ?, human_comment = COALESCE(?, human_comment)
                WHERE thread_id = ? AND turn_number = ?
                """,
                (action, human_comment, thread_id, turn_number),
            )

    def finalize_session(
        self,
        thread_id: str,
        outcome: str,                   # 'approved' | 'abandoned'
        final_response: Optional[str] = None,
    ):
        """Clôture une session après approbation ou abandon."""
        with self._connect() as conn:
            # Calcul avg confidence sur tous les tours
            row = conn.execute(
                "SELECT AVG(score_confidence) as avg_c, COUNT(*) as cnt FROM feedback_turns WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            avg_conf = row["avg_c"] if row else None
            total = row["cnt"] if row else 0

            conn.execute(
                """
                UPDATE sessions
                SET outcome = ?, final_response = ?, total_turns = ?,
                    avg_confidence = ?, finalized_at = ?
                WHERE thread_id = ?
                """,
                (outcome, final_response, total, avg_conf, _now(), thread_id),
            )

    def get_session_stats(self) -> dict:
        """Statistiques globales — utile pour un dashboard."""
        with self._connect() as conn:
            stats = conn.execute("""
                SELECT
                    COUNT(*) as total_sessions,
                    SUM(CASE WHEN outcome = 'approved' THEN 1 ELSE 0 END) as approved,
                    AVG(total_turns) as avg_turns,
                    AVG(avg_confidence) as avg_confidence
                FROM sessions WHERE outcome IS NOT NULL
            """).fetchone()

            first_turn_approved = conn.execute("""
                SELECT COUNT(*) as cnt FROM sessions s
                WHERE s.outcome = 'approved' AND s.total_turns = 1
            """).fetchone()

            total = stats["total_sessions"] or 0
            approved = stats["approved"] or 0
            return {
                "total_sessions": total,
                "approved": approved,
                "approval_rate": round(approved / max(total, 1) * 100, 1),
                "avg_turns_before_approval": round(stats["avg_turns"] or 0, 2),
                "avg_confidence_score": round(stats["avg_confidence"] or 0, 2),
                "first_turn_approval_count": first_turn_approved["cnt"] or 0,
            }

    def get_dataset_for_finetuning(self) -> list[dict]:
        """
        Exporte les données au format (prompt, draft, feedback, score)
        pour fine-tuning ou reward modeling.
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT
                    s.human_request,
                    t.draft,
                    t.human_comment,
                    t.human_action,
                    t.score_coherence,
                    t.score_tone,
                    t.score_feedback,
                    t.score_confidence,
                    t.eval_rationale,
                    t.turn_number
                FROM feedback_turns t
                JOIN sessions s ON t.thread_id = s.thread_id
                WHERE t.human_action IS NOT NULL
                ORDER BY s.created_at DESC, t.turn_number ASC
            """).fetchall()
            return [dict(r) for r in rows]


# --- Singleton ---
# Une instance partagée par toute l'app
_store: Optional[FeedbackStore] = None

def get_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore()
        _store.init_db()
    return _store


# --- Helpers ---
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Exports ---
__all__ = ["FeedbackStore", "get_store", "EvalScores"]
