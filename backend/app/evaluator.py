"""
evaluator.py — LLM-as-judge via HuggingFace Inference API.
Utilise Qwen2.5-72B avec structured output JSON manuel (HF ne supporte pas with_structured_output).
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
import json
import re


class EvalScores(BaseModel):
    coherence: float = Field(ge=0, le=10)
    tone_clarity: float = Field(ge=0, le=10)
    feedback_respect: Optional[float] = Field(default=None, ge=0, le=10)
    confidence: float = Field(ge=0, le=10)
    rationale: str


# Judge model — température 0 pour évaluation déterministe
_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.01,
)
_judge = ChatHuggingFace(llm=_endpoint)


def evaluate_draft(
    human_request: str,
    draft: str,
    human_comment: Optional[str] = None,
    revision_count: int = 0,
) -> EvalScores:

    feedback_section = (
        f'\n### Feedback humain (tour {revision_count}) :\n"""{human_comment}"""\n'
        if human_comment else
        "\n### Aucun feedback — premier tour.\n"
    )

    prompt = f"""Tu es un évaluateur expert en qualité de réponses IA.
Évalue ce brouillon et réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ou après.

### Demande :
\"\"\"{human_request}\"\"\"

### Brouillon :
\"\"\"{draft}\"\"\"
{feedback_section}

Format JSON requis (scores de 0 à 10) :
{{
  "coherence": <float>,
  "tone_clarity": <float>,
  "feedback_respect": <float ou null si pas de feedback>,
  "confidence": <float>,
  "rationale": "<2-3 phrases d'explication>"
}}"""

    response = _judge.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Extraire le JSON même si le modèle ajoute du texte autour
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        data = json.loads(raw)
        return EvalScores(**data)
    except Exception:
        # Fallback si le parsing échoue
        return EvalScores(
            coherence=7.0,
            tone_clarity=7.0,
            feedback_respect=None,
            confidence=7.0,
            rationale="Évaluation automatique indisponible — score par défaut.",
        )


__all__ = ["evaluate_draft", "EvalScores"]
