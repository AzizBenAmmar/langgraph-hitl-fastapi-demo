# 🔁 LangGraph HITL — Production-Grade GenAI Pipeline

> **Fork & extension** of [esurovtsev/langgraph-hitl-fastapi-demo](https://github.com/esurovtsev/langgraph-hitl-fastapi-demo)  
> Extended with a **LLM-as-judge evaluation layer**, **persistent feedback store**, **HuggingFace Inference API** support, and a **Streamlit interface** — turning a demo into a production-ready GenAI feedback loop.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-purple)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Qwen2.5--72B-yellow?logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-red?logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-feedback--store-green?logo=sqlite)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📸 Screenshots

| HITL Chat + LLM-as-judge scorecard | Statistics dashboard & CSV export |
|---|---|
| ![HITL Flow](screenshots/hitl_flow.png) | ![Stats Panel](screenshots/stats_panel.png) |

---

## 🧠 What This Project Is About

**Human-in-the-Loop (HITL)** is a critical pattern in production GenAI systems — it allows a human to review, correct, and approve AI-generated content before it reaches end users.

This project goes beyond a basic HITL demo by adding what real production systems need:

- **Automated quality evaluation** via LLM-as-judge (not just human gut feeling)
- **Persistent feedback storage** to build fine-tuning datasets over time
- **Multi-turn revision tracking** with per-turn score history
- **Provider-agnostic LLM support** — drop OpenAI, plug in HuggingFace

---

## ✦ What I Added (vs. the Original)

The original project by [@esurovtsev](https://github.com/esurovtsev) is an excellent minimal HITL demo with FastAPI + React. Here is what I designed and implemented on top of it:

### 1. 🏛️ LLM-as-judge Evaluation Layer (`evaluator.py`)

A dedicated evaluation node integrated directly into the LangGraph graph. After every draft generation, the judge model scores the output on **4 criteria**:

| Criterion | Description |
|---|---|
| **Coherence** | Does the response address the user's request precisely? |
| **Tone & Clarity** | Is the response well-structured and appropriately toned? |
| **Feedback Respect** | Was the human's feedback correctly incorporated? (revision turns only) |
| **Global Confidence** | Is this draft ready for the end user? |

Each score is `0–10` with a textual rationale. Implemented using structured output (Pydantic `BaseModel`) for reliable JSON parsing, with a regex fallback for robustness.

### 2. 💾 Persistent Feedback Store (`feedback_store.py`)

A SQLite-based store that persists every HITL interaction — built for **fine-tuning dataset generation**:

```
sessions          → one row per conversation (thread_id, outcome, avg_confidence)
feedback_turns    → one row per draft/feedback/eval cycle
```

Key capabilities:
- `get_session_stats()` — real-time approval rates, avg turns, avg confidence
- `get_dataset_for_finetuning()` — exports `(prompt, draft, human_feedback, action, scores)` tuples
- CSV export via one-click Streamlit button
- WAL mode enabled for safe concurrent access

### 3. 🔁 Extended LangGraph Graph (`graph.py`)

Two new nodes inserted into the original graph flow:

```
START → assistant_draft → [evaluator ✦] → human_feedback (pause)
                ↑                               ↓
         assistant_draft ← [feedback_logger ✦] ←┘
                                ↓ (approved)
                        assistant_finalize → END
```

New state fields: `eval_scores`, `eval_rationale`, `revision_count`, `thread_id`

### 4. 🤗 HuggingFace Inference API Integration

Replaced OpenAI with **`Qwen/Qwen2.5-72B-Instruct`** via HuggingFace Inference API — free tier, no credit card, state-of-the-art quality. Both the draft model and the judge model use HuggingFace, making the entire pipeline **OpenAI-free**.

### 5. 🎛️ Streamlit Interface (`streamlit_app.py`)

Replaced the React frontend with a self-contained Streamlit app that embeds LangGraph directly (no FastAPI layer needed):

- Real-time score bars per criterion with color coding (green/orange/red)
- Revision history with role-based chat display
- Sidebar statistics dashboard
- One-click CSV export of the feedback dataset

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Streamlit App                       │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ HITL Chat│  │ Scorecard │  │  Stats + CSV │  │
│  └──────────┘  └───────────┘  └──────────────┘  │
└──────────────────────┬──────────────────────────┘
                       │ direct import (no API layer)
┌──────────────────────▼──────────────────────────┐
│              LangGraph Graph                     │
│                                                  │
│  assistant_draft → evaluator → human_feedback    │
│        ↑                            ↓            │
│   assistant_draft ← feedback_logger              │
│                          ↓ approved              │
│                   assistant_finalize → END       │
└──────────┬───────────────────┬──────────────────┘
           │                   │
    ┌──────▼──────┐    ┌───────▼──────┐
    │  HuggingFace│    │  SQLite DB   │
    │  Qwen2.5-72B│    │ feedback.db  │
    └─────────────┘    └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [pyenv](https://github.com/pyenv/pyenv) (recommended)
- A free [HuggingFace account](https://huggingface.co/settings/tokens) with a `Read` token

### Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/langgraph-hitl-genai-pipeline.git
cd langgraph-hitl-genai-pipeline

# 2. Create and activate Python env
pyenv virtualenv 3.11.10 env_hitl
cd backend
pyenv local env_hitl

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your HuggingFace token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# 5. Launch
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) — the app is ready.

---

## 📁 Project Structure

```
langgraph-hitl-genai-pipeline/
│
├── backend/
│   ├── app/
│   │   ├── graph.py               # ✦ Extended: +evaluator +feedback_logger nodes
│   │   ├── evaluator.py           # ✦ New: LLM-as-judge, 4 criteria, Pydantic output
│   │   ├── feedback_store.py      # ✦ New: SQLite persistence + dataset export
│   │   ├── models.py              # ✦ Extended: EvalScoresResponse, StatsResponse
│   │   ├── lesson_01_blocking.py  # ✦ Extended: returns eval scores, /stats, /dataset
│   │   ├── lesson_02_streaming.py # ✦ Extended: SSE 'eval' event
│   │   ├── lesson_03_async_mcp.py # Original: async MCP tool approval
│   │   ├── mcp_agent.py           # Original: ReAct agent with HITL tool wrapping
│   │   └── main.py                # Original: FastAPI entry point
│   ├── streamlit_app.py           # ✦ New: full Streamlit UI, no FastAPI needed
│   ├── requirements.txt
│   └── hitl_feedback.db           # Auto-generated on first run
│
├── screenshots/
│   ├── hitl_flow.png
│   └── stats_panel.png
│
└── README.md
```

---

## 🔬 GenAI Engineering Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **HITL workflow** | LangGraph `interrupt_before` + `update_state` + resume |
| **LLM-as-judge** | Dedicated evaluator node with structured Pydantic output |
| **Agentic state machine** | Typed `DraftReviewState` with explicit transitions |
| **Feedback loop for fine-tuning** | SQLite store → CSV export of `(prompt, draft, feedback, score)` |
| **Multi-turn revision tracking** | `revision_count` in state, per-turn score history |
| **Provider-agnostic LLM** | LangChain abstraction — swap OpenAI ↔ HuggingFace in one line |
| **Structured output parsing** | Pydantic `BaseModel` + regex fallback for robustness |
| **Persistent checkpointing** | LangGraph `MemorySaver` for in-session state |

---

## 📊 Feedback Dataset Schema

Every approved or rejected interaction is persisted and exportable as CSV:

```python
{
    "human_request":    str,   # original user prompt
    "draft":            str,   # LLM-generated draft
    "human_comment":    str,   # human feedback (if any)
    "human_action":     str,   # "approved" | "feedback"
    "score_coherence":  float, # 0-10
    "score_tone":       float, # 0-10
    "score_feedback":   float, # 0-10 | None
    "score_confidence": float, # 0-10
    "eval_rationale":   str,   # judge's explanation
    "turn_number":      int    # revision index
}
```

This dataset can be directly used for **RLHF**, **DPO**, or **supervised fine-tuning**.

---

## 🔧 Configuration

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace API token | required |
| `HITL_DB_PATH` | SQLite database path | `backend/hitl_feedback.db` |
| `repo_id` in `graph.py` | HuggingFace model | `Qwen/Qwen2.5-72B-Instruct` |

To switch back to OpenAI, replace in `graph.py` and `evaluator.py`:

```python
# Replace HuggingFace with:
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")
```

---

## 🙏 Credits

Base project: [esurovtsev/langgraph-hitl-fastapi-demo](https://github.com/esurovtsev/langgraph-hitl-fastapi-demo)  
Extended by: **[@azbenammar](https://github.com/azbenammar)**

---

## 📄 License

MIT
