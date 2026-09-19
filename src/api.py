"""FastAPI backend for the loan decision engine.

Provides the HTTP API and serves the web frontend. The API uses the same
model and pipeline components as the CLI, keeping scoring, decision routing,
and agent workflows consistent across application interfaces.

The frontend is served from the app/ directory, allowing the API and web
application to run from a single process.

Run:
poetry run uvicorn src.api:app --reload

Then open:
http://localhost:8000
"""


import io
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import model, pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"

app = FastAPI(title="Loan approval decision engine")

_model = None
_thresholds = None


def get_model():
    global _model
    if _model is None:
        _model = model.load_model()
    return _model


def get_thresholds():
    global _thresholds
    if _thresholds is None:
        _thresholds = model.load_thresholds()
    return _thresholds


DEMO_IDS = [124883, 384178, 433605, 284617, 309296]


@app.get("/api/demo-ids")
def demo_ids():
    """A curated set of known-good applicant IDs, for the UI's quick-pick list.
    Every result is still a live call -- nothing here is pre-baked."""
    return {"ids": DEMO_IDS}


@app.get("/api/applicant/{sk_id}")
def get_applicant(sk_id: int):
    try:
        score = model.score_applicant(sk_id, get_model())
    except ValueError:
        raise HTTPException(status_code=404, detail=f"applicant {sk_id} not found in the dataset")

    thresholds = get_thresholds()
    decision = model.route(score, thresholds)
    customer_name = model.fake_name(sk_id)
    result = {
        "SK_ID_CURR": sk_id,
        "customer_name": customer_name,
        "risk_score": round(score, 4),
        "decision": decision,
    }

    profile_row = model.applicant_summary(sk_id)
    result["profile"] = profile_row

    if decision == "human_review":
        result["final_status"] = "human_review_no_llm"
    else:
        agent_log = pipeline.process_applicant(sk_id, decision, profile_row, customer_name, verbose=False)
        result.update({k: v for k, v in agent_log.items() if k not in ("SK_ID_CURR", "decision")})

    return result


class NewApplicantRequest(BaseModel):
    customer_name: str
    fields: dict


class NewApplicantsBatchRequest(BaseModel):
    csv_text: str


_pseudo_id_counter = 0


def _next_pseudo_id() -> int:
    # negative + monotonic, so batch rows never collide with each other or
    # with a real (always-positive) SK_ID_CURR from the dataset.
    global _pseudo_id_counter
    _pseudo_id_counter -= 1
    return -(int(time.time() * 1000) % 1_000_000_000) + _pseudo_id_counter


def _score_new_applicant_record(customer_name: str, fields: dict) -> dict:
    """Shared by the single-entry and CSV-batch endpoints
    Score one new applicant's raw fields and, for approve/deny, run the AI agent pipeline."""
    score, history_fields = model.score_new_applicant(fields, get_model())
    thresholds = get_thresholds()
    decision = model.route(score, thresholds)

    pseudo_id = _next_pseudo_id()
    result = {
        "SK_ID_CURR": pseudo_id,
        "customer_name": customer_name,
        "risk_score": round(score, 4),
        "decision": decision,
        "new_customer": True,
        "history_fields_provided": history_fields,
    }

    profile_row = model.new_applicant_summary(fields)
    result["profile"] = profile_row

    if decision == "human_review":
        result["final_status"] = "human_review_no_llm"
    else:
        agent_log = pipeline.process_applicant(pseudo_id, decision, profile_row, customer_name, verbose=False)
        result.update({k: v for k, v in agent_log.items() if k not in ("SK_ID_CURR", "decision")})

    return result


@app.post("/api/new-applicant")
def score_new_applicant(payload: NewApplicantRequest):
    """Score a genuinely new applicant, not already in the processed dataset.

    `fields` supplies application-level values (income, credit amount,
    education, etc.) by their raw column name 
    Anything not supplied, which for a real new customer
    is normally every bureau/previous-loan/payment-history field, falls back
    to a training-set median. `history_fields_provided` in the response is
    the honesty check: empty means this score leans almost entirely on the
    application-level fields, not credit history.
    """
    return _score_new_applicant_record(payload.customer_name, payload.fields)


@app.post("/api/new-applicants-batch")
def score_new_applicants_batch(payload: NewApplicantsBatchRequest):
    """Score a CSV of new applicants at once. Expected columns: `customer_name`
    plus any of the raw application-level fields from selected_features_full.txt
    (income, credit amount, education, etc.) -- unrecognized/omitted columns are
    fine, same fallback-to-median behavior as the single-applicant endpoint."""
    try:
        df = pd.read_csv(io.StringIO(payload.csv_text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"couldn't parse CSV: {e}")

    if "customer_name" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a customer_name column")

    results = []
    for _, row in df.iterrows():
        fields = {k: (None if pd.isna(v) else v) for k, v in row.items() if k != "customer_name"}
        results.append(_score_new_applicant_record(str(row["customer_name"]), fields))

    return {"results": results}


# serve the frontend last, so /api/* routes above take priority
app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="app")
