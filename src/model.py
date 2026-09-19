"""Model loading, applicant scoring, and risk-tier routing.
"""

import json
import random
from pathlib import Path

import joblib
import pandas as pd

from src import features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "lightgbm_full_schema_tuned.joblib"
THRESHOLDS_PATH = MODELS_DIR / "routing_thresholds_v2.json"
MODEL_READY_CSV = DATA_DIR / "processed" / "application_train_model_ready_full.csv"
READABLE_CSV = DATA_DIR / "processed" / "application_selected_features_full.csv"


def load_model():
    return joblib.load(MODEL_PATH)


def load_thresholds() -> dict:
    with open(THRESHOLDS_PATH) as f:
        return json.load(f)


def route(score: float, thresholds: dict) -> str:
    if score < thresholds["approve_cutoff"]:
        return "approved"
    elif score >= thresholds["reject_cutoff"]:
        return "denied"
    else:
        return "human_review"


def score_applicant(sk_id: int, model) -> float:
    """Look up an existing applicant and return the model's risk score.

    Uses the full model-ready feature set generated during model development.
    This path scores applicants already present in the processed dataset.
    """
    df = pd.read_csv(MODEL_READY_CSV)
    row = df[df["SK_ID_CURR"] == sk_id]
    if row.empty:
        raise ValueError(f"SK_ID_CURR {sk_id} not found in {MODEL_READY_CSV.name}")
    X = row.drop(columns=["SK_ID_CURR", "TARGET"])
    return float(model.predict_proba(X)[:, 1][0])


def score_new_applicant(raw: dict, model) -> tuple[float, list[str]]:
    """Score a new applicant from application-level fields.

    Returns the risk score and the history fields available for the applicant.
    For a genuinely new applicant, unavailable bureau, previous-loan, and
    payment-history features fall back to training-set medians. As a result,
    the score relies primarily on the applicant's current application data.
    """
    X = features.transform(raw)
    score = float(model.predict_proba(X)[:, 1][0])
    return score, features.history_fields_provided(raw)


def new_applicant_summary(raw: dict) -> str:
    """Build a human-readable profile for an LLM prompt from new-applicant data."""
    age_years = -raw["DAYS_BIRTH"] / 365 if raw.get("DAYS_BIRTH") else None
    parts = []
    if age_years is not None:
        parts.append(f"Age: {age_years:.0f}")
    if raw.get("AMT_INCOME_TOTAL"):
        parts.append(f"Income: ${raw['AMT_INCOME_TOTAL']:,.0f}")
    if raw.get("AMT_CREDIT"):
        parts.append(f"Requested credit: ${raw['AMT_CREDIT']:,.0f}")
    if raw.get("NAME_EDUCATION_TYPE"):
        parts.append(f"Education: {raw['NAME_EDUCATION_TYPE']}")
    if raw.get("NAME_INCOME_TYPE"):
        parts.append(f"Income type: {raw['NAME_INCOME_TYPE']}")
    if raw.get("OCCUPATION_TYPE"):
        parts.append(f"Occupation: {raw['OCCUPATION_TYPE']}")
    return ", ".join(parts) if parts else "New applicant, minimal profile provided."


_FIRST_NAMES = [
    "James", "Maria", "Wei", "Fatima", "Liam", "Aisha", "Noah", "Sofia",
    "Kenji", "Priya", "Lucas", "Elena", "Omar", "Grace", "Daniel", "Anya",
]
_LAST_NAMES = [
    "Carter", "Nguyen", "Rossi", "Kim", "Patel", "Novak", "Silva", "Johansson",
    "Tanaka", "Okafor", "Moreau", "Andersson", "Haddad", "Fischer", "Costa", "Wong",
]


def fake_name(sk_id: int) -> str:
    """Generate a deterministic synthetic display name for an applicant.

    The Home Credit dataset contains anonymized applicant IDs rather than real
    customer names. The generated name is synthetic and deterministic, so the
    same applicant receives the same display name across runs and API requests.
    """
    rng = random.Random(sk_id)
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def applicant_summary(sk_id: int) -> str:
    """Build a human-readable applicant profile for LLM prompts.

    Uses available raw feature labels rather than one-hot encoded column names.
    Only fields present in the current selected feature set are included, so the
    profile remains compatible with changes to the feature-selection output.
    """
    df = pd.read_csv(READABLE_CSV)
    row = df[df["SK_ID_CURR"] == sk_id]
    if row.empty:
        raise ValueError(f"SK_ID_CURR {sk_id} not found in {READABLE_CSV.name}")
    row = row.iloc[0]

    parts = []
    if "DAYS_BIRTH" in row and pd.notna(row["DAYS_BIRTH"]):
        parts.append(f"Age: {-row['DAYS_BIRTH'] / 365:.0f}")
    if "AMT_INCOME_TOTAL" in row and pd.notna(row["AMT_INCOME_TOTAL"]):
        parts.append(f"Income: ${row['AMT_INCOME_TOTAL']:,.0f}")
    if "AMT_CREDIT" in row and pd.notna(row["AMT_CREDIT"]):
        parts.append(f"Requested credit: ${row['AMT_CREDIT']:,.0f}")
    if "NAME_EDUCATION_TYPE" in row and pd.notna(row["NAME_EDUCATION_TYPE"]):
        parts.append(f"Education: {row['NAME_EDUCATION_TYPE']}")
    if "NAME_INCOME_TYPE" in row and pd.notna(row["NAME_INCOME_TYPE"]):
        parts.append(f"Income type: {row['NAME_INCOME_TYPE']}")
    if "OCCUPATION_TYPE" in row and pd.notna(row["OCCUPATION_TYPE"]):
        parts.append(f"Occupation: {row['OCCUPATION_TYPE']}")
    return ", ".join(parts) if parts else f"Applicant {sk_id}, minimal profile available."
