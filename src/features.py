"""Feature transformation for a new applicant.

Applies the same cleaning, feature-selection, and encoding decisions used
during model development to a single new applicant record.

Training-time artifacts provide the values and schema required for inference:

  - `application_train_full_schema.csv` -> training-set medians for imputation
  - `selected_features_full.txt`        -> selected model features
  - `application_train_model_ready_full.csv` (header only) -> encoded feature
    schema expected by the model

Limitation: 35 of the selected features are bureau/previous-loan/payment-
history aggregates (BUREAU_*, PREV_*, POS_*, INST_*, CC_*). A new applicant
does not have records in these history tables, so these features fall back
to training-set medians.

As a result, new-applicant scores rely primarily on application-level
information such as income, credit amount, and education, and are less
informed by credit history than scores for applicants with existing history.
"""


import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FULL_SCHEMA_CSV = DATA_DIR / "processed" / "application_train_full_schema.csv"
SELECTED_FEATURES_TXT = DATA_DIR / "processed" / "selected_features_full.txt"
MODEL_READY_CSV = DATA_DIR / "processed" / "application_train_model_ready_full.csv"

# DAYS_EMPLOYED sentinel value used in the source data
DAYS_EMPLOYED_SENTINEL = 365243

# Feature prefixes representing bureau and historical credit data
HISTORY_PREFIXES = ("BUREAU_", "PREV_", "POS_", "INST_", "CC_")

_cache: dict = {}


def _load():
    if _cache:
        return _cache
    full = pd.read_csv(FULL_SCHEMA_CSV)
    selected = Path(SELECTED_FEATURES_TXT).read_text().split()
    cat_cols = [c for c in selected if not pd.api.types.is_numeric_dtype(full[c])]
    medians = full[[c for c in selected if c not in cat_cols]].median()
    model_ready_cols = pd.read_csv(MODEL_READY_CSV, nrows=0).columns.tolist()
    model_ready_cols = [c for c in model_ready_cols if c not in ("SK_ID_CURR", "TARGET")]
    _cache.update(medians=medians, selected=selected, cat_cols=cat_cols, model_ready_cols=model_ready_cols)
    return _cache


def history_fields_provided(raw: dict) -> list[str]:
    """Return any bureau or historical credit fields supplied by the caller."""
    return [k for k in raw if k.startswith(HISTORY_PREFIXES) and raw[k] not in (None, "")]


def transform(raw: dict) -> pd.DataFrame:
    """Transform one new applicant record into the model's expected input format."""
    cache = _load()
    raw = dict(raw)
    if raw.get("DAYS_EMPLOYED") == DAYS_EMPLOYED_SENTINEL:
        raw["DAYS_EMPLOYED"] = None

    row = {}
    for col in cache["selected"]:
        val = raw.get(col)
        if val in (None, ""):
            val = None if col in cache["cat_cols"] else cache["medians"].get(col)
        row[col] = val

    df = pd.DataFrame([row])
    if cache["cat_cols"]:
        df = pd.get_dummies(df, columns=cache["cat_cols"], drop_first=False)
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in df.columns]
    df = df.reindex(columns=cache["model_ready_cols"], fill_value=0)
    return df.astype(float)
