# Home Credit Loan Decision Engine

A **loan decision engine** built on Kaggle's [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset.

A tuned **LightGBM** model (**ROC-AUC 0.77**) uses data from all six relational credit-history tables to estimate each applicant's default risk. The resulting risk score is passed through a routing layer that assigns the application to one of three operational paths:

* **Auto-approve** — high-confidence approval
* **Human review** — borderline or uncertain cases
* **Auto-decline** — high-confidence decline

For automated decisions, a local LLM drafts the customer-facing decision email. A two-stage bias-detection agent checks the generated response for potentially discriminatory language before it can be sent. For declined applications, a next-best-offer agent can generate a relevant alternative to the original application.

The project separates **ML scoring, decision routing, AI-assisted communication, and response validation** into distinct components to demonstrate an end-to-end ML application architecture.

## What's in here

* `notebooks/01_model_development.ipynb` — model development and validation, covering business requirements, data profiling, data cleaning, feature engineering across all six tables, feature selection, encoding, baseline models, hyperparameter tuning, and model comparison. An ensemble blend was evaluated but did not improve on the tuned LightGBM model, so the single model is used in the application.
* `notebooks/02_test_predictions.ipynb` — applies the trained model and the same feature-processing logic to unseen test data and generates batch predictions.
* `src/` — application and inference code:

  * `model.py` — model loading, risk scoring, and decision routing
  * `agents.py` — LLM-based response, bias-checking, and next-best-offer agents
  * `pipeline.py` — end-to-end pipeline orchestration
  * `main.py` — command-line interface
  * `api.py` — FastAPI backend
  * `features.py` — feature transformation for a new applicant (imputation, encoding, schema alignment)
* `app/` — web frontend (`index.html`) served by the API.
* `models/` — trained LightGBM model artifact and routing thresholds.
* `data/` — Home Credit source data and processed datasets used by the application.

## Setup

Requires [Poetry](https://python-poetry.org/) and **Python 3.12**.

```bash
poetry env use /opt/homebrew/bin/python3.12   # adjust path if Python 3.12 lives elsewhere
poetry install
```

### Data

This repository does not include the Home Credit dataset itself — it's a Kaggle competition dataset and isn't redistributed here. To run the notebooks or the application:

1. Download the data from the [Home Credit Default Risk competition page](https://www.kaggle.com/competitions/home-credit-default-risk/data) (requires a free Kaggle account), or via the Kaggle CLI:

   ```bash
   kaggle competitions download -c home-credit-default-risk
   ```
2. Unzip the files into a `data/` folder at the project root, so you have `data/application_train.csv`, `data/bureau.csv`, etc.
3. Run `notebooks/01_model_development.ipynb` end to end first — it generates the processed artifacts in `data/processed/` (medians, selected features, encoded schema) that the application and `notebooks/02_test_predictions.ipynb` depend on.

### Local LLM with Ollama

The AI agents run locally through [Ollama](https://ollama.com), so no external LLM API is required.

Install Ollama and pull the required model:

```bash
brew install ollama
ollama pull llama3.1:8b
ollama serve
```

The CLI and API check that Ollama is available at startup and return an error if the local model server is not running.

## Running the Application

### Model Development

Open `notebooks/01_model_development.ipynb` in VS Code or Jupyter and select the **"Home Credit Approval (clean)"** kernel.

### CLI

Run a prediction for an applicant already present in the processed dataset:

```bash
poetry run python -m src.main --applicant-id 433605
```

Run the built-in demo:

```bash
poetry run python -m src.main --demo --json
```

### Web Application

Start the FastAPI backend:

```bash
poetry run uvicorn src.api:app --reload
```

Then open:

```text
http://localhost:8000
```

The web application sends requests to the API, which runs the model-scoring and decision pipeline.

## Decision Pipeline

The application follows this flow:

**Applicant data → Feature processing → LightGBM scoring → Risk routing → AI response → Bias validation → Final response**

Applications routed to **human review** bypass the AI response-generation stage and are returned for manual underwriting.

## Model

**Algorithm:** LightGBM
**Training data:** Home Credit application data + six relational credit-history tables
**Evaluation metric:** ROC-AUC
**Test ROC-AUC:** 0.77

The trained model is saved as a reusable artifact and loaded by the application for inference.

## Limitations

* The application scores **existing applicants from the processed dataset by applicant ID**, and also accepts **genuinely new applicants** through a manual-entry form or CSV batch upload. New-applicant scores lean almost entirely on application-level fields (income, credit amount, education, etc.), since a new applicant has no bureau, previous-loan, or payment-history records by definition — 35 of the model's 74 features fall back to a training-set median in that case, so these scores are materially less informed than an existing customer's.
* The bias-detection layer uses a local **8B parameter model**. During development and testing, it identified potentially problematic proxy-discrimination language in generated responses, but the detector should not be treated as a guaranteed or comprehensive fairness check.
* The project is a **production-style portfolio implementation**, not a deployed lending system. A real lending deployment would require additional controls such as independent model validation, monitoring, audit logging, access controls, fairness testing, security controls, regulatory review, and human governance.

## Future Work

* Incorporate alternative data sources (e.g. utility or rental payment history) to improve scoring for new applicants who lack bureau and previous-loan records.
* Surface a confidence indicator alongside the risk score when an applicant has little or no historical data available, so the score's reliability is visible at the point of use.
* Replace the local 8B bias-detection model with a hybrid local-generation / API-judged architecture for more reliable bias screening.
* Add monitoring for model and data drift if the system were run against live, ongoing applications rather than a static dataset.
