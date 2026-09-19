# Home Credit Loan Decision Engine

## What this is

An end-to-end, production-style loan decision engine built on Kaggle's [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset.

A tuned LightGBM model estimates each applicant's default risk using `application_train.csv` together with six relational credit-history tables:

* `bureau.csv`
* `bureau_balance.csv`
* `previous_application.csv`
* `POS_CASH_balance.csv`
* `installments_payments.csv`
* `credit_card_balance.csv`

The model output is passed to a decision-routing layer that assigns each application to one of three paths:

* **Auto-approve**
* **Human review**
* **Auto-decline**

For automated decisions, a local LLM generates the customer-facing decision email. A two-stage bias-detection agent validates the response for potentially discriminatory language before it can be sent. For declined applications, a next-best-offer agent generates a concrete alternative based on the available application information.

The project separates **risk scoring, decision routing, AI-generated communication, and response validation** into independent components.

## Business Case

Loan applications need to be processed efficiently while maintaining appropriate human oversight.

A fully manual process can become a bottleneck as application volume increases, while fully automated decisions can be inappropriate when model confidence is low.

This system addresses that trade-off by:

* Automating high-confidence cases.
* Routing uncertain applications to human underwriters.
* Applying validation to AI-generated customer communications.
* Providing alternative options for declined applications.

The objective is to support faster processing while keeping human oversight and safety controls in the decision workflow.

## Data & Methodology

The model-development pipeline (`notebooks/01_model_development.ipynb`) follows a CRISP-DM-style workflow covering sections A through J.

### Data Integration

All six relational tables are joined to the raw application data before the main cleaning process.

This creates a consistent modelling dataset and allows missingness to be assessed across the complete feature schema.

### Data Cleaning

The pipeline applies targeted data-quality rules:

* Corrects the `DAYS_EMPLOYED` sentinel value (`365243`) representing no employment record.
* Removes sparse columns from the original application dataset using a >60% missing-value threshold.
* Retains history-based columns from the relational tables because missing history represents a meaningful business condition rather than simply missing data.
* Uses dedicated `HAS_*` indicators to capture the presence or absence of available credit history.

### Feature Engineering

Four domain-based features are added before feature selection:

* Credit-to-income ratio
* Annuity-to-income ratio
* Implied credit term
* Days-employed-to-age ratio

These features provide additional information about affordability, repayment burden, loan structure, and employment history.

### Feature Encoding & Selection

Categorical variables are one-hot encoded before feature selection.

Feature selection then applies:

1. Variance filtering.
2. Correlation-based redundancy filtering (`|corr| > 0.9`).
3. Random Forest feature importance.

One-hot encoded categories are evaluated individually so that useful categories can be retained without requiring the entire original categorical field to remain in the model.

### Final Feature Set

The final selected feature set contains **72 raw fields**, expanding to **74 encoded model features** after one-hot encoding.

## Model Results

| Development stage         | Tuned ROC-AUC |
| ------------------------- | ------------: |
| Previous methodology      |        0.7714 |
| + `EXT_SOURCE_1` restored |        0.7763 |
| + Domain-based features   |        0.7809 |
| + Encode-before-select    |        0.7798 |

The current pipeline uses the encode-before-select methodology because it provides a more consistent feature-selection process, even though the change resulted in a small reduction of approximately 0.001 ROC-AUC.

The methodology was retained based on the data-processing design rather than selecting the approach solely for the highest metric.

Before hyperparameter tuning, LightGBM achieved the highest ROC-AUC among the tested baseline models:

* LightGBM: 0.7772
* Logistic Regression: 0.7615
* XGBoost: 0.7613
* Random Forest: 0.7438

This supported selecting LightGBM as the primary model for hyperparameter tuning.

A blend of the four tuned models was also evaluated but did not outperform the standalone tuned LightGBM model. The application therefore uses the single tuned LightGBM model.

Model evaluation includes:

* ROC curve
* Row-normalized confusion matrix
* Classification report
* Validation-set ROC-AUC

## Decision Routing

The application uses rank-based routing thresholds derived from the model's validation-score distribution rather than treating the raw predicted probability as a calibrated probability.

This is important because `class_weight="balanced"` affects the model's probability output.

The routing thresholds are calibrated using the 20th and 95th percentiles of the validation scores, producing approximately:

* **20% Auto-approve**
* **75% Human review**
* **5% Auto-decline**

This routing layer is separate from the ML model itself. The model produces the risk score; business rules determine the operational path.

## AI Agent Pipeline

The AI workflow is implemented in `src/agents.py` and `src/pipeline.py`.

All LLM operations run locally through Ollama using `llama3.1:8b`, so the application does not require an external LLM API or API key.

For automated decisions, the pipeline performs the following steps:

### 1. Response Generation

The LLM generates a customer-facing decision email using the applicant and decision context.

For declined applications, the response can also include a next-best-offer.

### 2. Bias Validation

The generated response passes through two bias-detection stages.

A second validation pass provides an additional check for potentially discriminatory language that may not be identified by the first stage.

During development and testing, the detector identified instances of potentially problematic proxy-based reasoning, including references to employment history.

### 3. Next-Best Offer

For declined applications, the system can generate a concrete alternative rather than returning only a rejection.

The offer is based on income and requested credit amount and does not use education, occupation, gender, or age.

### 4. Safety Escalation

If either bias-validation stage flags the generated response, the case is escalated to human review.

This provides a shared safety path for both uncertain model decisions and AI-generated communication that requires additional review.

## Application Layer

The application is implemented using FastAPI and a static web frontend.

`src/api.py` serves both the API and frontend from a single process, while the core inference components remain separated into individual modules.

The application supports two inference paths for new applicants:

### Manual Entry

A form accepts approximately 20 application-level fields that would typically be available when a new application is submitted.

### CSV Upload

Multiple new applicants can be submitted in a CSV file for batch scoring.

Both paths use the same feature transformation logic in `src/features.py`.

The transformation pipeline uses the training-time artifacts for:

* Median values.
* Selected feature list.
* Encoded feature schema.

The model is not retrained or refitted during inference.

### New Applicant Data Coverage

The model contains 74 features, including 35 features derived from bureau, previous-loan, and payment-history data.

A genuinely new applicant will not have these historical records available.

As a result, new-applicant predictions rely primarily on application-level information and are less informed than predictions for applicants with existing credit history.

This limitation is surfaced directly in the application rather than hidden from the user.

## Test-Set Predictions

`notebooks/02_test_predictions.ipynb` applies the complete inference pipeline to Kaggle's held-out `application_test.csv`.

The test set contains **48,744 applicants** and does not include the `TARGET` column.

The pipeline uses the same:

* Data integration logic.
* Cleaning rules.
* Feature engineering.
* Encoding.
* Feature-selection schema.
* Training-derived median values.

No preprocessing statistics are recalculated from the test set.

The generated output is validated against `sample_submission.csv` to confirm:

* Matching columns.
* Matching row count.
* Matching applicant IDs.

The predictions are not submitted to the Kaggle leaderboard because the ground-truth target values for the test set are withheld.

The notebook completes the end-to-end batch inference workflow and produces the expected prediction format.

## Limitations

* Model performance is based on the Home Credit dataset and may not represent performance on a different lending population.
* Feature engineering and the six-table data integration contributed more to model performance than hyperparameter tuning alone.
* The encode-before-select methodology produced a small ROC-AUC regression of approximately 0.001 compared with the previous approach.
* New applicants do not have the 35 historical features derived from existing credit records, so their predictions contain less information.
* The bias-detection layer uses a local 8B model and should be treated as a validation layer rather than a guarantee of unbiased communication.
* A real lending deployment would require additional controls including independent model validation, monitoring, audit logging, security controls, fairness testing, regulatory review, and human governance.

## Future Work

* Incorporate alternative data sources (e.g. utility or rental payment history) to improve scoring for new applicants who lack bureau and previous-loan records.
* Surface a confidence indicator alongside the risk score when an applicant has little or no historical data available, so the score's reliability is visible at the point of use, not only in the documentation.
* Replace the local 8B bias-detection model with a hybrid local-generation / API-judged architecture for more reliable bias screening.
* Add monitoring for model and data drift if the system were run against live, ongoing applications rather than a static dataset.

## Project Files

* `notebooks/01_model_development.ipynb` — model development and evaluation
* `notebooks/02_test_predictions.ipynb` — batch inference on the held-out test data
* `src/model.py` — model loading, scoring, and decision routing
* `src/features.py` — live feature transformation
* `src/agents.py` — AI response and validation agents
* `src/pipeline.py` — end-to-end pipeline orchestration
* `src/api.py` — FastAPI backend
* `src/main.py` — CLI entry point
* `app/index.html` — web frontend
* `models/` — trained model and routing thresholds
* `data/processed/` — training-derived artifacts used by the inference pipeline
