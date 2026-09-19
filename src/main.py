"""CLI entry point for the loan decision engine.

Usage (from the project root, via Poetry):

    poetry run python -m src.main --applicant-id 124883
    poetry run python -m src.main --demo
    poetry run python -m src.main --applicant-id 124883 --json

Scores an existing applicant, routes the application to an operational path,
and runs the AI agent workflow for automated approve/decline decisions:
draft response -> two-stage bias validation -> send or escalate.
Denied applications also receive a next-best-offer recommendation.
"""

import argparse
import json
import sys

import requests

from src import model, pipeline

DEMO_IDS = [124883, 384178, 433605, 284617, 309296]


def check_ollama() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False


def run_one(sk_id: int, lgb_model, thresholds: dict, as_json: bool) -> dict:
    score = model.score_applicant(sk_id, lgb_model)
    decision = model.route(score, thresholds)
    customer_name = model.fake_name(sk_id)

    result: dict = {
        "SK_ID_CURR": sk_id,
        "customer_name": customer_name,
        "risk_score": round(score, 4),
        "decision": decision,
    }

    if decision == "human_review":
        result["final_status"] = "human_review_no_llm"
        if not as_json:
            print(f"applicant {sk_id} ({customer_name}): risk score {score:.4f} -> human_review")
            print("mid-range risk -- routed straight to a human underwriter, no AI drafting by design.\n")
    else:
        profile = model.applicant_summary(sk_id)
        agent_log = pipeline.process_applicant(sk_id, decision, profile, customer_name, verbose=not as_json)
        result.update({k: v for k, v in agent_log.items() if k not in ("SK_ID_CURR", "decision")})

    return result


def main():
    parser = argparse.ArgumentParser(description="Loan approval decision engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--applicant-id", type=int, help="SK_ID_CURR of an applicant already in the dataset")
    group.add_argument("--demo", action="store_true", help="run the 5 cases used throughout development")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a transcript")
    args = parser.parse_args()

    if not check_ollama():
        print(
            "Ollama isn't reachable at localhost:11434. Start it first: `ollama serve`, "
            "and make sure llama3.1:8b is pulled (`ollama pull llama3.1:8b`).",
            file=sys.stderr,
        )
        sys.exit(1)

    lgb_model = model.load_model()
    thresholds = model.load_thresholds()

    ids = DEMO_IDS if args.demo else [args.applicant_id]
    results = []
    for sk_id in ids:
        try:
            results.append(run_one(sk_id, lgb_model, thresholds, args.json))
        except ValueError as e:
            print(f"applicant {sk_id}: {e}", file=sys.stderr)
            if not args.demo:
                sys.exit(1)

    if args.json and results:
        print(json.dumps(results if args.demo else results[0], indent=2))


if __name__ == "__main__":
    main()
