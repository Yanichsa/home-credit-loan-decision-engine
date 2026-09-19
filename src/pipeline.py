"""Orchestrate the AI workflow for an automated loan decision.

Routes an approved or denied application through the AI agents to generate
the customer email, run two-stage bias validation, and generate a next-best
offer for denied applications. If either bias check fails, the application
is escalated to human review.

Human-review decisions do not enter this workflow and are handled separately.
"""

from src import agents


def process_applicant(sk_id: int, decision: str, profile: str, customer_name: str, verbose: bool = True) -> dict:
    """Process an automated loan decision through the AI workflow.

    The decision must be either "approved" or "denied". Applications routed to
    "human_review" bypass the AI workflow and are sent directly for human review.
    """
    log: dict = {"SK_ID_CURR": sk_id, "decision": decision}

    offer = None
    if decision == "denied":
        offer = agents.next_best_offer(profile)
        log["next_best_offer"] = offer

    email = agents.draft_email(decision, profile, customer_name, offer)
    log["email_draft"] = email

    score1, reason1 = agents.bias_check(email, tougher=False)
    log["bias_score_stage1"] = score1
    if score1 >= agents.BIAS_FAIL_THRESHOLD:
        log["final_status"] = "escalated_to_human"
        log["escalation_reason"] = f"stage 1 bias check failed ({score1}): {reason1}"
    else:
        score2, reason2 = agents.bias_check(email, tougher=True)
        log["bias_score_stage2"] = score2
        if score2 >= agents.BIAS_TOUGH_THRESHOLD:
            log["final_status"] = "escalated_to_human"
            log["escalation_reason"] = f"stage 2 tougher check failed ({score2}): {reason2}"
        else:
            log["final_status"] = "sent"

    if verbose:
        print(f"=== applicant {sk_id}, {customer_name} ({decision}) ===")
        if offer:
            print(f"next best offer: {offer}")
        print(f"email:\n{email}\n")
        print(f"stage 1 bias score: {score1}")
        if "bias_score_stage2" in log:
            print(f"stage 2 (tougher) bias score: {log['bias_score_stage2']}")
        print(f"final status: {log['final_status']}")
        print()

    return log
