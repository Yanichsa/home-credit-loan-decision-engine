"""The three local LLM agents: email drafting, two-stage bias detection, next-best-offer.

Requires Ollama running locally with llama3.1:8b pulled (`ollama serve`,
`ollama pull llama3.1:8b`) free, no API key
"""

import re

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

# stage 1: at or above this, straight to human. stage 2 (tougher): at or above
# this, also to human. Lowered from an initial 60/30 after the exact flagged
# email re-scored at exactly 60 and a strict ">" check let it through on a
# boundary technicality -- see docs/project-summary.md.
BIAS_FAIL_THRESHOLD = 50
BIAS_TOUGH_THRESHOLD = 25

PROXY_FACTORS = "education level, occupation, income type, gender, or age"

LENDER_NAME = "NS Bank" # Fictional Companny

EMAIL_SYSTEM = (
    f"You are a loan officer at {LENDER_NAME} writing a short, professional, respectful email "
    "to a customer about their loan application decision. Address the customer by the name "
    "given. Never be rude, dismissive, or judgmental. Sign off as the "
    f"{LENDER_NAME} loan team, not a personal name. Keep it under 120 words."
)

OFFER_SYSTEM = (
    "You generate a brief, realistic next-best-offer for a denied loan applicant, based on "
    "their profile. Suggest ONE concrete alternative (e.g. a smaller loan amount, a secured "
    "credit product, or a financial counseling resource). Base it only on income and requested "
    "credit amount, never on education, occupation, gender, or age. One sentence only."
)


def call_llm(prompt: str, system: str | None = None, temperature: float | None = None) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def draft_email(decision: str, profile: str, customer_name: str, offer: str | None = None) -> str:
    prompt = f"Customer name: {customer_name}\nApplicant profile: {profile}\nDecision: {decision}."
    if offer:
        prompt += f" Include this next-step recommendation naturally in the email: {offer}"
    prompt += "\nWrite the email now."
    return call_llm(prompt, system=EMAIL_SYSTEM)


def _parse_score(text: str, default: int = 100) -> int:
    # fail-safe: if we can't parse a score, treat it as maximally concerning
    # so it escalates to a human rather than silently passing.
    match = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else default


def bias_check(email_text: str, tougher: bool = False) -> tuple[int, str]:
    strictness = (
        "Be strict and skeptical -- flag anything even mildly insensitive, dismissive, "
        "or that could read as discriminatory."
        if tougher
        else "Flag clear bias, discrimination, or offensive/insulting language."
    )
    system = (
        f"You are a bias-detection reviewer for customer-facing loan emails. {strictness} "
        f"CRITICAL RULE, regardless of tone or politeness: if the email cites the applicant's "
        f"{PROXY_FACTORS} as a REASON or JUSTIFICATION for the decision or the loan terms offered, "
        f"that is proxy discrimination and MUST score at least 70, even if the email is polite. "
        f"A decision may be based on income amount, credit amount, or repayment risk -- never on "
        f"who the applicant is. "
        "Respond in exactly this format:\nSCORE: <0-100, where 0 is completely clean and "
        "100 is severely biased/offensive>\nREASON: <one sentence>"
    )
    # temperature=0 on scoring calls: re-scoring the SAME email text produced
    # 20, then 60, then 30 across separate calls at default temperature --
    # non-determinism in the judge itself, independent of the prompt.
    response = call_llm(f"Email to review:\n{email_text}", system=system, temperature=0)
    return _parse_score(response), response


def next_best_offer(profile: str) -> str:
    return call_llm(f"Denied applicant profile: {profile}", system=OFFER_SYSTEM)
