"""
CLAIRE — src/respond.py
Interactive complaint responder: takes a consumer complaint narrative,
runs the full CLAIRE pipeline, and generates an automated response.

Usage (interactive):
    python src/respond.py

Usage (single-shot CLI):
    python src/respond.py --narrative "I was charged $450 without my consent..."

Usage (JSON output):
    python src/respond.py --narrative "..." --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SETTINGS
from src.complaint_parser import ComplaintRecord, extract_llm_fields, extract_regex_fields

# Load .env for API key
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Response generation prompt
# ---------------------------------------------------------------------------

_RESPONSE_SYSTEM_PROMPT = """You are CLAIRE, an expert analyst of CFPB consumer financial complaints filed against PayPal.
Given a complaint narrative, classify it using the official CFPB taxonomy and return ONLY valid JSON. Your tone is professional, empathetic, and action-oriented.

Given structured complaint analysis, write a response that:
1. Acknowledges the consumer's specific situation and validates their concern
2. Explains what typically happens in this type of case
3. Provides 2-3 clear, actionable next steps the consumer should take
4. States a realistic resolution timeline based on the SLA days provided
5. Closes with a contact information placeholder

Rules:
- Match tone to sentiment: be warmer for "angry"/"frustrated", more informative for "confused"/"neutral"
- For urgency=high: emphasise speed and escalation paths
- For urgency=low: focus on process and patience
- Keep response under {max_words} words
- Use plain English, no legal jargon
- Never fabricate company names, case numbers, or guarantees
- Reference the SLA timeline naturally (e.g. "PayPal typically resolves this within X days")
- End with: "Reference ID: [CASE-{{complaint_id}}]"
""".format(max_words=SETTINGS["RESPONSE_MAX_TOKENS"])

_RESPONSE_USER_TEMPLATE = """Complaint analysis:
- Product type  : {cfpb_product}
- Issues        : {cfpb_issues}
- Urgency       : {urgency_level}
- Sentiment     : {sentiment}
- Consumer goal : {consumer_goal}
- SLA days      : {sla_days}
- Dollar amounts: {dollar_amounts}
- Contact attempts: {contact_attempts}
- Contextual cues : {contextual_cues}

Original narrative:
\"\"\"{narrative}\"\"\"

Write the automated response now."""


@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(
        multiplier=2,
        min=SETTINGS["RETRY_WAIT_MIN"],
        max=SETTINGS["RETRY_WAIT_MAX"]
    ),
    stop=stop_after_attempt(SETTINGS["RETRY_MAX_ATTEMPTS"]),
)
def _generate_response(record: ComplaintRecord) -> str:
    """Call gpt-4o-mini to generate a consumer-facing response."""
    time.sleep(SETTINGS["SLEEP_RESPONSE_GEN"])
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        max_retries=SETTINGS["MAX_RETRIES"]
    )

    user_msg = _RESPONSE_USER_TEMPLATE.format(
        cfpb_product=record.cfpb_product,
        cfpb_issues=", ".join(record.cfpb_issues) if record.cfpb_issues else record.cfpb_issue,
        urgency_level=record.urgency_level,
        sentiment=record.sentiment,
        consumer_goal=record.consumer_goal,
        sla_days=record.sla_days,
        dollar_amounts=record.dollar_amounts or "none",
        contact_attempts=record.contact_attempts,
        contextual_cues=record.contextual_cues or "none",
        narrative=record.raw_narrative[:SETTINGS["NARRATIVE_MAX_CHARS"]],
    )

    resp = client.chat.completions.create(
        model=SETTINGS["MODEL_NAME"],
        messages=[
            {"role": "system", "content": _RESPONSE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=SETTINGS["RESPONSE_TEMPERATURE"],
        max_tokens=SETTINGS["RESPONSE_MAX_TOKENS"],
    )
    text = resp.choices[0].message.content.strip()
    return text.replace("{complaint_id}", record.complaint_id)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def analyse_and_respond(narrative: str, complaint_id: str = "MANUAL") -> dict:
    """Run full CLAIRE pipeline and return analysis + response."""
    record = ComplaintRecord(
        complaint_id=complaint_id,
        raw_narrative=narrative,
        product="", sub_product="", issue="", sub_issue="",
        state="", submitted_via="manual", timely_response="",
    )

    print("\n[1/3] Running regex extraction...")
    extract_regex_fields(record)

    print("[2/3] Running LLM classification...")
    extract_llm_fields(record)

    print("[3/3] Generating automated response...\n")
    response_text = _generate_response(record)

    return {
        "complaint_id": record.complaint_id,
        "analysis": {
            "cfpb_product":              record.cfpb_product,
            "cfpb_sub_product":          record.cfpb_sub_product,
            "cfpb_issues":               record.cfpb_issues,
            "cfpb_sub_issue":            record.cfpb_sub_issue,
            "cfpb_confidence":           record.cfpb_confidence,
            "sla_days":                  record.sla_days,
            "inferred_product_type":     record.inferred_product_type,
            "primary_issue_keyword":     record.primary_issue_keyword,
            "urgency_level":             record.urgency_level,
            "sentiment":                 record.sentiment,
            "consumer_goal":             record.consumer_goal,
            "dollar_amounts":            record.dollar_amounts,
            "dates_mentioned":           record.dates_mentioned,
            "contact_attempts":          record.contact_attempts,
            "account_numbers_redacted":  record.account_numbers_redacted,
            "contextual_cues":           record.contextual_cues,
        },
        "automated_response": response_text,
    }


def print_result(result: dict) -> None:
    """Pretty-print analysis and response."""
    a   = result["analysis"]
    sep = "=" * 70

    print(sep)
    print("  CLAIRE COMPLAINT ANALYSIS")
    print(sep)
    print(f"  Product         : {a['cfpb_product']}")
    print(f"  Sub-product     : {a['cfpb_sub_product']}")
    print(f"  Issues          : {', '.join(a['cfpb_issues']) if a['cfpb_issues'] else 'N/A'}")
    print(f"  Sub-issue       : {a['cfpb_sub_issue']}")
    print(f"  Confidence      : {a['cfpb_confidence']}")
    print(f"  SLA days        : {a['sla_days']} days")
    print(f"  Urgency         : {a['urgency_level']}")
    print(f"  Sentiment       : {a['sentiment']}")
    print(f"  Consumer goal   : {a['consumer_goal']}")
    print(f"  Dollar amounts  : {a['dollar_amounts'] or 'none'}")
    print(f"  Contact attempts: {a['contact_attempts']}")
    print(f"  Account redacted: {a['account_numbers_redacted']}")
    print(f"  Contextual cues : {', '.join(a['contextual_cues']) if a['contextual_cues'] else 'none'}")
    print()
    print(sep)
    print("  AUTOMATED RESPONSE")
    print(sep)
    for line in result["automated_response"].splitlines():
        if line.strip():
            print(textwrap.fill(line, width=68,
                                initial_indent="  ",
                                subsequent_indent="  "))
        else:
            print()
    print(sep)


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def interactive_mode() -> None:
    print("\n" + "=" * 70)
    print("  CLAIRE — Consumer Complaint Auto-Responder")
    print("  Type your complaint and press Enter twice. Type 'quit' to exit.")
    print("=" * 70)

    session = 0
    while True:
        print("\nPaste your complaint narrative below (blank line to submit):")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.lower().strip() in {"quit", "exit", "q"}:
                print("Goodbye.")
                return
            if line == "" and lines:
                break
            lines.append(line)

        narrative = "\n".join(lines).strip()
        if not narrative:
            print("No input detected. Try again or type 'quit'.")
            continue

        session += 1
        cid = f"MANUAL-{session:03d}"

        try:
            result = analyse_and_respond(narrative, complaint_id=cid)
            print_result(result)
        except openai.AuthenticationError:
            print("ERROR: Invalid or missing OPENAI_API_KEY.")
            break
        except Exception as exc:
            print(f"ERROR: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLAIRE: Automated complaint responder")
    p.add_argument("--narrative", metavar="TEXT",
                   help="Complaint narrative (omit to launch interactive mode)")
    p.add_argument("--id", default="MANUAL-001", metavar="ID",
                   help="Complaint ID (default: MANUAL-001)")
    p.add_argument("--json", action="store_true", dest="json_out",
                   help="Output raw JSON instead of formatted text")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to your .env file.")
        raise SystemExit(1)

    if args.narrative:
        result = analyse_and_respond(args.narrative, complaint_id=args.id)
        if args.json_out:
            print(json.dumps(result, indent=2))
        else:
            print_result(result)
    else:
        interactive_mode()
