"""
CLAIRE split.py
Stratified train/test split of the CFPB complaints CSV.
Filters to PayPal-related complaints only, then further filters to
PayPal SLA-covered issue categories before splitting.

Usage:
    python src/split.py --input "data/raw/complaints.csv"

Outputs:
    data/processed/train.csv  — 80% stratified sample (PayPal + SLA issues only)
    data/processed/test.csv   — 20% stratified sample (use this for reasoning loop testing)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Allow imports from project root when running from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SETTINGS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NARRATIVE_COL = SETTINGS["NARRATIVE_COL"]
STRATIFY_COL  = SETTINGS["STRATIFY_COL"]
RANDOM_STATE  = SETTINGS["RANDOM_STATE"]
TEST_SIZE     = SETTINGS["TEST_SIZE"]

# ── PAYPAL FILTER ─────────────────────────────────────────────────────────────
# Exact company name as confirmed in the CFPB dataset (verified May 2026)
# Screenshot confirmed: "Paypal Holdings, Inc" (mixed case, with comma)
PAYPAL_EXACT_NAMES = [
    "paypal holdings, inc",
]

# Subsidiary/brand keywords — source: PayPal 10-K FY2024 (SEC filing)
# https://www.sec.gov/Archives/edgar/data/0001633917/000119312524090908/d554828ddef14a1.pdf
PAYPAL_COMPANY_KEYWORDS = [
    "paypal",        # PayPal Holdings, PayPal Inc, PayPal Credit etc.
    "venmo",         # PayPal-owned digital wallet
    "honey",         # PayPal-owned shopping/coupon platform
    "braintree",     # PayPal-owned payment gateway
    "xoom",          # PayPal-owned international money transfer
    "hyperwallet",   # PayPal-owned payout platform
    "zettle",        # PayPal-owned POS platform
    "paidy",         # PayPal-owned BNPL platform (Japan)
]

# ── SLA-COVERED ISSUE CATEGORIES ─────────────────────────────────────────────
# Based on PayPal's published dispute/claim categories with defined resolution SLAs:
#   Unauthorized transaction  → 10 business days (fraud claim)
#   Item Not Received (INR)   → 20 days from dispute open
#   Billing errors            → 20 days
#   Account management        → 20 days
# Source: PayPal Resolution Center & User Agreement
SLA_ISSUE_KEYWORDS = [
    # Unauthorized / Fraud
    "unauthorized",
    "fraud",
    "identity theft",
    "account takeover",
    # Item Not Received
    "item not received",
    "didn't receive",
    "not received",
    # Billing / Charge errors
    "incorrect amount",
    "duplicate",
    "wrong amount",
    "billing",
    "charged",
    # Cancelled / Subscription
    "canceled recurring",
    "cancelled recurring",
    "subscription",
    "recurring",
    # Credit not processed / Refund
    "credit not processed",
    "refund",
    "not as described",
    "significantly not",
    # Account access / freeze
    "account freeze",
    "account closed",
    "account limited",
    "managing",
    "opening",
]


def is_paypal(company: str) -> bool:
    """Return True if the company field refers to PayPal or a subsidiary.

    Checks both the exact confirmed CFPB company string ('Paypal Holdings, Inc')
    and subsidiary brand keywords (Venmo, Honey, Braintree, Xoom, etc.)
    Source for subsidiaries: PayPal 10-K FY2024 SEC filing.
    """
    if pd.isna(company):
        return False
    company_lower = str(company).lower()
    # Check exact known name first (fast path)
    if company_lower in PAYPAL_EXACT_NAMES:
        return True
    # Fall back to keyword match for subsidiaries
    return any(kw in company_lower for kw in PAYPAL_COMPANY_KEYWORDS)


def is_sla_issue(issue: str) -> bool:
    """Return True if the issue falls under a PayPal SLA-covered category."""
    if pd.isna(issue):
        return False
    issue_lower = str(issue).lower()
    return any(kw in issue_lower for kw in SLA_ISSUE_KEYWORDS)


def run_split(input_path: str) -> None:
    log.info("Loading complaints from %s (chunked, this may take a minute)...", input_path)

    chunks = []
    chunksize = 10_000
    total = 0

    for chunk in pd.read_csv(
        input_path,
        dtype=str,
        low_memory=False,
        chunksize=chunksize,
    ):
        # Normalise column names
        chunk.columns = [c.strip() for c in chunk.columns]

        # Keep only rows with a narrative
        narrative_col = next(
            (c for c in chunk.columns if "narrative" in c.lower()), None
        )
        if narrative_col is None:
            raise ValueError("Could not find narrative column. Check CSV headers.")

        chunk = chunk[
            chunk[narrative_col].notna() &
            (chunk[narrative_col].str.strip() != "")
        ]
        chunks.append(chunk)
        total += len(chunk)
        log.info("  ... loaded %d rows with narratives so far", total)

    df = pd.concat(chunks, ignore_index=True)
    log.info("Total rows with narratives: %d", len(df))

    # ── STEP 1: FILTER TO PAYPAL ONLY ─────────────────────────────────────────
    company_col = next(
        (c for c in df.columns if c.lower() == "company"), None
    )
    if company_col is None:
        raise ValueError("Could not find 'Company' column. Check CSV headers.")

    before = len(df)
    df = df[df[company_col].apply(is_paypal)]
    log.info(
        "After PayPal filter: %d rows (removed %d non-PayPal rows)",
        len(df), before - len(df)
    )

    if len(df) == 0:
        raise ValueError(
            "No PayPal rows found. Check that the 'Company' column "
            "contains PayPal entries (e.g. 'PAYPAL, INC.')."
        )

    # ── STEP 2: FILTER TO SLA-COVERED ISSUES ONLY ─────────────────────────────
    issue_col = next(
        (c for c in df.columns if c.lower() == "issue"), None
    )
    if issue_col is None:
        raise ValueError("Could not find 'Issue' column. Check CSV headers.")

    before = len(df)
    df = df[df[issue_col].apply(is_sla_issue)]
    log.info(
        "After SLA issue filter: %d rows (removed %d out-of-scope issue rows)",
        len(df), before - len(df)
    )

    if len(df) == 0:
        raise ValueError(
            "No rows matched the SLA issue filter. "
            "Check the Issue column values in your dataset."
        )

    # Log the issue distribution after filtering
    log.info("\nIssue distribution after filtering:")
    for issue, count in df[issue_col].value_counts().items():
        log.info("  %-60s %d", issue, count)

    # ── STEP 3: STRATIFIED TRAIN/TEST SPLIT ───────────────────────────────────
    # Drop rows where issue is null
    df = df.dropna(subset=[issue_col])

    # Remove issue categories with fewer than 2 samples (can't stratify)
    counts = df[issue_col].value_counts()
    valid = counts[counts >= 2].index
    removed = counts[counts < 2].index.tolist()
    if removed:
        log.info("Removing rare issue categories (< 2 samples): %s", removed)
    df = df[df[issue_col].isin(valid)]
    log.info("Rows after removing rare issue categories: %d", len(df))

    log.info(
        "Splitting: 80%% train / 20%% test (stratified on Issue, random_state=%d)...",
        RANDOM_STATE
    )
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[issue_col],
    )

    log.info("Train set: %d rows", len(train_df))
    log.info("Test set : %d rows", len(test_df))

    # Verify stratification
    log.info("\nIssue distribution in test set:")
    dist = test_df[issue_col].value_counts(normalize=True).round(3)
    for issue, pct in dist.items():
        log.info("  %-60s %.1f%%", issue, pct * 100)

    # Save to data/processed/
    train_out = SETTINGS["TRAIN_CSV"]
    test_out  = SETTINGS["TEST_CSV"]
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out,  index=False)
    log.info(
        "\nSaved train.csv (%d rows) → %s",
        len(train_df), train_out
    )
    log.info(
        "Saved test.csv  (%d rows) → %s",
        len(test_df), test_out
    )

    # Print a random example from test set
    narrative_col = next(c for c in df.columns if "narrative" in c.lower())
    sample = test_df.sample(1, random_state=RANDOM_STATE).iloc[0]
    print("\n" + "=" * 70)
    print("RANDOM TEST SAMPLE (PayPal | SLA-covered issue)")
    print("=" * 70)
    print(f"Complaint ID : {sample.get('Complaint ID', 'N/A')}")
    print(f"Company      : {sample.get(company_col, 'N/A')}")
    print(f"Issue        : {sample.get(issue_col, 'N/A')}")
    print(f"Narrative    :\n{sample.get(narrative_col, '')[:500]}...")
    print("=" * 70)
    print("\nPaste the narrative above into src/complaint_parser.py to test.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CLAIRE: Stratified train/test split (PayPal + SLA issues only)"
    )
    p.add_argument(
        "--input", required=True, metavar="FILE",
        help="Path to CFPB complaints CSV"
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_split(args.input)
