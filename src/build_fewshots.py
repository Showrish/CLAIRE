"""
CLAIRE — src/build_fewshots.py
Samples real CFPB complaint narratives from train.csv per issue category
to use as few-shot examples in the system prompt.

Outputs:
    data/processed/fewshot_examples.csv  — sampled examples with narratives
    prompts/fewshot_block.txt            — formatted few-shot block ready
                                           to paste into system_prompt.py

Usage:
    python src/build_fewshots.py
    python src/build_fewshots.py --n 3 --categories "Fraud or scam" "Managing an account"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SETTINGS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Issue categories we want few-shot examples for
# Ordered by value_counts() frequency in the real dataset
DEFAULT_CATEGORIES = [
    "Managing, opening, or closing your mobile wallet account",
    "Fraud or scam",
    "Unauthorized transactions or other transaction problem",
    "Managing an account",
    "Account opening, closing, or management",
    "Wrong amount charged or received",
    "Billing disputes",
    "Opening an account",
    "Identity theft / Fraud / Embezzlement",
    "Charged fees or interest you didn't expect",
    "Unauthorized transactions/trans. issues",
    "Unauthorized withdrawals or charges",
]

NARRATIVE_MAX = SETTINGS["FEWSHOT_NARRATIVE_MAX_CHARS"]
N_PER_CAT     = SETTINGS["FEWSHOT_EXAMPLES_PER_CATEGORY"]
RANDOM_STATE  = SETTINGS["RANDOM_STATE"]


def sample_fewshots(
    train_path: str,
    categories: list[str],
    n: int = N_PER_CAT,
) -> pd.DataFrame:
    """Sample n examples per issue category from train.csv."""
    log.info("Loading train.csv from %s...", train_path)
    df = pd.read_csv(train_path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Find key columns
    narrative_col = next((c for c in df.columns if "narrative" in c.lower()), None)
    issue_col     = next((c for c in df.columns if c.lower() == "issue"), None)
    product_col   = next((c for c in df.columns if c.lower() == "product"), None)
    sub_prod_col  = next((c for c in df.columns if "sub-product" in c.lower() or
                          "sub_product" in c.lower()), None)
    sub_issue_col = next((c for c in df.columns if "sub-issue" in c.lower() or
                          "sub_issue" in c.lower()), None)

    if not all([narrative_col, issue_col, product_col]):
        raise ValueError("Could not find required columns in train.csv")

    # Drop rows without narrative
    df = df[df[narrative_col].notna() & (df[narrative_col].str.strip() != "")]
    log.info("Rows with narratives: %d", len(df))

    samples = []
    for cat in categories:
        cat_df = df[df[issue_col].str.strip() == cat]
        if len(cat_df) == 0:
            log.warning("No rows found for category: %s", cat)
            continue

        n_sample = min(n, len(cat_df))
        sampled  = cat_df.sample(n=n_sample, random_state=RANDOM_STATE)

        for _, row in sampled.iterrows():
            narrative = str(row.get(narrative_col, "")).strip()
            # Truncate long narratives
            if len(narrative) > NARRATIVE_MAX:
                narrative = narrative[:NARRATIVE_MAX] + "..."

            def safe_get(col):
                """Get column value safely — handles NaN (float) from CSV."""
                if col is None:
                    return ""
                val = row.get(col, "")
                return "" if pd.isna(val) else str(val).strip()

            samples.append({
                "issue":       safe_get(issue_col),
                "product":     safe_get(product_col),
                "sub_product": safe_get(sub_prod_col),
                "sub_issue":   safe_get(sub_issue_col),
                "narrative":   narrative,
                "complaint_id": row.get("Complaint ID", row.get("complaint_id", "N/A")),
            })
        log.info("Sampled %d examples for: %s", n_sample, cat)

    result_df = pd.DataFrame(samples)
    log.info("Total examples sampled: %d", len(result_df))
    return result_df


def format_fewshot_block(df: pd.DataFrame) -> str:
    """Format sampled examples into a few-shot block for the system prompt."""
    lines = ["=== FEW-SHOT EXAMPLES (sampled from real CFPB PayPal complaints) ===", ""]

    for i, row in df.iterrows():
        example = {
            "cfpb_product":    row["product"],
            "cfpb_sub_product": row["sub_product"],
            "cfpb_issues":     [row["issue"]],
            "cfpb_sub_issue":  row["sub_issue"],
            "cfpb_confidence": "high",
            "sla_days":        "<<FILL_SLA>>",
        }
        import json
        lines.append(f'NARRATIVE: "{row["narrative"]}"')
        lines.append(f"OUTPUT: {json.dumps(example)}")
        lines.append("")

    lines.append("=== END EXAMPLES ===")
    return "\n".join(lines)


def run(train_path: str, categories: list[str], n: int) -> None:
    df = sample_fewshots(train_path, categories, n)

    # Save CSV
    out_csv = SETTINGS["FEWSHOT_CSV"]
    df.to_csv(out_csv, index=False)
    log.info("Saved fewshot examples to: %s", out_csv)

    # Save formatted block
    block     = format_fewshot_block(df)
    block_out = SETTINGS["PROJECT_ROOT"] / "prompts" / "fewshot_block.txt"
    block_out.write_text(block)
    log.info("Saved formatted few-shot block to: %s", block_out)

    # Print preview
    print("\n" + "=" * 70)
    print("FEW-SHOT BLOCK PREVIEW (first 3 examples)")
    print("=" * 70)
    preview_rows = df.head(3)
    for _, row in preview_rows.iterrows():
        print(f"\nISSUE   : {row['issue']}")
        print(f"PRODUCT : {row['product']}")
        print(f"NARRATIVE: {row['narrative'][:200]}...")
    print("\n" + "=" * 70)
    print(f"Next step: review prompts/fewshot_block.txt")
    print(f"Then paste approved examples into prompts/system_prompt.py")
    print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CLAIRE: Sample few-shot examples from train.csv"
    )
    p.add_argument("--train",  default=str(SETTINGS["TRAIN_CSV"]),
                   help="Path to train.csv")
    p.add_argument("--n",      type=int, default=N_PER_CAT,
                   help=f"Examples per category (default: {N_PER_CAT})")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                   help="Issue categories to sample (default: all failing categories)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.train, args.categories, args.n)