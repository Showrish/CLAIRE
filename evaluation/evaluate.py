"""
CLAIRE — evaluation/evaluate.py
Standalone evaluation backend for the v4a CFPB-taxonomy pipeline.

Unlike src/complaint_parser.py's run_csv_eval (which logs to console and writes
CSVs to disk), this module RETURNS structured results so a UI (Streamlit) or a
notebook can render them and offer downloads. The scoring semantics are identical
to v4a: exact + near-equivalence Issue matching, exact Product matching, and the
documented DATASET_INCONSISTENCY auto-categorisation for near-equivalent matches.

Reuses the v4a building blocks rather than re-implementing them:
    row_to_record, extract_regex_fields, extract_llm_fields, _issue_match

CLI (optional — mirrors the old `python src/complaint_parser.py --eval`):
    python evaluation/evaluate.py --input data/processed/test.csv --min-per-category 10
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

# Allow imports from project root (config/, prompts/, src/) when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import SETTINGS
from src.complaint_parser import (
    _PROMPT_VERSION,
    _issue_match,
    extract_llm_fields,
    extract_regex_fields,
    row_to_record,
)


# ---------------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------------

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same column normalisation the v4a pipeline expects."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("?", "") for c in df.columns]
    return df


def inspect_columns(df: pd.DataFrame) -> dict:
    """Report which evaluation columns are present in a (already-normalised) df.

    Returns a dict describing capability:
        narrative_col : name of the narrative column, or None
        issue_col     : 'issue' if present, else None  (needed for issue accuracy)
        product_col   : 'product' if present, else None (needed for product accuracy)
        id_col        : 'complaint_id' if present, else None
        can_eval      : True if at minimum narrative + issue are present
    """
    cols = set(df.columns)
    narrative_col = next((c for c in df.columns if "narrative" in c), None)
    issue_col = "issue" if "issue" in cols else None
    product_col = "product" if "product" in cols else None
    id_col = "complaint_id" if "complaint_id" in cols else None
    return {
        "narrative_col": narrative_col,
        "issue_col": issue_col,
        "product_col": product_col,
        "id_col": id_col,
        "can_eval": bool(narrative_col and issue_col),
    }


def _synthesise_id(narrative: str, idx: int) -> str:
    """Stable per-row id when the CSV has no complaint_id.

    Hashing the narrative keeps the cache key stable across re-uploads of the
    same data (so re-runs hit the file cache), while staying unique per row.
    Without this, blank ids collapse to one cache key and every row returns
    the first row's cached classification.
    """
    h = hashlib.md5(narrative.encode("utf-8")).hexdigest()[:12]
    return f"UP-{idx:05d}-{h}"


# ---------------------------------------------------------------------------
# Sampling (mirrors run_csv_eval semantics)
# ---------------------------------------------------------------------------

def sample_rows(
    df: pd.DataFrame,
    issue_col: str,
    limit: int = 30,
    min_per_category: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified-with-min-per-category sampling, identical to v4a's logic."""
    if min_per_category > 0:
        parts = []
        remaining = set(df.index)
        for _cat, cat_df in df.groupby(issue_col):
            n = min(min_per_category, len(cat_df))
            picked = cat_df.sample(n=n, random_state=random_state)
            parts.append(picked)
            remaining -= set(picked.index)
        sampled = pd.concat(parts)

        if limit > len(sampled) and remaining:
            remaining_df = df.loc[list(remaining)]
            extra_n = min(limit - len(sampled), len(remaining_df))
            extra = remaining_df.sample(n=extra_n, random_state=random_state)
            sampled = pd.concat([sampled, extra])
        return sampled.reset_index(drop=True)

    if len(df) < limit:
        limit = len(df)
    return df.sample(n=limit, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core evaluation — returns structured results, writes nothing
# ---------------------------------------------------------------------------

def run_evaluation(
    df: pd.DataFrame,
    limit: int = 30,
    min_per_category: int = 0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Evaluate the v4a pipeline on a labelled CFPB dataframe.

    Args:
        df: raw uploaded dataframe (columns will be normalised here).
        limit: total rows to evaluate.
        min_per_category: guarantee >= this many rows per Issue category.
        progress_callback: optional fn(done, total, complaint_id) for UI progress.

    Returns a dict:
        summary        : overall counts + accuracies + thresholds met
        per_issue      : list of {issue, correct, total, accuracy}
        error_rows     : DataFrame of misclassified + near-equivalent rows
        eval_results   : DataFrame (one OVERALL row + per-issue rows) for download
        meta           : prompt_version, sample size, small-sample warning, etc.
    """
    df = normalise_columns(df)
    info = inspect_columns(df)

    if not info["narrative_col"]:
        raise ValueError("No narrative column found (need a column containing 'narrative').")
    if not info["issue_col"]:
        raise ValueError("No 'issue' column found — required as ground truth for scoring.")

    narrative_col = info["narrative_col"]
    issue_col = info["issue_col"]
    score_product = info["product_col"] is not None

    # Keep only rows that actually have a narrative
    df = df[df[narrative_col].notna() & (df[narrative_col].astype(str).str.strip() != "")]
    available = len(df)

    sample_df = sample_rows(df, issue_col, limit=limit, min_per_category=min_per_category)
    total = len(sample_df)

    product_correct = 0
    issue_correct = 0
    issue_results: dict[str, dict[str, int]] = {}
    failed = 0
    error_rows: list[dict] = []

    for idx, row in sample_df.iterrows():
        record = row_to_record(row)

        # Guarantee a unique, stable cache key even when the CSV has no id
        if not record.complaint_id.strip():
            record.complaint_id = _synthesise_id(record.raw_narrative, idx)

        if not record.raw_narrative.strip():
            continue

        extract_regex_fields(record)
        extract_llm_fields(record)

        if progress_callback:
            progress_callback(idx + 1, total, record.complaint_id)

        # Skip rows where the LLM call failed entirely
        if not record.cfpb_product and not record.cfpb_issue:
            failed += 1
            continue

        prod_match = (
            record.cfpb_product.strip().lower() == record.product.strip().lower()
            if score_product else None
        )
        issue_match, match_type = _issue_match(record.issue, record.cfpb_issues)

        if score_product and prod_match:
            product_correct += 1
        if issue_match:
            issue_correct += 1

        gt_issue = record.issue.strip() or "unknown"
        issue_results.setdefault(gt_issue, {"correct": 0, "total": 0})
        issue_results[gt_issue]["total"] += 1
        if issue_match:
            issue_results[gt_issue]["correct"] += 1

        # Log misclassified AND near-equivalent rows (same as v4a error log)
        if (not issue_match) or match_type == "NEAR_EQUIVALENT":
            auto_cat = ""
            taxonomy_note = ""
            if match_type == "NEAR_EQUIVALENT":
                auto_cat = "DATASET_INCONSISTENCY"
                taxonomy_note = (
                    f"Near-equivalent group: model predicted a label in the same "
                    f"semantic group as ground truth. GT: '{record.issue}' | "
                    f"Pred: '{record.cfpb_issues[0] if record.cfpb_issues else ''}'"
                )
            error_rows.append({
                "complaint_id": record.complaint_id,
                "gt_product": record.product,
                "predicted_product": record.cfpb_product,
                "product_match": prod_match,
                "gt_issue": record.issue,
                "predicted_issues": " | ".join(record.cfpb_issues),
                "issue_match": issue_match,
                "match_type": match_type,
                "cfpb_confidence": record.cfpb_confidence,
                "sla_days": record.sla_days,
                "urgency": record.urgency_level,
                "sentiment": record.sentiment,
                "dollar_amounts": ", ".join(record.dollar_amounts) if record.dollar_amounts else "",
                "contact_attempts": record.contact_attempts,
                "narrative_snippet": record.raw_narrative[:300].replace("\n", " "),
                "model_reasoning": record.reasoning,
                "error_category": auto_cat,
                "cfpb_taxonomy_note": taxonomy_note,
            })

    evaluated = total - failed
    product_acc = (product_correct / evaluated) if (evaluated and score_product) else 0.0
    issue_acc = (issue_correct / evaluated) if evaluated else 0.0

    per_issue = [
        {
            "issue": issue,
            "correct": c["correct"],
            "total": c["total"],
            "accuracy": (c["correct"] / c["total"]) if c["total"] else 0.0,
        }
        for issue, c in sorted(issue_results.items(), key=lambda x: -x[1]["total"])
    ]

    summary = {
        "sample_size": total,
        "evaluated": evaluated,
        "failed": failed,
        "scored_product": score_product,
        "product_correct": product_correct,
        "product_accuracy": product_acc,
        "issue_correct": issue_correct,
        "issue_accuracy": issue_acc,
        "product_threshold": SETTINGS["MIN_PRODUCT_ACCURACY"],
        "issue_threshold": SETTINGS["MIN_ISSUE_ACCURACY"],
        "product_threshold_met": score_product and product_acc >= SETTINGS["MIN_PRODUCT_ACCURACY"],
        "issue_threshold_met": issue_acc >= SETTINGS["MIN_ISSUE_ACCURACY"],
    }

    # Build the downloadable eval_results table (OVERALL + per-issue), matching v4a columns
    results_rows = [{
        "prompt_version": _PROMPT_VERSION,
        "sample_size": evaluated,
        "failed_rows": failed,
        "product_accuracy": f"{product_acc * 100:.1f}%" if score_product else "n/a (no product column)",
        "issue_accuracy": f"{issue_acc * 100:.1f}%",
        "category": "OVERALL",
        "cat_correct": issue_correct,
        "cat_total": evaluated,
        "cat_accuracy": f"{issue_acc * 100:.1f}%",
        "notes": "near-equivalence scoring applied",
    }]
    for r in per_issue:
        results_rows.append({
            "prompt_version": _PROMPT_VERSION,
            "sample_size": evaluated,
            "failed_rows": failed,
            "product_accuracy": f"{product_acc * 100:.1f}%" if score_product else "n/a",
            "issue_accuracy": f"{issue_acc * 100:.1f}%",
            "category": r["issue"],
            "cat_correct": r["correct"],
            "cat_total": r["total"],
            "cat_accuracy": f"{r['accuracy'] * 100:.1f}%",
            "notes": "",
        })

    meta = {
        "prompt_version": _PROMPT_VERSION,
        "rows_available": available,
        "small_sample_warning": total < 40,
        "score_product": score_product,
        "id_synthesised": info["id_col"] is None,
    }

    return {
        "summary": summary,
        "per_issue": per_issue,
        "error_rows": pd.DataFrame(error_rows),
        "eval_results": pd.DataFrame(results_rows),
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Optional CLI (parity with the old --eval flow, but prints the structured result)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="CLAIRE standalone evaluation")
    p.add_argument("--input", default=str(SETTINGS["TEST_CSV"]), help="Path to labelled test CSV")
    p.add_argument("--limit", type=int, default=30, help="Total rows to evaluate")
    p.add_argument("--min-per-category", type=int, default=0, dest="min_per_category",
                   help="Minimum rows per issue category (0 = random sample)")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (add it to .env).")
        raise SystemExit(1)

    frame = pd.read_csv(args.input, dtype=str, low_memory=False)
    out = run_evaluation(frame, limit=args.limit, min_per_category=args.min_per_category,
                         progress_callback=lambda d, t, cid: print(f"  [{d}/{t}] {cid}"))
    s = out["summary"]
    print("\n" + "=" * 60)
    print(f"  Prompt version : {out['meta']['prompt_version']}")
    print(f"  Evaluated      : {s['evaluated']} (failed: {s['failed']})")
    if s["scored_product"]:
        print(f"  Product accuracy: {s['product_accuracy']*100:.1f}%  "
              f"({'PASS' if s['product_threshold_met'] else 'below threshold'})")
    else:
        print("  Product accuracy: n/a (no 'product' column in upload)")
    print(f"  Issue accuracy : {s['issue_accuracy']*100:.1f}%  "
          f"({'PASS' if s['issue_threshold_met'] else 'below threshold'})")
    print("=" * 60)
    for r in out["per_issue"]:
        print(f"    {r['issue'][:55]:<55} {r['correct']}/{r['total']} ({r['accuracy']*100:.0f}%)")
    if out["meta"]["small_sample_warning"]:
        print("\n  WARNING: <40 rows — accuracy may be overstated. Use --min-per-category 10.")
