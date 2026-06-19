"""
CLAIRE — config/settings.py
Central configuration. All constants in one place.
SLA rules are loaded from config/sla_config.json so they can be
updated without touching Python code.

Usage:
    from config.settings import SETTINGS
    model = SETTINGS["MODEL_NAME"]
    sla  = SETTINGS["SLA_DAYS"]["unauthorized"]
"""

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent

# Load SLA config from JSON
_sla_path = _HERE / "sla_config.json"
with _sla_path.open() as _f:
    _sla_cfg = json.load(_f)

# Build a flat SLA lookup from the JSON rules
def _build_sla_lookup(rules: list, default: int) -> dict:
    lookup = {"default": default}
    for rule in rules:
        for kw in rule["issue_keywords"]:
            # Use first meaningful word as key
            key = kw.split()[0].lower().rstrip(",")
            lookup[key] = rule["sla_days"]
    return lookup

SETTINGS = {
    # ── Model ────────────────────────────────────────────────────────────────
    "MODEL_NAME":               "gpt-4o-mini-2024-07-18",  # pinned snapshot (alias 'gpt-4o-mini' resolves here) — stops silent model drift
    "TEMPERATURE":              0,
    "SEED":                     42,    # best-effort determinism on cache misses (used with temperature=0)
    "MAX_TOKENS":               600,
    "MAX_RETRIES":              0,

    # ── Rate limiting ────────────────────────────────────────────────────────
    "SLEEP_BETWEEN_CALLS":      20,    # seconds (free tier = 3 RPM)
    "SLEEP_RESPONSE_GEN":        3,
    "RETRY_MAX_ATTEMPTS":        8,
    "RETRY_WAIT_MIN":           25,
    "RETRY_WAIT_MAX":          120,

    # ── Paths ────────────────────────────────────────────────────────────────
    "PROJECT_ROOT":             PROJECT_ROOT,
    "CACHE_DIR":                PROJECT_ROOT / ".cache",
    "DATA_RAW_DIR":             PROJECT_ROOT / "data" / "raw",
    "DATA_PROCESSED_DIR":       PROJECT_ROOT / "data" / "processed",
    "TRAIN_CSV":                PROJECT_ROOT / "data" / "processed" / "train.csv",
    "TEST_CSV":                 PROJECT_ROOT / "data" / "processed" / "test.csv",
    "FEWSHOT_CSV":              PROJECT_ROOT / "data" / "processed" / "fewshot_examples.csv",

    # ── Data loading ─────────────────────────────────────────────────────────
    "CHUNK_SIZE":               5_000,
    "RANDOM_STATE":             42,
    "TEST_SIZE":                0.20,
    "NARRATIVE_MAX_CHARS":      1_500,
    "NARRATIVE_COL":            "Consumer complaint narrative",  # CFPB dataset column name
    "STRATIFY_COL":             "Issue",                         # Column used for stratified split

    # ── SLA (loaded from sla_config.json) ────────────────────────────────────
    "SLA_RULES":                _sla_cfg["sla_rules"],
    "SLA_DEFAULT_DAYS":         _sla_cfg["default_sla_days"],
    "SLA_DAYS":                 _build_sla_lookup(
                                    _sla_cfg["sla_rules"],
                                    _sla_cfg["default_sla_days"]
                                ),

    # ── Evaluation thresholds ────────────────────────────────────────────────
    "MIN_PRODUCT_ACCURACY":     0.80,
    "MIN_ISSUE_ACCURACY":       0.80,

    # ── Response generation ──────────────────────────────────────────────────
    "RESPONSE_MAX_TOKENS":      400,
    "RESPONSE_TEMPERATURE":     0.4,

    # ── Few-shot sampling ────────────────────────────────────────────────────
    "FEWSHOT_EXAMPLES_PER_CATEGORY": 2,   # how many examples to sample per issue
    "FEWSHOT_NARRATIVE_MAX_CHARS":   600, # truncate long narratives in few-shot examples
}

# Ensure runtime dirs exist
for _d in ["CACHE_DIR", "DATA_RAW_DIR", "DATA_PROCESSED_DIR"]:
    SETTINGS[_d].mkdir(parents=True, exist_ok=True)
