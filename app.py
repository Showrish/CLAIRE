"""
CLAIRE — Streamlit deployment
Run from the project root:  streamlit run app.py

Two tabs:
  • Triage   — single complaint → CFPB classification, SLA TAT, drafted response
               (UI adapted from Kai's reference app.py; pipeline = src/respond.py)
  • Evaluate — upload a labelled CFPB CSV → accuracy vs ground truth with
               near-equivalence scoring, per-issue breakdown, downloadable logs
               (backend = evaluation/evaluate.py)

Project layout is unchanged: app.py sits at the root next to src/, prompts/,
config/, evaluation/. The API key is read from st.secrets or the environment;
a password field is offered as a fallback for quick local runs.
"""

import html
import io
import json
import os
import sys
from pathlib import Path

import streamlit as st

# Make config/, prompts/, src/, evaluation/ importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── API key: st.secrets → env (must be set BEFORE importing the pipeline,
#    which reads os.environ["OPENAI_API_KEY"] at call time) ──────────────────
try:
    if "OPENAI_API_KEY" in st.secrets and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # no secrets.toml present — fall back to env / text input below

from config.settings import SETTINGS
from src.complaint_parser import _PROMPT_VERSION
from src.respond import analyse_and_respond
from evaluation.evaluate import inspect_columns, normalise_columns, run_evaluation


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLAIRE — Complaint Triage & Evaluation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu                        { display: none !important; }
footer                           { display: none !important; }

.claire-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    background-color: var(--secondary-background-color);
}
.claire-card .card-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; opacity: 0.6; margin-bottom: 5px;
}
.claire-card .card-value {
    font-size: 1rem; font-weight: 700; color: var(--text-color);
    white-space: normal; word-break: normal; overflow-wrap: break-word; line-height: 1.4;
}
.badge { display:inline-block; padding:3px 12px; border-radius:12px; font-size:0.78rem; font-weight:700; }
.badge-high   { background:#e53935; color:#fff; }
.badge-medium { background:#fb8c00; color:#fff; }
.badge-low    { background:#43a047; color:#fff; }
.badge-pass   { background:#43a047; color:#fff; }
.badge-fail   { background:#e53935; color:#fff; }
</style>
""", unsafe_allow_html=True)


# ── Helpers (from Kai's reference UI) ──────────────────────────────────────
def card(label: str, value: str, raw_html: bool = False) -> str:
    val = value if raw_html else html.escape(str(value))
    return (
        f'<div class="claire-card">'
        f'<div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{val}</div>'
        f'</div>'
    )

def badge(level: str) -> str:
    return f'<span class="badge badge-{level.lower()}">{level.upper()}</span>'

SENTIMENT = {
    "angry": "😡 Angry", "frustrated": "😤 Frustrated",
    "neutral": "😐 Neutral", "confused": "🤔 Confused",
}


# ── Header ─────────────────────────────────────────────────────────────────
st.title("⚖️ CLAIRE")
st.caption(f"Consumer Language AI for Intelligent Reasoning and Extraction  ·  prompt {_PROMPT_VERSION}")

# ── API key gate (fallback only — secrets/env take priority) ───────────────
if not os.environ.get("OPENAI_API_KEY", "").strip():
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-proj-…")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key.strip()
    else:
        st.warning("Enter your OpenAI API key to use CLAIRE "
                   "(or set it in .streamlit/secrets.toml for deployment).", icon="⚠️")
        st.stop()

st.divider()

tab_triage, tab_eval = st.tabs(["📝 Triage", "📊 Evaluate"])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TRIAGE  (single complaint → classify + SLA + response)
# ════════════════════════════════════════════════════════════════════════════
with tab_triage:
    input_col, result_col = st.columns([1, 1], gap="large")

    with input_col:
        st.subheader("📝 Complaint Narrative")
        narrative = st.text_area(
            label="Narrative", label_visibility="collapsed", height=220,
            placeholder=(
                "Example: Someone logged into my PayPal account without permission and sent "
                "$650 to an unknown recipient. I called three times and emailed twice but the "
                "case was closed without explanation. I need the $650 refunded immediately."
            ),
        )
        complaint_id = st.text_input("Complaint ID (optional)", value="MANUAL-001")
        submit = st.button("⚡ Analyze & Respond", type="primary", use_container_width=True)

    with result_col:
        if submit:
            if not narrative.strip():
                st.warning("Please enter a complaint narrative before submitting.")
                st.stop()
            with st.spinner("Analyzing complaint…"):
                try:
                    result = analyse_and_respond(
                        narrative.strip(),
                        complaint_id=complaint_id.strip() or "MANUAL-001",
                    )
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.stop()

            a = result["analysis"]
            dollar_text = ", ".join(a["dollar_amounts"]) if a["dollar_amounts"] else "None detected"

            st.subheader("🔍 Analysis")
            r1c1, r1c2 = st.columns(2)
            r1c1.markdown(card("🏦 CFPB Product", a["cfpb_product"] or "—"), unsafe_allow_html=True)
            r1c2.markdown(card("🚨 Urgency", badge(a["urgency_level"] or "low"), raw_html=True), unsafe_allow_html=True)

            r2c1, r2c2 = st.columns(2)
            r2c1.markdown(card("⚠️ CFPB Issue", (a["cfpb_issues"][0] if a["cfpb_issues"] else "—")), unsafe_allow_html=True)
            r2c2.markdown(card("💬 Sentiment", SENTIMENT.get(a["sentiment"], (a["sentiment"] or "neutral").title())), unsafe_allow_html=True)

            r3c1, r3c2 = st.columns(2)
            r3c1.markdown(card("📅 SLA / TAT", f"{a['sla_days']} days"), unsafe_allow_html=True)
            r3c2.markdown(card("🎯 Confidence", (a["cfpb_confidence"] or "—").title()), unsafe_allow_html=True)

            r4c1, r4c2 = st.columns(2)
            r4c1.markdown(card("🎯 Consumer Goal", (a["consumer_goal"] or "—").replace("_", " ").title()), unsafe_allow_html=True)
            r4c2.markdown(card("📞 Contact Attempts", str(a["contact_attempts"])), unsafe_allow_html=True)

            st.markdown(card("💵 Dollar Amounts", dollar_text), unsafe_allow_html=True)
            if a["cfpb_sub_product"] or a["cfpb_sub_issue"]:
                st.caption(f"Sub-product: {a['cfpb_sub_product'] or '—'}  ·  Sub-issue: {a['cfpb_sub_issue'] or '—'}")
            if a["contextual_cues"]:
                st.markdown("💡 **Cues:** " + " · ".join(a["contextual_cues"]))
            if a["dates_mentioned"]:
                st.markdown("📅 **Dates:** " + " · ".join(f"`{d}`" for d in a["dates_mentioned"]))
            if a["account_numbers_redacted"]:
                st.info("🔒 Account numbers redacted (XXXX).")

            st.divider()
            st.subheader("📨 Automated Response")
            with st.container(border=True):
                st.markdown(result["automated_response"])

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "⬇ Download JSON",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name=f"claire_{complaint_id or 'manual'}.json",
                mime="application/json", use_container_width=True,
            )
            dl2.download_button(
                "⬇ Download Response (TXT)",
                data=result["automated_response"],
                file_name=f"response_{complaint_id or 'manual'}.txt",
                mime="text/plain", use_container_width=True,
            )
        else:
            st.info("Enter a complaint on the left and click **Analyze & Respond**.", icon="👈")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EVALUATE  (upload labelled CSV → accuracy with near-equivalence)
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("📊 Batch Evaluation")
    st.caption(
        "Upload a labelled CFPB CSV. Needs a **narrative** column and an **Issue** column "
        "(ground truth). A **Product** column enables product accuracy too. Each row is a "
        "live model call — scoring uses the v4a near-equivalence groups."
    )

    uploaded = st.file_uploader("Upload labelled CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        st.info("Upload a CSV to begin. Tip: your `data/processed/test.csv` is the canonical set.", icon="📄")
    else:
        try:
            raw_df = __import__("pandas").read_csv(uploaded, dtype=str, low_memory=False)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        norm_df = normalise_columns(raw_df)
        info = inspect_columns(norm_df)

        if not info["can_eval"]:
            missing = []
            if not info["narrative_col"]:
                missing.append("a narrative column (name containing 'narrative')")
            if not info["issue_col"]:
                missing.append("an 'Issue' column (ground-truth labels)")
            st.error("This file can't be scored — missing " + " and ".join(missing) + ".")
            st.dataframe(raw_df.head(5), use_container_width=True)
            st.stop()

        # Capability summary
        c1, c2, c3 = st.columns(3)
        c1.success(f"Narrative: `{info['narrative_col']}`")
        c2.success("Issue accuracy: available")
        c3.success("Product accuracy: available") if info["product_col"] else \
            c3.warning("Product accuracy: no `product` column — issue-only")
        if info["id_col"] is None:
            st.caption("No `complaint_id` column — stable IDs will be synthesised from each narrative (cache-safe).")

        with st.expander("Preview (first 5 rows)"):
            st.dataframe(raw_df.head(5), use_container_width=True)

        n_rows = len(norm_df)
        n_cats = norm_df[info["issue_col"]].nunique()
        st.markdown(f"**{n_rows:,}** rows · **{n_cats}** issue categories")

        # ── Controls ───────────────────────────────────────────────────────
        st.markdown("##### Run settings")
        s1, s2, s3 = st.columns(3)
        min_per_cat = s1.number_input("Min rows per category", min_value=0, max_value=50, value=10,
                                      help="Guarantees coverage of rare categories. 10 = the recommended final-eval setting.")
        limit = s2.number_input("Row cap (total)", min_value=5, max_value=300, value=50,
                                help="Hard ceiling on rows evaluated, to bound time and API cost.")
        throttle = s3.number_input("Per-call delay (sec)", min_value=0.0, max_value=30.0, value=1.0, step=0.5,
                                   help="Free OpenAI tier (3 RPM) needs ~20s. Paid tier: 0–1s. This is the main time driver.")

        # Time estimate — rows actually run ≈ min(cap, max(min_per_cat*cats, ...))
        est_rows = min(int(limit), max(int(min_per_cat) * int(n_cats), int(limit)) if min_per_cat else int(limit))
        est_rows = min(est_rows, n_rows)
        per_row = float(throttle) + 1.5  # throttle + rough call latency
        est_min = est_rows * per_row / 60.0
        st.caption(f"≈ {est_rows} rows × ~{per_row:.1f}s ≈ **{est_min:.1f} min** on a cold run "
                   f"(cached rows return instantly). Results cached under prompt {_PROMPT_VERSION}.")

        run = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

        if run:
            # Apply the throttle globally for this run (pipeline reads SETTINGS at call time)
            SETTINGS["SLEEP_BETWEEN_CALLS"] = float(throttle)
            SETTINGS["SLEEP_RESPONSE_GEN"] = float(throttle)

            prog = st.progress(0.0, text="Starting…")

            def _on_progress(done: int, total: int, cid: str):
                prog.progress(min(done / total, 1.0), text=f"Classifying {done}/{total}  ·  {cid}")

            try:
                out = run_evaluation(
                    norm_df, limit=int(limit), min_per_category=int(min_per_cat),
                    progress_callback=_on_progress,
                )
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.stop()
            finally:
                prog.empty()

            s = out["summary"]
            meta = out["meta"]

            # ── Headline metrics ────────────────────────────────────────────
            st.markdown("#### Results")
            m1, m2, m3 = st.columns(3)

            if s["scored_product"]:
                pstat = "pass" if s["product_threshold_met"] else "fail"
                m1.markdown(card("PRODUCT ACCURACY",
                                 f"{s['product_accuracy']*100:.0f}%  "
                                 f"<span class='badge badge-{pstat}'>"
                                 f"{'PASS' if pstat=='pass' else 'BELOW'}</span>",
                                 raw_html=True), unsafe_allow_html=True)
            else:
                m1.markdown(card("PRODUCT ACCURACY", "n/a — no product column"), unsafe_allow_html=True)

            istat = "pass" if s["issue_threshold_met"] else "fail"
            m2.markdown(card("ISSUE ACCURACY",
                             f"{s['issue_accuracy']*100:.0f}%  "
                             f"<span class='badge badge-{istat}'>"
                             f"{'PASS' if istat=='pass' else 'BELOW'}</span>",
                             raw_html=True), unsafe_allow_html=True)
            m3.markdown(card("ROWS EVALUATED",
                             f"{s['evaluated']}  (failed: {s['failed']})"), unsafe_allow_html=True)

            st.caption(f"Thresholds: product ≥ {s['product_threshold']*100:.0f}% · "
                       f"issue ≥ {s['issue_threshold']*100:.0f}% · near-equivalence scoring applied.")

            if meta["small_sample_warning"]:
                st.warning("Fewer than 40 rows evaluated — accuracy can be overstated on this "
                           "imbalanced dataset. Raise the row cap / min-per-category for a reliable number.",
                           icon="⚠️")

            # ── Per-issue breakdown ─────────────────────────────────────────
            st.markdown("##### Per-issue breakdown")
            pi = out["per_issue"]
            pi_df = __import__("pandas").DataFrame([{
                "Issue": r["issue"],
                "Correct": r["correct"],
                "Total": r["total"],
                "Accuracy": f"{r['accuracy']*100:.0f}%",
            } for r in pi])
            st.dataframe(pi_df, use_container_width=True, hide_index=True)

            # ── Error log + results downloads ───────────────────────────────
            err_df = out["error_rows"]
            res_df = out["eval_results"]
            st.markdown("##### Downloads")
            d1, d2 = st.columns(2)

            res_buf = io.StringIO(); res_df.to_csv(res_buf, index=False)
            d1.download_button("⬇ eval_results.csv", data=res_buf.getvalue(),
                               file_name="eval_results.csv", mime="text/csv",
                               use_container_width=True)

            if not err_df.empty:
                err_buf = io.StringIO(); err_df.to_csv(err_buf, index=False)
                d2.download_button(f"⬇ error_log.csv ({len(err_df)} rows)", data=err_buf.getvalue(),
                                   file_name="error_log.csv", mime="text/csv",
                                   use_container_width=True)
                with st.expander(f"Misclassified & near-equivalent rows ({len(err_df)})"):
                    st.dataframe(err_df, use_container_width=True, hide_index=True)
            else:
                d2.success("No misclassifications.")
