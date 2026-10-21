"""
CLAIRE — Streamlit deployment
Run from the project root:  streamlit run app.py

Three tabs:
  • Assistant — a conversational, consumer-facing chat. The person describes what
                happened and CLAIRE replies like a caring support agent (warm message
                only). Triage details are tucked away for staff.
  • Triage    — the staff/analyst view: full CFPB classification, SLA, and the drafted
                message, shown as a structured card.
  • Evaluate  — upload a labelled CFPB CSV and score accuracy with near-equivalence.

API key is read from st.secrets / env; a password field is offered as a fallback.
"""

import html
import io
import json
import os
import sys
from pathlib import Path

import streamlit as st


def esc(text):
    """Escape text for HTML and neutralise $ so Streamlit doesn't render it as math."""
    return html.escape(str(text)).replace("$", "&#36;")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ── API key resolution: env → .streamlit/secrets.toml (BOM-tolerant) ────────
def _resolve_api_key():
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    sp = ROOT / ".streamlit" / "secrets.toml"
    if not sp.exists():
        return None
    try:
        text = sp.read_text(encoding="utf-8-sig")
        try:
            import tomllib
            data = tomllib.loads(text)
        except ModuleNotFoundError:
            import toml
            data = toml.loads(text)
    except Exception as exc:
        st.error(f"`.streamlit/secrets.toml` could not be parsed: {exc}", icon="🚫")
        return None
    key = str(data.get("OPENAI_API_KEY", "")).strip()
    if key:
        return key
    return None

_key = _resolve_api_key()
if _key:
    os.environ["OPENAI_API_KEY"] = _key

from config.settings import SETTINGS
from src.complaint_parser import _PROMPT_VERSION
from src.respond import analyse_and_respond
from evaluation.evaluate import inspect_columns, normalise_columns, run_evaluation

# Responsive interactive defaults (Evaluate tab overrides per-run). Assumes a paid
# OpenAI tier; raise these if you are on the free 3-RPM tier.
SETTINGS["SLEEP_BETWEEN_CALLS"] = 0.5
SETTINGS["SLEEP_RESPONSE_GEN"] = 0.5


# ── Page config + styling ───────────────────────────────────────────────────
st.set_page_config(page_title="CLAIRE", page_icon="⚖️", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"], #MainMenu, footer { display:none !important; }
.claire-card { border:1px solid rgba(128,128,128,0.25); border-radius:10px; padding:14px 18px;
  margin-bottom:8px; background-color: var(--secondary-background-color); }
.claire-card .card-label { font-size:0.72rem; font-weight:700; text-transform:uppercase;
  letter-spacing:0.07em; opacity:0.6; margin-bottom:5px; }
.claire-card .card-value { font-size:1rem; font-weight:700; color:var(--text-color);
  overflow-wrap:break-word; line-height:1.4; }
.badge { display:inline-block; padding:3px 12px; border-radius:12px; font-size:0.78rem; font-weight:700; }
.badge-high{background:#e53935;color:#fff;} .badge-medium{background:#fb8c00;color:#fff;}
.badge-low{background:#43a047;color:#fff;} .badge-pass{background:#43a047;color:#fff;}
.badge-fail{background:#e53935;color:#fff;}
.chip { display:inline-block; padding:4px 12px; border-radius:14px; font-size:0.8rem;
  background:rgba(20,184,166,0.15); color:#0D9488; font-weight:600; margin-right:6px; }
.msg-sent { font-size:1.05rem; line-height:1.5; }
</style>
""", unsafe_allow_html=True)

def card(label, value, raw=False):
    v = value if raw else esc(str(value))
    return f'<div class="claire-card"><div class="card-label">{esc(label)}</div><div class="card-value">{v}</div></div>'

def badge(level):
    return f'<span class="badge badge-{level.lower()}">{level.upper()}</span>'

SENTIMENT = {"angry":"😡 Angry","frustrated":"😤 Frustrated","neutral":"😐 Neutral","confused":"🤔 Confused"}

st.title("⚖️ CLAIRE")
st.caption(f"Consumer Language AI for Intelligent Reasoning and Extraction  ·  prompt {_PROMPT_VERSION}")

# ── API key gate ────────────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY", "").strip():
    k = st.text_input("OpenAI API Key", type="password", placeholder="sk-proj-…")
    if k:
        os.environ["OPENAI_API_KEY"] = k.strip()
    else:
        st.warning("Enter your OpenAI API key to use CLAIRE (or set it in .streamlit/secrets.toml).", icon="⚠️")
        st.stop()

st.divider()
tab_chat, tab_analysis, tab_eval = st.tabs(["💬 Assistant", "🔍 Analysis (customer service staff)", "📊 Evaluate (development team)"])


# ════════════════════════════════════════════════════════════ TAB 1 — ASSISTANT (chat)
with tab_chat:
    INTRO = ("Hi, I'm CLAIRE. I'm sorry you're dealing with something — I'm here to help. "
             "Tell me what happened in your own words, and I'll make sure it's logged and looked into.")
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "text": INTRO}]

    # render history
    for m in st.session_state.chat:
        with st.chat_message(m["role"], avatar="⚖️" if m["role"] == "assistant" else "🧑"):
            st.markdown(f'<div class="msg-sent">{esc(m["text"])}</div>', unsafe_allow_html=True)
            if m.get("chip"):
                st.markdown(m["chip"], unsafe_allow_html=True)
            if m.get("analysis"):
                a = m["analysis"]
                with st.expander("Triage details (staff only)"):
                    st.markdown(
                        f"**Issue:** {(a['cfpb_issues'][0] if a['cfpb_issues'] else '—')}  \n"
                        f"**Product:** {a['cfpb_product'] or '—'}  \n"
                        f"**SLA / resolution target:** {a['sla_days']} days  \n"
                        f"**Urgency:** {a['urgency_level'] or '—'}  ·  "
                        f"**Sentiment:** {a['sentiment'] or '—'}  ·  "
                        f"**Confidence:** {a['cfpb_confidence'] or '—'}")

    prompt = st.chat_input("Describe what happened…")
    if prompt:
        st.session_state.chat.append({"role": "user", "text": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(esc(prompt))
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("CLAIRE is reading your message…"):
                try:
                    cid = f"CHAT-{len([m for m in st.session_state.chat if m['role']=='user']):03d}"
                    result = analyse_and_respond(prompt.strip(), complaint_id=cid)
                    a = result["analysis"]
                    reply = result["automated_response"]
                    chip = (f'<span class="chip">Logged: {esc((a["cfpb_issues"][0] if a["cfpb_issues"] else "your complaint"))}</span>'
                            f'<span class="chip">Update within {a["sla_days"]} days</span>')
                    st.markdown(f'<div class="msg-sent">{esc(reply)}</div>', unsafe_allow_html=True)
                    st.markdown(chip, unsafe_allow_html=True)
                    with st.expander("Triage details (staff only)"):
                        st.markdown(
                            f"**Issue:** {(a['cfpb_issues'][0] if a['cfpb_issues'] else '—')}  \n"
                            f"**Product:** {a['cfpb_product'] or '—'}  \n"
                            f"**SLA / resolution target:** {a['sla_days']} days  \n"
                            f"**Urgency:** {a['urgency_level'] or '—'}  ·  "
                            f"**Sentiment:** {a['sentiment'] or '—'}  ·  "
                            f"**Confidence:** {a['cfpb_confidence'] or '—'}")
                    st.session_state.chat.append({"role": "assistant", "text": reply,
                                                  "chip": chip, "analysis": a})
                except Exception as exc:
                    err = "I'm sorry — something went wrong on our end while logging that. Please try again in a moment."
                    st.markdown(err)
                    st.caption(f"({exc})")
                    st.session_state.chat.append({"role": "assistant", "text": err})

    if len(st.session_state.chat) > 1:
        if st.button("Start a new conversation"):
            st.session_state.chat = [{"role": "assistant", "text": INTRO}]
            st.rerun()


# ════════════════════════════════════════════════════════════ TAB 2 — TRIAGE (staff)
with tab_analysis:
    st.caption("Internal analyst view — full classification, SLA, and the drafted consumer message.")
    in_col, out_col = st.columns([1, 1], gap="large")
    with in_col:
        narrative = st.text_area("Complaint narrative", height=220, label_visibility="collapsed",
            placeholder="Paste a complaint narrative to classify and draft a reply…")
        complaint_id = st.text_input("Complaint ID", value="MANUAL-001")
        go = st.button("⚡ Analyze & draft", type="primary", use_container_width=True)

    with out_col:
        if go:
            if not narrative.strip():
                st.warning("Enter a narrative first.")
                st.stop()
            with st.spinner("Analyzing…"):
                try:
                    result = analyse_and_respond(narrative.strip(), complaint_id=complaint_id.strip() or "MANUAL-001")
                except Exception as exc:
                    st.error(f"Error: {exc}"); st.stop()
            a = result["analysis"]

            # headline card — Issue / Timeline / Message (matches the consumer-reply spec)
            issue = a["cfpb_issues"][0] if a["cfpb_issues"] else "—"
            st.markdown(card("Issue identified", issue), unsafe_allow_html=True)
            st.markdown(card("Expected resolution timeline", f"{a['sla_days']} days"), unsafe_allow_html=True)
            st.markdown('<div class="claire-card"><div class="card-label">Message to consumer</div>'
                        f'<div class="msg-sent">{esc(result["automated_response"])}</div></div>',
                        unsafe_allow_html=True)

            with st.expander("Full triage analysis"):
                c1, c2 = st.columns(2)
                c1.markdown(card("Product", a["cfpb_product"] or "—"), unsafe_allow_html=True)
                c2.markdown(card("Urgency", badge(a["urgency_level"] or "low"), raw=True), unsafe_allow_html=True)
                c3, c4 = st.columns(2)
                c3.markdown(card("Sentiment", SENTIMENT.get(a["sentiment"], (a["sentiment"] or "neutral").title())), unsafe_allow_html=True)
                c4.markdown(card("Confidence", (a["cfpb_confidence"] or "—").title()), unsafe_allow_html=True)
                c5, c6 = st.columns(2)
                c5.markdown(card("Consumer goal", (a["consumer_goal"] or "—").replace("_", " ").title()), unsafe_allow_html=True)
                c6.markdown(card("Contact attempts", str(a["contact_attempts"])), unsafe_allow_html=True)
                st.markdown(card("Dollar amounts", ", ".join(a["dollar_amounts"]) if a["dollar_amounts"] else "None"), unsafe_allow_html=True)

            st.download_button("⬇ Download JSON", data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name=f"claire_{complaint_id or 'manual'}.json", mime="application/json")
        else:
            st.info("Enter a complaint on the left and click **Analyze & draft**.", icon="👈")


# ════════════════════════════════════════════════════════════ TAB 3 — EVALUATE
with tab_eval:
    st.subheader("📊 Batch Evaluation")
    st.caption("Upload a labelled CFPB CSV (needs a narrative column and an Issue column; a Product "
               "column enables product accuracy). Each row is a live model call; scoring uses the v4a near-equivalence groups.")
    uploaded = st.file_uploader("Upload labelled CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        st.info("Upload a CSV to begin — your data/processed/test.csv is the canonical set.", icon="📄")
    else:
        import pandas as pd
        try:
            raw_df = pd.read_csv(uploaded, dtype=str, low_memory=False)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}"); st.stop()
        norm_df = normalise_columns(raw_df)
        info = inspect_columns(norm_df)
        if not info["can_eval"]:
            missing = []
            if not info["narrative_col"]: missing.append("a narrative column")
            if not info["issue_col"]: missing.append("an 'Issue' column (ground truth)")
            st.error("Can't score this file — missing " + " and ".join(missing) + ".")
            st.dataframe(raw_df.head(5), use_container_width=True); st.stop()
        c1, c2, c3 = st.columns(3)
        c1.success(f"Narrative: `{info['narrative_col']}`")
        c2.success("Issue accuracy: available")
        if info["product_col"]:
            c3.success("Product accuracy: available")
        else:
            c3.warning("Product accuracy: no `product` column — issue-only")
        with st.expander("Preview (first 5 rows)"):
            st.dataframe(raw_df.head(5), use_container_width=True)
        n_rows = len(norm_df); n_cats = norm_df[info["issue_col"]].nunique()
        st.markdown(f"**{n_rows:,}** rows · **{n_cats}** issue categories")
        # Sampling strategy is fixed to a plain random sample (min-per-category = 0),
        # the canonical evaluation setting. The balanced-sample option remains
        # available via the CLI / run_evaluation() for methodology testing, but is
        # intentionally not exposed in this results-facing UI.
        min_per_cat = 0
        s2, s3 = st.columns(2)
        limit = s2.number_input("Row cap (total)", 5, 300, 50)
        throttle = s3.number_input("Per-call delay (sec)", 0.0, 30.0, 1.0, 0.5,
            help="Free tier (3 RPM) ≈ 20s; paid tier 0–1s.")
        est = min(int(limit), n_rows); per = float(throttle) + 1.5
        st.caption(f"≈ {est} rows × ~{per:.1f}s ≈ **{est*per/60:.1f} min** cold; cached rows are instant.")

        # Identify the current upload so stored results can be tied to it (and
        # cleared automatically when a different file is uploaded).
        file_key = f"{uploaded.name}:{getattr(uploaded, 'size', len(raw_df))}"

        # ── Run: compute results and STASH them in session_state. We deliberately
        #    do NOT render here. Rendering happens below, on every rerun, so a
        #    download-button click (which reruns the script) can't wipe the results.
        if st.button("▶ Run Evaluation", type="primary", use_container_width=True):
            SETTINGS["SLEEP_BETWEEN_CALLS"] = float(throttle)
            SETTINGS["SLEEP_RESPONSE_GEN"] = float(throttle)
            prog = st.progress(0.0, text="Starting…")
            def _cb(done, total, cid): prog.progress(min(done/total, 1.0), text=f"Classifying {done}/{total} · {cid}")
            try:
                out = run_evaluation(norm_df, limit=int(limit), min_per_category=int(min_per_cat), progress_callback=_cb)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}"); st.stop()
            finally:
                prog.empty()
            # Persist across reruns. Pre-serialise the CSVs so the download
            # buttons never need the live `out` object again.
            _rb = io.StringIO(); out["eval_results"].to_csv(_rb, index=False)
            _eb = io.StringIO(); out["error_rows"].to_csv(_eb, index=False)
            st.session_state["eval_result"] = {
                "file_key": file_key,
                "summary": out["summary"],
                "per_issue": out["per_issue"],
                "meta": out["meta"],
                "error_rows": out["error_rows"],
                "results_csv": _rb.getvalue(),
                "error_csv": _eb.getvalue(),
                "n_errors": len(out["error_rows"]),
            }

        # ── Render the results from session_state (survives download-button reruns).
        #    Only shown if they belong to the file currently uploaded; uploading a
        #    different file leaves them hidden until the next run.
        res = st.session_state.get("eval_result")
        if res and res["file_key"] == file_key:
            sm = res["summary"]
            st.markdown("#### Results")
            m1, m2, m3 = st.columns(3)
            if sm["scored_product"]:
                _pc = "pass" if sm["product_threshold_met"] else "fail"
                m1.markdown(card("Product accuracy",
                    f"{sm['product_accuracy']*100:.0f}%  <span class='badge badge-{_pc}'>{'PASS' if sm['product_threshold_met'] else 'BELOW'}</span>", raw=True), unsafe_allow_html=True)
            else:
                m1.markdown(card("Product accuracy", "n/a — no product column"), unsafe_allow_html=True)
            ip = "pass" if sm["issue_threshold_met"] else "fail"
            m2.markdown(card("Issue accuracy",
                f"{sm['issue_accuracy']*100:.0f}%  <span class='badge badge-{ip}'>{'PASS' if sm['issue_threshold_met'] else 'BELOW'}</span>", raw=True), unsafe_allow_html=True)
            m3.markdown(card("Rows evaluated", f"{sm['evaluated']} (failed: {sm['failed']})"), unsafe_allow_html=True)
            if res["meta"]["small_sample_warning"]:
                st.warning("Fewer than 40 rows evaluated — raise the cap / min-per-category for a reliable number.", icon="⚠️")
            import pandas as pd
            pi_df = pd.DataFrame([{"Issue": r["issue"], "Correct": r["correct"], "Total": r["total"],
                                   "Accuracy": f"{r['accuracy']*100:.0f}%"} for r in res["per_issue"]])
            st.markdown("##### Per-issue breakdown")
            st.dataframe(pi_df, use_container_width=True, hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇ eval_results.csv", res["results_csv"], "eval_results.csv", "text/csv",
                               use_container_width=True, key="dl_eval_results")
            if res["n_errors"] > 0:
                d2.download_button(f"⬇ error_log.csv ({res['n_errors']})", res["error_csv"], "error_log.csv",
                                   "text/csv", use_container_width=True, key="dl_error_log")
                with st.expander(f"Misclassified & near-equivalent rows ({res['n_errors']})"):
                    st.dataframe(res["error_rows"], use_container_width=True, hide_index=True)
            else:
                d2.success("No misclassifications.")