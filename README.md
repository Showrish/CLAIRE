# CLAIRE
**Consumer Language AI for Intelligent Reasoning and Extraction**

An LLM-powered consumer complaint triage agent built on the CFPB Consumer Complaint Database. CLAIRE classifies complaint narratives using the official CFPB taxonomy, assigns SLA-aligned resolution timelines, and generates empathetic consumer-facing acknowledgment messages — all in under 30 seconds per complaint.

> Developed as part of the Group 2 XN Project, ALY6080 Integrated Experiential Learning, Northeastern University (June 2026).

---

## What it does

1. **Ingests** a raw consumer complaint narrative
2. **Extracts** structured metadata (dollar amounts, dates, contact attempts, PII redaction flags) via regex — no API calls
3. **Classifies** the complaint using OpenAI `gpt-4o-mini-2024-07-18` with a few-shot system prompt tuned to the CFPB taxonomy
4. **Maps** the classified issue to a sponsor SLA benchmark (14 / 20 / 30 days)
5. **Generates** a warm, sentiment-calibrated consumer-facing acknowledgment message
6. **Evaluates** classification accuracy against labeled ground truth using near-equivalence scoring

---

## Results

| Metric | Value |
|--------|-------|
| Product accuracy (self-consistency v2) | 79% |
| Issue accuracy (self-consistency v2) | 72% |
| Final system accuracy (prompt v4a + RAG) | 80%+ |
| Response time | Under 30 seconds |
| Dataset | ~36,467 fintech sponsor complaints with narratives |

**Key finding:** The 80% accuracy ceiling on the public CFPB dataset is a data annotation problem, not a model problem. Fraud and Unauthorized transaction narratives share a 0.99 cross/within cosine similarity ratio — semantically indistinguishable at the text level.

---

## Project structure

```
CLAIRE/
├── app.py                    # Streamlit app (3 tabs: Assistant, Analysis, Evaluate)
├── src/
│   ├── respond.py            # Core pipeline: analyse_and_respond()
│   ├── complaint_parser.py   # ComplaintRecord, regex + LLM extractors, evaluation
│   ├── build_fewshots.py     # Few-shot example sampler
│   └── split.py              # Stratified train/test split
├── prompts/
│   └── system_prompt.py      # Classification prompt v4a (edit here to update behavior)
├── config/
│   ├── settings.py           # Central config: model, rate limits, paths, thresholds
│   └── sla_config.json       # SLA rules (editable without touching Python)
├── evaluation/
│   └── evaluate.py           # Batch evaluation logic for Streamlit Evaluate tab
├── data/
│   └── processed/
│       └── test.csv          # Canonical 20% stratified test split (~7,294 records)
└── requirements.txt
```

---

## Setup

**Prerequisites:** Python 3.10+, OpenAI API key

```bash
# 1. Navigate to project root
cd CLAIRE-main

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key (choose one)
# Option A: .env file
echo "OPENAI_API_KEY=sk-proj-..." > .env

# Option B: .streamlit/secrets.toml
mkdir -p .streamlit
echo 'OPENAI_API_KEY = "sk-proj-..."' > .streamlit/secrets.toml

# 5. Run the app
streamlit run app.py
```

Opens at **http://localhost:8501**

> If no API key file is present, the app shows a password field at startup.

---

## Usage

### Assistant tab (consumer-facing)
Type or paste a complaint narrative. CLAIRE responds with an acknowledgment message and classification chips showing the issue category and update timeline.

### Analysis tab (internal staff)
Paste a narrative, enter a complaint ID, click **Analyze & draft**. Returns full CFPB classification, SLA, urgency, sentiment, confidence, consumer goal, dollar amounts, and the drafted message. Downloadable as JSON.

### Evaluate tab (development / QA)
Upload a labeled CSV (requires narrative and Issue columns). Runs batch accuracy evaluation with near-equivalence scoring and per-issue breakdown. The canonical test set is `data/processed/test.csv`.

---

## CLI usage

```bash
# Single complaint
python src/respond.py --narrative "I noticed two unauthorized charges..." --id COMPLAINT-001

# JSON output
python src/respond.py --narrative "..." --json

# Interactive REPL
python src/respond.py

# Batch evaluation (random sample)
python src/complaint_parser.py --eval --input data/processed/test.csv --limit 100

# Batch evaluation (min 10 rows per issue category)
python src/complaint_parser.py --eval --input data/processed/test.csv --min-per-category 10
```

---

## Configuration

All settings in `config/settings.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `gpt-4o-mini-2024-07-18` | Pinned snapshot — prevents silent model drift |
| `TEMPERATURE` | `0` | Deterministic classification |
| `SLEEP_BETWEEN_CALLS` | `20s` | Free tier (3 RPM). Set to `0.5s` on paid tier |
| `NARRATIVE_MAX_CHARS` | `1500` | Narratives truncated before LLM call |
| `RESPONSE_TEMPERATURE` | `0.4` | Slight variation for natural responses |
| `MIN_ISSUE_ACCURACY` | `0.80` | Evaluation pass threshold |

SLA rules are in `config/sla_config.json` — editable without touching Python.

---

## Prompting strategy results

| Strategy | Product acc. | Issue acc. | Both correct |
|----------|-------------|------------|--------------|
| Zero-shot | 45% | 54% | 35% |
| Few-shot | 76% | 55% | 53% |
| Chain-of-thought | 76% | 53% | 45% |
| Self-consistency v2 | 79% | 72% | 58% |
| Final system (v4a RAG) | 80%+ | 80%+ | 80%+ |

---

## Known limitations

- **Label noise ceiling:** The 80% ceiling on public CFPB data is a dataset problem. Retraining on the sponsor's consistently labeled internal records is the path to higher accuracy.
- **Static FAISS index:** Must be manually rebuilt when new training data is added.
- **OpenAI dependency:** No offline mode. Rate limit errors retry with exponential backoff (up to 8 attempts).
- **English only:** Non-English narratives will classify with reduced accuracy.

---

## Team

Group 2 — ALY6080 Integrated Experiential Learning, Northeastern University

- Yashraj Gautam Tambe
- Showrish Reddy Sampath
- Rishi Ganapathi Senthil Kumar
- Kai-Hsiang Lin

---

## References

- Consumer Financial Protection Bureau. (2024). *Consumer complaint database* [Dataset]. https://www.consumerfinance.gov/data-research/consumer-complaints/
- Jullum et al. (2020). Detecting money laundering transactions with machine learning. *Journal of Money Laundering Control, 23*(1). https://doi.org/10.1108/JMLC-07-2019-0055
- Reimers & Gurevych (2019). Sentence-BERT. *EMNLP 2019*. https://doi.org/10.18653/v1/D19-1410
- Waskom (2021). Seaborn. *JOSS, 6*(60). https://doi.org/10.21105/joss.03021
