# Reshaped x Rénan
### Technical assignment Software & AI Engineer: Turn the Safety Management Manual into action — at the point of need

This is Rénan's solution to the technical assignment for the role of Software & AI Engineer at Reshaped.

## Running the application

### Prerequisites

- Python 3.10 or newer (developed on 3.14)
- At least one of the two models:
  - a Gemini API key — [ai.google.dev](https://ai.google.dev/gemini-api/docs/api-key) (primary)
  - [Ollama](https://ollama.com) running locally (offline fallback)

### 1. Install the dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set the Gemini API key

```bash
export GEMINI_API_KEY="your-key-here"
```

Set this before launching. If the variable is missing the app still starts — Gemini is
skipped entirely and every call goes straight to the local model instead, which then stops
being optional (see step 3).

### 3. Pull the local fallback model

```bash
ollama pull qwen3:8b
```

Optional when `GEMINI_API_KEY` is set: the local model only takes over if a Gemini call
fails. Required if you run without a key, since it is then the only model available.

### 4. Start the app

```bash
streamlit run app.py
```

Streamlit prints a local URL (by default <http://localhost:8501>). Describe a situation in
the chat input — for example *"we lost steering"* or *"someone almost fell overboard"* — and
the assistant answers from the manual.

All commands are run from the project root.

## Rebuilding the manual index

`data/processed/sections.json` is committed, so this step is only needed after changing the
source PDF or the ingestion logic:

```bash
python -m src.ingest
```

This re-reads `data/raw/Case Manual marineops_sms.pdf` and overwrites the JSON in place.

## Running the evaluation scripts

Each script prints its results to the terminal and makes live LLM calls, so one of the two
models has to be reachable:

```bash
python -m eval.test_select   # scored retrieval cases, plus which resolution path was taken
python -m eval.test_router   # retrieval, prompt-injection defence, hallucination consistency
python -m eval.debug_finger  # one hard case: minor injury vs. the first aid section
```

