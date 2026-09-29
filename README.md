# Urdu NER — FastAPI Client Demo

A polished client-facing Urdu Named Entity Recognition demo powered by:

- **FastAPI**
- **Qwen/Qwen3.5-9B** through Hugging Face Inference Providers
- **Structured JSON output**
- **Temperature = 0**
- **Thinking disabled**
- **Persistent sentence cache**
- **Professional dark Urdu-inspired UI**
- **Entity highlighting for PERSON, LOCATION, ORGANIZATION, DATE, TIME, NUMBER**

## Benchmark

Final benchmark configuration (v1.1.0):

- Model requested: `Qwen/Qwen3.5-9B:cheapest`
- Returned model: `Qwen/Qwen3.5-9B`
- Prompt version: `qwen-urdu-ner-6class-v1`
- Temperature: `0`
- Thinking: `False`
- Test sentences: `56`
- Gold entity spans: `98`

Results:

| Metric | Result |
|---|---:|
| Entity Precision | 91.1% |
| Entity Recall | 93.9% |
| **Exact Entity F1** | **92.5%** |
| LughaatNLP baseline F1 | 54.0% |
| Improvement | +38.5 pp |

Per-class F1:

| Entity | F1 |
|---|---:|
| PERSON | 80.0% |
| LOCATION | 95.9% |
| ORGANIZATION | 94.7% |
| DATE | 100.0% |
| TIME | 85.7% |
| NUMBER | 100.0% |

> The benchmark is a manually curated practical Urdu NER evaluation set, not a published research benchmark.

---

## What the demo does

Users can paste Urdu text and receive:

- highlighted named entities
- entity cards
- label counts
- model metadata
- cache status
- processing time

Supported entity labels:

- `PERSON`
- `LOCATION`
- `ORGANIZATION`
- `DATE`
- `TIME`
- `NUMBER`

The frontend includes ready-to-use example cards for fast client demonstrations.

---

## Architecture

1. **Browser Client**: Modern web UI with live Urdu word counter and interactive entity highlight badges.
2. **FastAPI Backend**: Handles routing, input length validation, and static asset serving.
3. **NER Service Layer**:
   - Performs sentence tokenization and regex splitting.
   - Computes SHA-256 fingerprint for persistent sentence caching.
   - Queries Hugging Face Inference Providers for uncached sentences.
   - Enforces strict exact-match entity extraction and character offset calculation.
4. **Model**: `Qwen/Qwen3.5-9B:cheapest` via Hugging Face Inference API (backend-only credentials).

---

## Project structure

```bash
urdu_ner_fastapi/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── ner_service.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
│
├── cache/
│   └── .gitkeep
│
├── data/
│   └── test_6class.jsonl
│
├── results/
│   ├── benchmark/
│   │   ├── v1.0.0/
│   │   │   ├── evaluation_report.txt
│   │   │   └── metrics.json
│   │   └── v1.1.0-six-class-derived/
│   │       ├── evaluation_report.txt
│   │       └── metrics.json
│   └── demo_runs/
│       └── .gitkeep
│
├── scripts/
│   ├── health_check.py
│   └── run_benchmark.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

---

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Configure Hugging Face

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set:

```ini
HF_TOKEN=hf_your_token_here
```

The token must be allowed to call Hugging Face Inference Providers.

Do not commit `.env`.

---

## 4. Run the application

```powershell
uvicorn app.main:app --reload --port 8005
```

Then open **http://127.0.0.1:8005** in your browser.

---

## 5. Run with Docker

Make sure your `.env` file is present and `HF_TOKEN` is configured, then run:

```powershell
docker compose up --build
```

Compose publishes port `8005` by default. Set `HOST_PORT` to use another host port, for example `HOST_PORT=8080 docker compose up --build` in a POSIX shell. The container keeps listening on port `8005`.

The container runs as an unprivileged user. Compose stores the sentence cache and demo run history in the `ner_cache` and `ner_results` Docker volumes so they survive container recreation. Existing files in the local `cache/` and `results/` folders are not copied into those volumes. The benchmark reports remain in the project folder.

Or using standard Docker CLI:

```powershell
docker build -t urdu-ner-fastapi .
docker run -p 8005:8005 --env-file .env -v ner_cache:/app/cache -v ner_results:/app/results urdu-ner-fastapi
```

Then open **http://127.0.0.1:8005** in your browser.

Check the container's health with `docker compose ps` or call `GET /api/health`. The health endpoint checks the web service; it does not make a paid model request. For deployment, provide `HF_TOKEN` through the platform's secret or environment settings. The image does not contain `.env`, cached predictions, or demo run history.

---

## API endpoints

### `GET /`

Client demo UI.

### `POST /api/ner`

Request:

```json
{
  "text": "ڈائریکٹر مریم علی نے 12 جون 2026 کو کراچی میں گوگل کے نمائندوں سے ملاقات کی۔"
}
```

Response:

```json
{
  "input": "ڈائریکٹر مریم علی نے 12 جون 2026 کو کراچی میں گوگل کے نمائندوں سے ملاقات کی۔",
  "entities": [
    {
      "text": "مریم علی",
      "label": "PERSON",
      "sentence_index": 0,
      "start": 9,
      "end": 17
    },
    {
      "text": "12 جون 2026",
      "label": "DATE",
      "sentence_index": 0,
      "start": 22,
      "end": 33
    },
    {
      "text": "کراچی",
      "label": "LOCATION",
      "sentence_index": 0,
      "start": 37,
      "end": 42
    },
    {
      "text": "گوگل",
      "label": "ORGANIZATION",
      "sentence_index": 0,
      "start": 47,
      "end": 51
    }
  ],
  "entity_count": 4,
  "sentence_count": 1,
  "cache_hits": 0,
  "api_calls": 1,
  "latency_ms": 420,
  "model": "Qwen/Qwen3.5-9B:cheapest",
  "version": "1.1.0"
}
```

### `GET /api/health`

Returns app/model configuration status. It does not call the model.

---

## API protection

This client demo intentionally restricts each submission.

Default limits:

- **Maximum Urdu words**: 80
- **Maximum sentences**: 4
- **Maximum API calls**: 1 per user request

Why 4 sentences?

The benchmark used batches of 4 sentences successfully. Keeping the demo to a single small batch minimizes Hugging Face usage.

Additional protection:

- exact-sentence persistent cache
- duplicate sentence deduplication
- one Hugging Face call maximum per request
- no retry for malformed responses
- no retry for HTTP 400/402/403/404
- no token exposed in browser code
- temperature fixed at 0
- thinking disabled
- structured JSON schema

If all submitted sentences are cached: `API calls = 0`.

---

## Cache

Predictions are stored on disk under `cache/qwen_ner_<fingerprint>.json`.

The fingerprint includes:

- model
- prompt version
- temperature
- thinking mode

Changing the prompt or model creates a different cache.
