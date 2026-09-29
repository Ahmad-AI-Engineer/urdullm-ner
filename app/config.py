from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

APP_VERSION = os.getenv("APP_VERSION", "1.1.0")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "qwen-urdu-ner-6class-v1")
PORT = int(os.getenv("PORT", "8005"))

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3.5-9B:cheapest").strip()

MAX_URDU_WORDS = int(os.getenv("MAX_URDU_WORDS", "80"))
MAX_SENTENCES = int(os.getenv("MAX_SENTENCES", "4"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"
DEMO_RUNS_DIR = RESULTS_DIR / "demo_runs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEMO_RUNS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_LABELS = [
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "DATE",
    "TIME",
    "NUMBER",
]

BENCHMARK = {
    "precision": 91.1,
    "recall": 93.9,
    "f1": 92.5,
    "baseline_f1": 54.0,
    "sentences": 56,
    "gold_spans": 98,
    "scope": "6-class benchmark (DESIGNATION excluded)",
}
