from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI, APIConnectionError, APIStatusError

from app.config import (
    ALLOWED_LABELS,
    APP_VERSION,
    CACHE_DIR,
    DEMO_RUNS_DIR,
    HF_MODEL,
    HF_TOKEN,
    MAX_SENTENCES,
    MAX_URDU_WORDS,
    PROMPT_VERSION,
    REQUEST_TIMEOUT_SECONDS,
    RESULTS_DIR,
)

SYSTEM_PROMPT = """You are an Urdu Named Entity Recognition engine.

Extract named entities from each Urdu sentence.

Allowed labels only:
PERSON
LOCATION
ORGANIZATION
DATE
TIME
NUMBER

Rules:
1. Copy entity text exactly from the input sentence.
2. Do not invent entities.
3. Do not change spelling.
4. Do not include words that are not entities.
5. Return every entity that belongs to one of the allowed labels.
6. Return results in the same sentence order.
7. Do not explain anything.
"""

# Urdu/Arabic-script words. Numbers are included as ordinary tokens for the demo
# word-limit calculation because NUMBER is an entity class.
TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[0-9۰-۹]+", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[۔؟!?])\s+|\n+")


class NERServiceError(RuntimeError):
    def __init__(self, message: str, public_message: str, status_code: int = 500):
        super().__init__(message)
        self.public_message = public_message
        self.status_code = status_code


def _fingerprint() -> str:
    raw = (
        f"{HF_MODEL}|{PROMPT_VERSION}|{SYSTEM_PROMPT}|"
        "temperature=0|thinking=false|schema=v1"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _cache_path() -> Path:
    return CACHE_DIR / f"qwen_ner_{_fingerprint()}.json"


def _load_cache() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    path = _cache_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"[ \t]+", " ", text.strip())
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(cleaned) if p.strip()]
    return parts or ([cleaned] if cleaned else [])


def count_urdu_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _response_schema(expected_count: int) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "urdu_ner_results",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "minItems": expected_count,
                        "maxItems": expected_count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "entities": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "label": {
                                                "type": "string",
                                                "enum": ALLOWED_LABELS,
                                            },
                                        },
                                        "required": ["text", "label"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["id", "entities"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
        },
    }


def _clean_entities(raw_entities: list[dict], sentence: str) -> list[dict]:
    seen = set()
    clean = []

    for entity in raw_entities:
        text = str(entity.get("text", "")).strip()
        label = str(entity.get("label", "")).strip().upper()

        if not text or label not in ALLOWED_LABELS:
            continue

        # The frozen prompt requires exact source text.
        if text not in sentence:
            continue

        key = (text, label)
        if key in seen:
            continue

        seen.add(key)
        clean.append({"text": text, "label": label})

    return clean


def _map_offsets(sentence: str, entities: list[dict]) -> list[dict]:
    """Map exact entity text back to source offsets without changing predictions."""
    occupied: list[tuple[int, int]] = []
    output = []

    for entity in entities:
        text = entity["text"]
        starts = [m.start() for m in re.finditer(re.escape(text), sentence)]
        chosen = None

        for start in starts:
            end = start + len(text)
            if not any(not (end <= a or start >= b) for a, b in occupied):
                chosen = (start, end)
                break

        item = dict(entity)
        if chosen:
            item["start"], item["end"] = chosen
            occupied.append(chosen)
        else:
            item["start"] = None
            item["end"] = None

        output.append(item)

    output.sort(
        key=lambda e: (
            e["start"] is None,
            e["start"] if e["start"] is not None else 10**9,
        )
    )
    return output


def _save_demo_run(payload: dict) -> None:
    now = datetime.now(timezone.utc)
    date_dir = DEMO_RUNS_DIR / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    stamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = date_dir / f"run_{stamp}.json"
    latest = RESULTS_DIR / "latest_run.json"

    data = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(data, encoding="utf-8")
    latest.write_text(data, encoding="utf-8")


class UrduNERService:
    def __init__(self):
        self.cache = _load_cache()
        self.client = None

        if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=HF_TOKEN,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )

    def _predict_uncached(self, indexed_sentences: list[tuple[int, str]]):
        if self.client is None:
            raise NERServiceError(
                "HF_TOKEN is missing.",
                "The model service is not configured. Add a Hugging Face token in .env.",
                503,
            )

        prompt_cases = [
            {"id": idx, "sentence": sentence}
            for idx, sentence in indexed_sentences
        ]

        user_prompt = (
            "Extract entities from these Urdu sentences.\n"
            "Return one result object for each input id.\n\n"
            + json.dumps(prompt_cases, ensure_ascii=False, indent=2)
        )

        try:
            completion = self.client.chat.completions.create(
                model=HF_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=1200,
                response_format=_response_schema(len(indexed_sentences)),
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
            )
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)

            if status == 402:
                raise NERServiceError(
                    f"Hugging Face HTTP 402: {getattr(exc, 'body', None)!r}",
                    "Hugging Face inference credits are currently unavailable. Please try again later.",
                    503,
                ) from exc

            if status == 429:
                raise NERServiceError(
                    f"Hugging Face HTTP 429: {getattr(exc, 'body', None)!r}",
                    "The model service is temporarily busy. Please try again shortly.",
                    503,
                ) from exc

            if status in (400, 403, 404):
                raise NERServiceError(
                    f"Hugging Face HTTP {status}: {getattr(exc, 'body', None)!r}",
                    "The model service could not process this request.",
                    502,
                ) from exc

            raise NERServiceError(
                f"Hugging Face HTTP {status}: {getattr(exc, 'body', None)!r}",
                "The model service returned an unexpected error.",
                502,
            ) from exc

        except APIConnectionError as exc:
            raise NERServiceError(
                f"Hugging Face connection error: {exc}",
                "Unable to reach the model service. Please try again.",
                503,
            ) from exc

        content = completion.choices[0].message.content or ""
        if not content.strip():
            raise NERServiceError(
                "Model returned empty content.",
                "The model returned an unexpected response.",
                502,
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise NERServiceError(
                f"Invalid JSON from model: {exc}",
                "The model returned an unexpected response.",
                502,
            ) from exc

        results = payload.get("results")
        if not isinstance(results, list):
            raise NERServiceError(
                "Structured output missing results list.",
                "The model returned an unexpected response.",
                502,
            )

        expected = {idx: sentence for idx, sentence in indexed_sentences}
        by_id = {}

        for result in results:
            rid = int(result["id"])
            if rid not in expected:
                continue
            by_id[rid] = _clean_entities(
                result.get("entities", []),
                expected[rid],
            )

        missing = set(expected) - set(by_id)
        if missing:
            raise NERServiceError(
                f"Model omitted sentence ids: {sorted(missing)}",
                "The model returned an incomplete response.",
                502,
            )

        returned_model = getattr(completion, "model", None)
        return by_id, str(returned_model) if returned_model else None

    def analyze(self, text: str) -> dict:
        started = time.perf_counter()

        text = text.strip()
        if not text:
            raise NERServiceError(
                "Empty input.",
                "Please enter an Urdu sentence.",
                400,
            )

        sentences = split_sentences(text)
        if len(sentences) > MAX_SENTENCES:
            raise NERServiceError(
                f"Too many sentences: {len(sentences)}",
                f"This demo supports up to {MAX_SENTENCES} sentences per request.",
                400,
            )

        word_count = count_urdu_tokens(text)
        if word_count > MAX_URDU_WORDS:
            raise NERServiceError(
                f"Too many Urdu tokens: {word_count}",
                f"This demo supports up to {MAX_URDU_WORDS} Urdu words per request.",
                400,
            )

        # Deduplicate exact repeated sentences while preserving original positions.
        sentence_to_positions = defaultdict(list)
        for idx, sentence in enumerate(sentences):
            sentence_to_positions[sentence].append(idx)

        unique_sentences = list(sentence_to_positions.keys())

        predicted_by_sentence = {}
        cache_hits = 0
        uncached = []

        for sentence in unique_sentences:
            cached = self.cache.get(sentence)
            if isinstance(cached, dict) and isinstance(cached.get("entities"), list):
                predicted_by_sentence[sentence] = cached["entities"]
                cache_hits += 1
            else:
                uncached.append(sentence)

        api_calls = 0
        returned_model = None

        if uncached:
            indexed = [(i, sentence) for i, sentence in enumerate(uncached)]
            by_id, returned_model = self._predict_uncached(indexed)
            api_calls = 1

            for i, sentence in indexed:
                entities = by_id[i]
                predicted_by_sentence[sentence] = entities
                self.cache[sentence] = {
                    "entities": entities,
                    "model": HF_MODEL,
                    "prompt_version": PROMPT_VERSION,
                }

            _save_cache(self.cache)

        all_entities = []

        for sentence_index, sentence in enumerate(sentences):
            entities = _map_offsets(
                sentence,
                predicted_by_sentence.get(sentence, []),
            )

            # Convert sentence-local offsets into full-input offsets.
            search_from = 0
            if sentence_index > 0:
                previous = sentences[sentence_index - 1]
                prev_pos = text.find(previous, search_from)
                search_from = prev_pos + len(previous) if prev_pos >= 0 else 0

            sentence_start = text.find(sentence, search_from)
            if sentence_start < 0:
                sentence_start = 0

            for entity in entities:
                item = dict(entity)
                item["sentence_index"] = sentence_index

                if item["start"] is not None:
                    item["start"] += sentence_start
                    item["end"] += sentence_start

                all_entities.append(item)

        latency_ms = round((time.perf_counter() - started) * 1000)

        run_payload = {
            "app_version": APP_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model_requested": HF_MODEL,
            "model_returned": returned_model,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input": text,
            "sentences": sentences,
            "entities": all_entities,
            "sentence_count": len(sentences),
            "entity_count": len(all_entities),
            "cache_hits": cache_hits,
            "api_calls": api_calls,
            "latency_ms": latency_ms,
            "success": True,
        }

        _save_demo_run(run_payload)

        return {
            "input": text,
            "entities": all_entities,
            "entity_count": len(all_entities),
            "sentence_count": len(sentences),
            "cache_hits": cache_hits,
            "api_calls": api_calls,
            "latency_ms": latency_ms,
            "model": HF_MODEL,
            "returned_model": returned_model,
            "version": APP_VERSION,
        }
