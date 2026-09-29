from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3.5-9B:cheapest").strip()
TOKEN = os.getenv("HF_TOKEN", "").strip()

DATA_FILE = BASE_DIR / "data" / "test_6class.jsonl"
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results" / "benchmark" / "v1.1.0-six-class-rerun"

BATCH_SIZE = 4
MAX_API_CALLS = 20

ALLOWED_LABELS = [
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "DATE",
    "TIME",
    "NUMBER",
]

PROMPT_VERSION = "qwen-urdu-ner-6class-v1"

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


def load_dataset():
    rows = []
    with DATA_FILE.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            entities = [
                {"text": e[0].strip(), "label": e[1].strip().upper()}
                for e in item["entities"]
            ]
            rows.append({
                "id": int(item["id"]),
                "sentence": item["sentence"].strip(),
                "entities": entities,
            })

    if len(rows) != 56:
        raise ValueError(f"Expected 56 sentences, found {len(rows)}.")

    if sum(len(x["entities"]) for x in rows) != 98:
        raise ValueError("Expected 98 gold spans after excluding DESIGNATION.")

    for row in rows:
        for e in row["entities"]:
            if e["label"] not in ALLOWED_LABELS:
                raise ValueError(f"Unexpected label: {e['label']}")
            if e["text"] not in row["sentence"]:
                raise ValueError(
                    f"Gold entity {e['text']!r} not found in sentence {row['id']}."
                )

    return rows


def fingerprint():
    raw = f"{MODEL}|{PROMPT_VERSION}|{SYSTEM_PROMPT}|temperature=0|thinking=false"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def cache_file():
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"qwen_ner_6class_{fingerprint()}.json"


def load_cache():
    path = cache_file()
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_cache(cache):
    path = cache_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def schema(expected_count):
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


def clean_entities(items, sentence):
    seen = set()
    out = []
    for item in items:
        text = str(item.get("text", "")).strip()
        label = str(item.get("label", "")).strip().upper()
        if not text or label not in ALLOWED_LABELS:
            continue
        if text not in sentence:
            continue
        key = (text, label)
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "label": label})
    return out


def counter(items):
    return Counter((x["text"], x["label"]) for x in items)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f


def pct(x):
    return f"{x*100:.1f}%"


def main():
    if not TOKEN or TOKEN == "hf_your_token_here":
        raise RuntimeError("HF_TOKEN missing in .env")

    rows = load_dataset()
    print("Dataset validation: PASS")
    print("56 sentences, 98 gold spans, 6 classes")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=TOKEN,
        timeout=60.0,
        max_retries=0,
    )

    cache = load_cache()
    predictions = {}
    cache_hits = 0
    missing = []

    for row in rows:
        key = str(row["id"])
        cached = cache.get(key)
        if (
            isinstance(cached, dict)
            and cached.get("sentence") == row["sentence"]
            and isinstance(cached.get("entities"), list)
        ):
            predictions[row["id"]] = cached["entities"]
            cache_hits += 1
        else:
            missing.append(row)

    api_calls = 0
    returned_models = set()

    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start:start+BATCH_SIZE]

        if api_calls >= MAX_API_CALLS:
            raise RuntimeError("API safety cap reached.")

        prompt_cases = [{"id": x["id"], "sentence": x["sentence"]} for x in batch]
        user_prompt = (
            "Extract entities from these Urdu sentences.\n"
            "Return one result object for each input id.\n\n"
            + json.dumps(prompt_cases, ensure_ascii=False, indent=2)
        )

        print(f"API batch {start//BATCH_SIZE + 1}: {len(batch)} sentences")

        try:
            api_calls += 1
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=1000,
                response_format=schema(len(batch)),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except (APIStatusError, APIConnectionError, RateLimitError) as exc:
            print(f"Stopped after {api_calls} calls: {exc}")
            print("Completed batches are saved in cache. Re-run to continue.")
            raise

        if getattr(completion, "model", None):
            returned_models.add(str(completion.model))

        content = completion.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("Empty model response. No retry performed.")

        payload = json.loads(content)
        by_id = {int(x["id"]): x for x in payload["results"]}

        for row in batch:
            result = by_id.get(row["id"])
            if result is None:
                raise RuntimeError(f"Model omitted id {row['id']}")

            entities = clean_entities(result.get("entities", []), row["sentence"])
            predictions[row["id"]] = entities
            cache[str(row["id"])] = {
                "sentence": row["sentence"],
                "entities": entities,
            }

        save_cache(cache)
        time.sleep(1.0)

    tp = fp = fn = 0
    class_counts = {
        label: {"tp": 0, "fp": 0, "fn": 0}
        for label in ALLOWED_LABELS
    }
    sentence_rows = []
    error_rows = []

    for row in rows:
        gold = counter(row["entities"])
        pred = counter(predictions[row["id"]])

        hit = gold & pred
        false = pred - gold
        miss = gold - pred

        stp = sum(hit.values())
        sfp = sum(false.values())
        sfn = sum(miss.values())

        tp += stp
        fp += sfp
        fn += sfn

        for (text, label), n in hit.items():
            class_counts[label]["tp"] += n

        for (text, label), n in false.items():
            class_counts[label]["fp"] += n
            for _ in range(n):
                error_rows.append({
                    "sentence_id": row["id"],
                    "sentence": row["sentence"],
                    "error_type": "FALSE_POSITIVE",
                    "entity_text": text,
                    "label": label,
                })

        for (text, label), n in miss.items():
            class_counts[label]["fn"] += n
            for _ in range(n):
                error_rows.append({
                    "sentence_id": row["id"],
                    "sentence": row["sentence"],
                    "error_type": "MISSED_OR_WRONG",
                    "entity_text": text,
                    "label": label,
                })

        sp, sr, sf1 = prf(stp, sfp, sfn)
        sentence_rows.append({
            "sentence_id": row["id"],
            "sentence": row["sentence"],
            "gold_entities": json.dumps(row["entities"], ensure_ascii=False),
            "predicted_entities": json.dumps(predictions[row["id"]], ensure_ascii=False),
            "precision": round(sp, 6),
            "recall": round(sr, 6),
            "f1": round(sf1, 6),
        })

    p, r, f1 = prf(tp, fp, fn)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR/"sentence_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=sentence_rows[0].keys())
        writer.writeheader()
        writer.writerows(sentence_rows)

    with (RESULTS_DIR/"entity_errors.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fields = ["sentence_id", "sentence", "error_type", "entity_text", "label"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(error_rows)

    lines = [
        "="*72,
        "QWEN URDU NER — 6-CLASS EVALUATION REPORT",
        "="*72,
        f"Requested Model:           {MODEL}",
        "Returned Model(s):         " + (
            ", ".join(sorted(returned_models))
            if returned_models else "cache only / not exposed"
        ),
        f"Prompt Version:            {PROMPT_VERSION}",
        "Temperature:               0",
        "Thinking Enabled:          False",
        f"Batch Size:                {BATCH_SIZE}",
        "",
        "TEST SET",
        f"Test sentences:            {len(rows)}",
        f"Gold entity spans:         {sum(len(x['entities']) for x in rows)}",
        "Entity classes:            6",
        "Excluded class:            DESIGNATION",
        "",
        "EXACT ENTITY RESULTS  <-- PRIMARY",
        f"Entity precision:          {pct(p)}",
        f"Entity recall:             {pct(r)}",
        f"Entity F1:                 {pct(f1)}",
        f"Correct entity spans:      {tp}",
        f"False-positive spans:      {fp}",
        f"Missed/wrong spans:        {fn}",
        "",
        "PER-CLASS EXACT ENTITY RESULTS",
        f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-"*60,
    ]

    for label in ALLOWED_LABELS:
        c = class_counts[label]
        cp, cr, cf = prf(c["tp"], c["fp"], c["fn"])
        support = c["tp"] + c["fn"]
        lines.append(
            f"{label:<16} {pct(cp):>10} {pct(cr):>10} {pct(cf):>10} {support:>10}"
        )

    lines += [
        "",
        "API USAGE & CACHE",
        f"API calls this run:        {api_calls}",
        f"Cache hits:                {cache_hits}",
        f"Hard API-call cap:         {MAX_API_CALLS}",
        "",
        "HISTORICAL REFERENCE",
        "Previous 7-class Qwen Entity F1: 89.8%",
        "",
        "NOTE",
        "This is a 6-class benchmark. DESIGNATION is intentionally outside the",
        "defined project taxonomy and is excluded from both gold annotations and",
        "the model's allowed output labels.",
        "="*72,
    ]

    report = "\n".join(lines)
    (RESULTS_DIR/"evaluation_report.txt").write_text(report, encoding="utf-8")

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_requested": MODEL,
        "models_returned": sorted(returned_models),
        "prompt_version": PROMPT_VERSION,
        "temperature": 0,
        "thinking_enabled": False,
        "classes": ALLOWED_LABELS,
        "excluded_class": "DESIGNATION",
        "test_sentences": len(rows),
        "gold_entity_spans": sum(len(x["entities"]) for x in rows),
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "api_calls": api_calls,
        "cache_hits": cache_hits,
    }
    (RESULTS_DIR/"run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(report)


if __name__ == "__main__":
    main()
