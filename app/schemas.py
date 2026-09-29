from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityLabel = Literal[
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "DATE",
    "TIME",
    "NUMBER",
]


class NERRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class EntityOut(BaseModel):
    text: str
    label: EntityLabel
    sentence_index: int
    start: int | None = None
    end: int | None = None


class NERResponse(BaseModel):
    input: str
    entities: list[EntityOut]
    entity_count: int
    sentence_count: int
    cache_hits: int
    api_calls: int
    latency_ms: int
    model: str
    returned_model: str | None = None
    version: str
