from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import (
    APP_VERSION,
    BASE_DIR,
    BENCHMARK,
    HF_MODEL,
    MAX_SENTENCES,
    MAX_URDU_WORDS,
)
from app.schemas import NERRequest, NERResponse
from app.services.ner_service import NERServiceError, UrduNERService

router = APIRouter()

# Use an absolute path so templates work no matter which folder
# uvicorn is started from.
templates = Jinja2Templates(
    directory=str(BASE_DIR / "app" / "templates")
)

service = UrduNERService()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Starlette's current TemplateResponse signature expects
    # request and name explicitly. Keyword arguments keep this
    # compatible and avoid the "unhashable type: dict" error.
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_version": APP_VERSION,
            "model": HF_MODEL,
            "benchmark": BENCHMARK,
            "max_words": MAX_URDU_WORDS,
            "max_sentences": MAX_SENTENCES,
        },
    )


@router.post("/api/ner", response_model=NERResponse)
async def run_ner(payload: NERRequest):
    try:
        return service.analyze(payload.text)
    except NERServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.public_message,
        ) from exc


@router.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": HF_MODEL,
        "max_words": MAX_URDU_WORDS,
        "max_sentences": MAX_SENTENCES,
        "benchmark_f1": BENCHMARK["f1"],
    }
