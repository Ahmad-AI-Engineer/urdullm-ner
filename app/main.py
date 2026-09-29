from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import APP_VERSION, BASE_DIR, PORT
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print()
    print(f"Urdu NER Demo v{APP_VERSION}")
    print(f"http://127.0.0.1:{PORT}")
    print()
    yield


app = FastAPI(
    title="Urdu NER",
    description="Client-facing Urdu Named Entity Recognition demo.",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Absolute path keeps static files working even if uvicorn
# is launched from a parent folder.
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "app" / "static")),
    name="static",
)

app.include_router(router)
