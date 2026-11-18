"""
api/main.py
===========
FastAPI application entry point.
Run with: uvicorn api.main:app --reload
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config.settings import settings
from db.repository import init_db
from api.routes import predictions, fantasy

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up API — Initializing Database...")
    if settings.enable_db:
        init_db()
    yield
    logger.info("Shutting down API...")

app = FastAPI(
    title="Football Intelligence & Fantasy API",
    version="3.0",
    description="API for Match Predictions and FPL Optimization.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(fantasy.router)

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "3.0"}
