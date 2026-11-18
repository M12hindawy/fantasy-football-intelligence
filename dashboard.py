"""
dashboard.py
============
FastAPI dashboard backend — يفتح صفحة ويب تعرض كل مباريات اليوم
مع تحليل كامل (نسب الفوز، التوقع، القرار) لكل مباراة.

تشغيل:
    uvicorn dashboard:app --reload --port 8000
ثم افتح المتصفح على:
    http://localhost:8000
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, date, timezone
from typing import Any, Optional
import logging

# UTF-8 fix for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from data.fetchers.football_data_fetcher import FootballDataFetcher
from data.schemas import MatchStatus
from models.dixon_coles import (
    score_probability_matrix,
    matrix_to_1x2,
    matrix_to_btts,
    matrix_to_over_under,
)
from data.schemas import PredictionOutput, ScoreMatrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

app = FastAPI(
    title="Football Intelligence Dashboard",
    description="تحليل مباريات اليوم - Football Value & Probability Analyzer",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Competition codes supported by Football-Data.org free tier ────────────────
COMPETITIONS = {
    "PL":  {"name": "Premier League",       "country": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England",  "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "PD":  {"name": "La Liga",              "country": "🇪🇸 Spain",    "flag": "🇪🇸"},
    "BL1": {"name": "Bundesliga",           "country": "🇩🇪 Germany",  "flag": "🇩🇪"},
    "SA":  {"name": "Serie A",              "country": "🇮🇹 Italy",    "flag": "🇮🇹"},
    "FL1": {"name": "Ligue 1",             "country": "🇫🇷 France",   "flag": "🇫🇷"},
    "CL":  {"name": "UEFA Champions League","country": "🇪🇺 UEFA",     "flag": "🏆"},
    "EL":  {"name": "UEFA Europa League",   "country": "🇪🇺 UEFA",     "flag": "🥈"},
    "PPL": {"name": "Primeira Liga",        "country": "🇵🇹 Portugal", "flag": "🇵🇹"},
    "DED": {"name": "Eredivisie",           "country": "🇳🇱 Holland",  "flag": "🇳🇱"},
    "BSA": {"name": "Brasileirão",          "country": "🇧🇷 Brazil",   "flag": "🇧🇷"},
    "WC":  {"name": "World Cup",            "country": "🌍 FIFA",      "flag": "🌍"},
    "EC":  {"name": "Euro Championship",    "country": "🇪🇺 UEFA",     "flag": "🇪🇺"},
}


def _baseline_prediction(fixture_id: str) -> PredictionOutput:
    """League-average baseline prediction when no fitted model is available."""
    mu_h, mu_a, rho = 1.35, 1.10, -0.10
    mat = score_probability_matrix(mu_h, mu_a, rho)
    p_home, p_draw, p_away = matrix_to_1x2(mat)
    p_btts = matrix_to_btts(mat)
    p_o25, _ = matrix_to_over_under(mat, 2.5)
    p_o15, _ = matrix_to_over_under(mat, 1.5)
    p_o35, _ = matrix_to_over_under(mat, 3.5)
    return PredictionOutput(
        fixture_id=fixture_id,
        model_name="BaselineDixonColes",
        model_version="baseline-league-avg",
        generated_at=datetime.now(timezone.utc),
        p_home_win=round(p_home, 4),
        p_draw=round(p_draw, 4),
        p_away_win=round(p_away, 4),
        expected_goals_home=mu_h,
        expected_goals_away=mu_a,
        p_btts=round(p_btts, 4),
        p_over_15=round(p_o15, 4),
        p_over_25=round(p_o25, 4),
        p_over_35=round(p_o35, 4),
        score_matrix=ScoreMatrix(matrix=mat.tolist()),
    )


def _decide(p_home: float, p_draw: float, p_away: float) -> dict:
    """Simple decision logic for dashboard display."""
    best_p = max(p_home, p_draw, p_away)
    if p_home == best_p:
        label, side = "Home Win", "home"
    elif p_draw == best_p:
        label, side = "Draw", "draw"
    else:
        label, side = "Away Win", "away"

    if best_p >= 0.60:
        confidence = "HIGH"
        color = "#10b981"   # green
    elif best_p >= 0.45:
        confidence = "MEDIUM"
        color = "#f59e0b"   # amber
    else:
        confidence = "LOW"
        color = "#6b7280"   # gray

    return {"label": label, "side": side, "probability": round(best_p * 100, 1),
            "confidence": confidence, "color": color}


def _format_match(fixture, pred: PredictionOutput, comp_info: dict) -> dict:
    """Format a fixture + prediction into a dashboard-ready dict."""
    decision = _decide(pred.p_home_win, pred.p_draw, pred.p_away_win)

    kickoff_str = fixture.kickoff.strftime("%H:%M") if fixture.kickoff else "--:--"
    kickoff_full = fixture.kickoff.strftime("%Y-%m-%d %H:%M UTC") if fixture.kickoff else ""

    return {
        "fixture_id": fixture.fixture_id,
        "home_team": fixture.home_team.name,
        "away_team": fixture.away_team.name,
        "competition": comp_info.get("name", fixture.competition),
        "competition_flag": comp_info.get("flag", "⚽"),
        "kickoff_time": kickoff_str,
        "kickoff_full": kickoff_full,
        "status": fixture.status.value,
        "home_score": fixture.home_score,
        "away_score": fixture.away_score,

        # Probabilities (%)
        "p_home_win": round(pred.p_home_win * 100, 1),
        "p_draw": round(pred.p_draw * 100, 1),
        "p_away_win": round(pred.p_away_win * 100, 1),

        # Expected goals
        "xg_home": pred.expected_goals_home,
        "xg_away": pred.expected_goals_away,

        # Derived markets
        "p_btts": round((pred.p_btts or 0) * 100, 1),
        "p_over_25": round((pred.p_over_25 or 0) * 100, 1),
        "p_over_15": round((pred.p_over_15 or 0) * 100, 1),
        "p_over_35": round((pred.p_over_35 or 0) * 100, 1),

        # Decision
        "decision": decision,
        "model_note": "⚠️ Baseline model (league average). Add historical data for calibrated output.",
    }


@app.get("/api/today")
async def get_today_matches():
    """
    Fetch all today's fixtures from Football-Data.org and run baseline predictions.
    Returns JSON list of match analyses.
    """
    today = date.today()
    fetcher = FootballDataFetcher()
    all_matches = []
    errors = []

    if not settings.football_data_api_key:
        # Return demo data when no API key is configured
        return JSONResponse({
            "date": str(today),
            "total": 3,
            "api_key_missing": True,
            "matches": _demo_matches(),
            "notice": (
                "⚠️ لم يتم ضبط Football-Data.org API key. "
                "البيانات المعروضة هي بيانات تجريبية فقط. "
                "أضف FOOTBALL_DATA_API_KEY في ملف .env للحصول على مباريات حقيقية."
            ),
        })

    # Fetch from all supported competitions
    for code, comp_info in COMPETITIONS.items():
        try:
            fixtures = fetcher.get_fixtures(
                competition_code=code,
                season=today.year if today.month >= 8 else today.year - 1,
                status="SCHEDULED,IN_PLAY,LIVE,PAUSED",
            )
            today_fixtures = [
                f for f in fixtures
                if f.kickoff and f.kickoff.date() == today
            ]
            for fixture in today_fixtures:
                pred = _baseline_prediction(fixture.fixture_id)
                match_data = _format_match(fixture, pred, comp_info)
                all_matches.append(match_data)
        except Exception as exc:
            errors.append(f"{code}: {str(exc)[:80]}")
            logger.warning("Error fetching %s: %s", code, exc)

    # Sort by kickoff time
    all_matches.sort(key=lambda m: m["kickoff_time"])

    return JSONResponse({
        "date": str(today),
        "total": len(all_matches),
        "api_key_missing": False,
        "matches": all_matches,
        "errors": errors,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notice": (
            "⚠️ هذا النظام يستخدم نموذج خط الأساس (متوسط الدوري). "
            "للحصول على تحليل أدق، شغّل python main.py backtest بعد تحميل بيانات تاريخية."
        ),
    })


@app.get("/api/odds-check")
async def odds_check(home: float, draw: float, away: float):
    """Quick overround + fair probability check."""
    from markets.market_math import overround, normalize_shin, raw_implied_probability
    odds = [home, draw, away]
    try:
        vig = overround(odds)
        fair = normalize_shin(odds)
        raw = [raw_implied_probability(o) for o in odds]
        return {
            "odds": {"home": home, "draw": draw, "away": away},
            "overround_pct": round(vig * 100, 2),
            "fair_probabilities": {
                "home": round(fair[0] * 100, 2),
                "draw": round(fair[1] * 100, 2),
                "away": round(fair[2] * 100, 2),
            },
            "method": "Shin",
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _demo_matches() -> list[dict]:
    """Demo matches shown when no API key is configured."""
    demos = [
        ("Manchester United", "Liverpool", "Premier League", "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "15:00",
         35.2, 27.8, 37.0, 1.28, 1.52, 56.3, 51.2),
        ("Real Madrid", "Barcelona", "La Liga", "🇪🇸", "21:00",
         48.1, 24.5, 27.4, 1.65, 1.20, 61.0, 58.7),
        ("Bayern Munich", "Dortmund", "Bundesliga", "🇩🇪", "18:30",
         55.3, 22.1, 22.6, 1.90, 1.15, 65.4, 62.1),
        ("PSG", "Marseille", "Ligue 1", "🇫🇷", "20:45",
         61.2, 21.0, 17.8, 2.10, 0.95, 68.0, 64.5),
        ("Juventus", "AC Milan", "Serie A", "🇮🇹", "20:45",
         40.5, 26.8, 32.7, 1.40, 1.30, 57.8, 52.3),
    ]
    results = []
    for home, away, comp, flag, ko, ph, pd_, pa, xgh, xga, pbtts, po25 in demos:
        decision = _decide(ph / 100, pd_ / 100, pa / 100)
        results.append({
            "fixture_id": f"demo_{home[:3]}_{away[:3]}",
            "home_team": home, "away_team": away,
            "competition": comp, "competition_flag": flag,
            "kickoff_time": ko, "kickoff_full": f"2026-08-31 {ko} UTC",
            "status": "pre_match",
            "home_score": None, "away_score": None,
            "p_home_win": ph, "p_draw": pd_, "p_away_win": pa,
            "xg_home": xgh, "xg_away": xga,
            "p_btts": pbtts, "p_over_25": po25,
            "p_over_15": 85.0, "p_over_35": 28.5,
            "decision": decision,
            "model_note": "⚠️ بيانات تجريبية — أضف API key للحصول على مباريات حقيقية",
        })
    return results


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML page."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
