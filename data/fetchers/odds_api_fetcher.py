"""
data/fetchers/odds_api_fetcher.py
=================================
Fetcher for The Odds API (https://api.the-odds-api.com/v4).

Free tier provides ~500 requests/month and covers the major leagues.
The fetcher also accepts manual decimal odds supplied by the user via
the CLI (--manual-odds flag) to avoid burning API quota.

All odds are stored as OddsRecord objects with mandatory:
  bookmaker, market_type, selection, line, decimal_odds, odds_timestamp,
  source, market_status, settlement_rule, freshness_status.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import requests

from config.freshness import freshness_checker
from config.settings import settings
from data.cache import FootballCache
from data.schemas import (
    FreshnessStatusEnum,
    MarketBundle,
    MarketType,
    OddsRecord,
    SourceRecord,
)

BASE_URL = "https://api.the-odds-api.com/v4"
PROVIDER = "the-odds-api"

# Map The Odds API market keys to internal MarketType
_MARKET_MAP: dict[str, MarketType] = {
    "h2h": MarketType.ONE_X_TWO,
    "totals": MarketType.TOTALS,
    "spreads": MarketType.ASIAN_HANDICAP,
    "btts": MarketType.BTTS,
    "double_chance": MarketType.DOUBLE_CHANCE,
    "draw_no_bet": MarketType.DRAW_NO_BET,
}

# Settlement rules per market type
_SETTLEMENT_RULES: dict[MarketType, str] = {
    MarketType.ONE_X_TWO: "Full-time result. 90 minutes + injury time only. Void on postponement.",
    MarketType.TOTALS: "Total goals in 90 minutes. Over/under the stated line. Void on postponement.",
    MarketType.ASIAN_HANDICAP: "Quarter-line split: half stake on each adjacent half-line.",
    MarketType.BTTS: "Both teams must score at least one goal in 90 minutes.",
    MarketType.DOUBLE_CHANCE: "Covers two of the three 1X2 outcomes.",
    MarketType.DRAW_NO_BET: "Stake refunded if match ends in a draw.",
}


class OddsApiFetcher:
    """
    Fetches pre-match odds from The Odds API with caching.

    Parameters
    ----------
    api_key : The Odds API key (reads from settings if empty)
    cache   : FootballCache instance
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[FootballCache] = None,
    ) -> None:
        self._api_key = api_key or settings.odds_api_key
        self._cache = cache or FootballCache(
            db_path=settings.cache_db_path,
            enabled=settings.cache_enabled,
        )
        self._session = requests.Session()

    # ── Internal HTTP ─────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{BASE_URL}{path}"
        cache_key = f"odds_api:{url}:{str(params)}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.payload

        full_params = {"apiKey": self._api_key, **(params or {})}
        try:
            resp = self._session.get(url, params=full_params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            raise RuntimeError(f"Odds API request timed out: {url}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                raise RuntimeError(
                    "Odds API: Invalid or missing API key.  Set ODDS_API_KEY in .env"
                )
            raise

        self._cache.set(cache_key, "prematch_odds", data)
        return data

    # ── Source helper ─────────────────────────────────────────────────────────

    def _make_source(self) -> SourceRecord:
        now = datetime.now(timezone.utc)
        status, _ = freshness_checker.check("prematch_odds", now)
        return SourceRecord(
            provider=PROVIDER,
            source_url=BASE_URL,
            retrieved_at=now,
            freshness_status=FreshnessStatusEnum(status.value),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_sports(self) -> list[dict]:
        """Return list of available sports (for discovery)."""
        return self._get("/sports")

    def get_odds(
        self,
        sport_key: str,
        regions: str = "eu",
        markets: str = "h2h,totals",
        bookmakers: str = "1xbet",
        event_id: Optional[str] = None,
    ) -> list[MarketBundle]:
        """
        Fetch odds for upcoming events in a sport.

        Parameters
        ----------
        sport_key   : e.g. "soccer_epl", "soccer_spain_la_liga"
        regions     : comma-separated regions (eu, uk, us, au)
        markets     : comma-separated market keys (h2h, totals, spreads)
        bookmakers  : comma-separated bookmaker IDs
        event_id    : optional; narrow to a specific event

        Returns
        -------
        list[MarketBundle] — one bundle per market per event
        """
        params: dict[str, Any] = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "bookmakers": bookmakers,
        }
        if event_id:
            data = self._get(f"/sports/{sport_key}/events/{event_id}/odds", params)
            events = [data] if isinstance(data, dict) else data
        else:
            data = self._get(f"/sports/{sport_key}/odds", params)
            events = data if isinstance(data, list) else []

        now = datetime.now(timezone.utc)
        bundles: list[MarketBundle] = []

        for event in events:
            event_id_str = str(event.get("id", ""))
            event_start = event.get("commence_time", "")
            try:
                kickoff = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
            except Exception:
                kickoff = now

            for bm in event.get("bookmakers", []):
                bm_name = bm.get("title", bookmakers)
                for market in bm.get("markets", []):
                    mkey = market.get("key", "")
                    mt = _MARKET_MAP.get(mkey, MarketType.OTHER)
                    mtime_str = market.get("last_update", "")
                    try:
                        mtime = datetime.fromisoformat(mtime_str.replace("Z", "+00:00"))
                    except Exception:
                        mtime = now

                    freshness_status, age = freshness_checker.check("prematch_odds", mtime)

                    odds_records: list[OddsRecord] = []
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = float(outcome.get("price", 0.0))
                        line = outcome.get("point")

                        if price <= 1.0:
                            continue  # Skip invalid odds

                        odds_records.append(OddsRecord(
                            bookmaker=bm_name,
                            market_id=f"{event_id_str}_{mkey}",
                            market_type=mt,
                            selection=name,
                            line=float(line) if line is not None else None,
                            decimal_odds=price,
                            odds_format="decimal",
                            odds_timestamp=mtime,
                            event_start_time=kickoff,
                            source=PROVIDER,
                            market_status="open",
                            settlement_rule=_SETTLEMENT_RULES.get(mt, ""),
                            freshness_status=FreshnessStatusEnum(freshness_status.value),
                        ))

                    if odds_records:
                        bundles.append(MarketBundle(
                            fixture_id=event_id_str,
                            market_type=mt,
                            bookmaker=bm_name,
                            odds=odds_records,
                            bundle_timestamp=mtime,
                        ))

        return bundles

    # ── Manual odds entry ─────────────────────────────────────────────────────

    @staticmethod
    def from_manual_input(
        fixture_id: str,
        bookmaker: str,
        market_type: MarketType,
        selections: list[tuple[str, float]],
        line: Optional[float] = None,
        odds_timestamp: Optional[datetime] = None,
        event_start_time: Optional[datetime] = None,
        settlement_rule: str = "",
    ) -> MarketBundle:
        """
        Build a MarketBundle from manually entered odds (e.g., from CLI or screenshot).

        Parameters
        ----------
        selections : list of (selection_name, decimal_odds) tuples
                     e.g. [("Home", 2.10), ("Draw", 3.30), ("Away", 3.50)]
        """
        now = datetime.now(timezone.utc)
        ts = odds_timestamp or now
        freshness_status, _ = freshness_checker.check("prematch_odds", ts)

        records = []
        for sel, price in selections:
            if price <= 1.0:
                raise ValueError(f"Invalid decimal odds {price} for '{sel}' — must be > 1.0")
            records.append(OddsRecord(
                bookmaker=bookmaker,
                market_type=market_type,
                selection=sel,
                line=line,
                decimal_odds=price,
                odds_format="decimal",
                odds_timestamp=ts,
                event_start_time=event_start_time,
                source="manual_entry",
                market_status="open",
                settlement_rule=settlement_rule or _SETTLEMENT_RULES.get(market_type, ""),
                freshness_status=FreshnessStatusEnum(freshness_status.value),
            ))

        return MarketBundle(
            fixture_id=fixture_id,
            market_type=market_type,
            bookmaker=bookmaker,
            odds=records,
            bundle_timestamp=ts,
            line=line,
        )
