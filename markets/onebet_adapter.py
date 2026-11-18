"""
markets/onebet_adapter.py
=========================
1xBet concrete implementation of BookmakerMarketAdapter.

Handles all 13 market families defined in the master spec (Section 3).
Settlement rules are stored verbatim from 1xBet's published help pages.

Important: This adapter does NOT scrape or impersonate 1xBet.  Odds must
be supplied via:
  1. An authorized odds API (e.g., The Odds API free tier)
  2. User-supplied CSV/JSON export from the bookmaker
  3. Manually typed decimal odds via the CLI

This adapter is for ANALYTICAL purposes only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from data.schemas import (
    FreshnessStatusEnum,
    MarketBundle,
    MarketType,
    OddsRecord,
)
from markets.adapter import BookmakerMarketAdapter
from markets.market_math import (
    normalize_additive,
    normalize_power,
    normalize_proportional,
    normalize_shin,
    overround,
    raw_implied_probability,
)

PROVIDER = "1xBet"

# ── Settlement rules ──────────────────────────────────────────────────────────

_SETTLEMENT_RULES: dict[MarketType, str] = {
    MarketType.ONE_X_TWO: (
        "Full-time result (90 min + stoppage time). Extra time and penalties "
        "do not count. Void if match abandoned before 90 minutes."
    ),
    MarketType.DOUBLE_CHANCE: (
        "1X covers Home Win or Draw. X2 covers Draw or Away Win. "
        "12 covers Home Win or Away Win. 90 min result only."
    ),
    MarketType.DRAW_NO_BET: (
        "Stake returned if the match ends in a draw. Home win or Away win "
        "determines the result. 90 min only."
    ),
    MarketType.ASIAN_HANDICAP: (
        "Quarter-line bets: stake split equally over the two adjacent half "
        "lines. Push (void) on the whole-number line. Half-win/half-loss on "
        "the half-line. 90 min only."
    ),
    MarketType.EUROPEAN_HANDICAP: (
        "Three-way handicap with a goals head start applied before 1X2 "
        "settlement. Handles push scenarios differently from Asian handicap."
    ),
    MarketType.TOTALS: (
        "Total goals in 90 minutes (inc. stoppage time). Over/Under the "
        "stated line. Void on postponement or abandonment before 90 min."
    ),
    MarketType.BTTS: (
        "Both Teams To Score: YES if both teams score ≥1 goal in 90 min; "
        "NO if either or both teams score 0 goals."
    ),
    MarketType.TEAM_TOTALS: (
        "Individual team goal total. Over/Under the stated line for one team."
    ),
    MarketType.CORRECT_SCORE: (
        "Exact score at full time (90 min). Any other score = loss."
    ),
    MarketType.HT_1X2: (
        "Half-time result after 45 minutes (+ stoppage time). "
        "Second-half play does not affect settlement."
    ),
    MarketType.HT_TOTALS: (
        "Total goals in the first half only (45 min + stoppage time)."
    ),
    MarketType.HT_BTTS: (
        "Both teams to score in the first half (45 min + stoppage time)."
    ),
    MarketType.CORNERS: (
        "Total corners or match/team corner lines. "
        "Corners only — no penalty shootout corners."
    ),
    MarketType.CARDS: (
        "Total bookings. Red card counts as two yellow cards for booking-points "
        "markets unless otherwise stated."
    ),
    MarketType.PLAYER_PROP: (
        "Player-specific props (to score, shots, assists, cards). "
        "Player must start or enter the pitch; void if player does not play."
    ),
    MarketType.LIVE: (
        "In-play market.  Settlement depends on real-time game state.  "
        "Odds and lines can change second-by-second."
    ),
    MarketType.OTHER: "Refer to 1xBet official rules for this market type.",
}

# ── Markets supported in this adapter ────────────────────────────────────────

_SUPPORTED_MARKETS = [
    MarketType.ONE_X_TWO,
    MarketType.DOUBLE_CHANCE,
    MarketType.DRAW_NO_BET,
    MarketType.ASIAN_HANDICAP,
    MarketType.EUROPEAN_HANDICAP,
    MarketType.TOTALS,
    MarketType.BTTS,
    MarketType.TEAM_TOTALS,
    MarketType.CORRECT_SCORE,
    MarketType.HT_1X2,
    MarketType.HT_TOTALS,
    MarketType.HT_BTTS,
    MarketType.CORNERS,
    MarketType.CARDS,
    MarketType.PLAYER_PROP,
    MarketType.LIVE,
]


class OneBetAdapter(BookmakerMarketAdapter):
    """
    1xBet concrete market adapter.

    Accepts odds via:
    - `parse_odds(dict)` for structured API/export payloads
    - `from_manual()` for user-typed CLI input
    """

    @property
    def provider_name(self) -> str:
        return PROVIDER

    @property
    def supported_markets(self) -> list[MarketType]:
        return _SUPPORTED_MARKETS

    # ── Parsing ───────────────────────────────────────────────────────────────

    def parse_odds(self, raw_payload: Any) -> list[OddsRecord]:
        """
        Parse a structured odds payload into OddsRecord objects.

        Expected payload format (provider-agnostic dict):
        {
          "fixture_id": "...",
          "market_type": "1X2",
          "odds_timestamp": "2024-11-01T14:30:00Z",
          "event_start_time": "2024-11-02T15:00:00Z",
          "line": null,
          "selections": [
            {"name": "Home", "decimal_odds": 2.10},
            {"name": "Draw", "decimal_odds": 3.30},
            {"name": "Away", "decimal_odds": 3.50}
          ]
        }
        """
        if not isinstance(raw_payload, dict):
            raise ValueError("raw_payload must be a dict")

        mt_str = raw_payload.get("market_type", "OTHER").upper()
        # Normalise common aliases
        alias_map = {
            "1X2": "ONE_X_TWO",
            "OVER_UNDER": "TOTALS",
            "HANDICAP_ASIAN": "ASIAN_HANDICAP",
            "HANDICAP_EUROPEAN": "EUROPEAN_HANDICAP",
            "HALF_TIME": "HT_1X2",
        }
        mt_str = alias_map.get(mt_str, mt_str)
        try:
            mt = MarketType[mt_str]
        except KeyError:
            mt = MarketType.OTHER

        ts_str = raw_payload.get("odds_timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)

        event_start_str = raw_payload.get("event_start_time", "")
        try:
            event_start = datetime.fromisoformat(event_start_str.replace("Z", "+00:00"))
        except Exception:
            event_start = None

        line = raw_payload.get("line")
        records: list[OddsRecord] = []

        for sel in raw_payload.get("selections", []):
            name = sel.get("name", "")
            price = float(sel.get("decimal_odds", 0.0))
            if price <= 1.0:
                continue
            records.append(OddsRecord(
                bookmaker=PROVIDER,
                market_type=mt,
                selection=name,
                line=float(line) if line is not None else None,
                decimal_odds=price,
                odds_format="decimal",
                odds_timestamp=ts,
                event_start_time=event_start,
                source="user_supplied",
                market_status="open",
                settlement_rule=_SETTLEMENT_RULES.get(mt, ""),
            ))

        return records

    def get_settlement_rule(self, market_type: MarketType, line: Optional[float] = None) -> str:
        base = _SETTLEMENT_RULES.get(market_type, "No settlement rule available.")
        if line is not None:
            base += f"  Line: {line:+.2f}."
        return base

    # ── Normalisation ─────────────────────────────────────────────────────────

    def normalize_market(
        self,
        bundle: MarketBundle,
        method: str = "shin",
    ) -> MarketBundle:
        """
        Remove the bookmaker margin from a MarketBundle.

        Market-specific method selection:
        - 1X2, Double Chance, DNB → Shin (most appropriate for 3-outcome markets)
        - Totals, BTTS → proportional
        - Correct Score → additive (margin spread across many outcomes)
        - Asian Handicap → power (handles quarter-line splits)
        - All others → caller-specified or default Shin
        """
        odds_values = [r.decimal_odds for r in bundle.odds]
        if not odds_values:
            return bundle

        # Select method based on market type
        mt = bundle.market_type
        if method == "auto":
            if mt == MarketType.CORRECT_SCORE:
                method = "additive"
            elif mt in {MarketType.ASIAN_HANDICAP, MarketType.EUROPEAN_HANDICAP}:
                method = "power"
            elif mt in {MarketType.TOTALS, MarketType.BTTS, MarketType.TEAM_TOTALS}:
                method = "proportional"
            else:
                method = "shin"

        fn = {
            "shin": normalize_shin,
            "proportional": normalize_proportional,
            "additive": normalize_additive,
            "power": normalize_power,
        }.get(method, normalize_shin)

        fair_probs = fn(odds_values)
        vig = overround(odds_values)

        # Update the bundle with computed values
        bundle.overround = round(vig, 6)
        bundle.margin_method_used = method

        # Attach fair probability as a note to each odds record
        # (full fair-prob objects are handled by MarketProbabilityAgent)
        for rec, fp in zip(bundle.odds, fair_probs):
            # Store in a dynamically added field (not in schema, for agent use)
            rec.__dict__["_fair_probability"] = round(fp, 6)

        return bundle

    # ── Asian handicap quarter-line helper ────────────────────────────────────

    def resolve_asian_handicap_quarter_line(
        self,
        bundle: MarketBundle,
    ) -> dict[str, Any]:
        """
        For quarter-line Asian handicap markets, split each selection
        into two half-line components.

        Returns a dict describing the split for use by the value agent.
        """
        from markets.market_math import is_quarter_line, split_quarter_line

        line = bundle.line
        if line is None or not is_quarter_line(line):
            return {"is_quarter_line": False}

        lo, hi = split_quarter_line(line)
        return {
            "is_quarter_line": True,
            "original_line": line,
            "lower_line": lo,
            "upper_line": hi,
            "description": (
                f"AH {line:+.2f} = 50% stake on AH {lo:+.2f} + 50% on AH {hi:+.2f}"
            ),
        }

    # ── Manual entry factory ──────────────────────────────────────────────────

    @classmethod
    def from_manual(
        cls,
        fixture_id: str,
        market_type: MarketType,
        selections: list[tuple[str, float]],
        line: Optional[float] = None,
        odds_timestamp: Optional[datetime] = None,
        event_start_time: Optional[datetime] = None,
    ) -> MarketBundle:
        """
        Build a MarketBundle directly from user-supplied decimal odds.
        Used by the CLI when --manual-odds is specified.
        """
        from data.fetchers.odds_api_fetcher import OddsApiFetcher
        return OddsApiFetcher.from_manual_input(
            fixture_id=fixture_id,
            bookmaker=PROVIDER,
            market_type=market_type,
            selections=selections,
            line=line,
            odds_timestamp=odds_timestamp,
            event_start_time=event_start_time,
            settlement_rule=_SETTLEMENT_RULES.get(market_type, ""),
        )
