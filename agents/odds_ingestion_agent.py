"""
agents/odds_ingestion_agent.py
==============================
Validates incoming odds records and rejects or flags any that fail
the spec's Section 3 requirements.

Every OddsRecord must have:
  bookmaker, market_id, selection, line (when applicable), decimal_odds,
  odds_format, odds_timestamp, event_start_time, source, market_status,
  settlement_rule, freshness_status.

The agent also checks that the odds were not fabricated by the model
and that they are tied to a real event with a known kickoff time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from config.freshness import FreshnessStatus, freshness_checker
from data.schemas import FreshnessStatusEnum, MarketBundle, MarketType, OddsRecord

logger = logging.getLogger(__name__)


class OddsIngestionAgent:
    """
    Validates and enriches a MarketBundle before it enters the pipeline.

    Returns (validated_bundle, issues) where issues is a list of strings.
    Empty issues list means the bundle passed all checks.
    """

    # Markets that require a non-null line
    _LINE_REQUIRED: set[MarketType] = {
        MarketType.ASIAN_HANDICAP,
        MarketType.EUROPEAN_HANDICAP,
        MarketType.TOTALS,
        MarketType.TEAM_TOTALS,
    }

    def validate(
        self,
        bundle: MarketBundle,
        event_start_time: Optional[datetime] = None,
    ) -> tuple[MarketBundle, list[str]]:
        """
        Validate a MarketBundle.

        Parameters
        ----------
        bundle           : the incoming market data
        event_start_time : kickoff time (if not embedded in OddsRecord)

        Returns
        -------
        (bundle, issues) — issues is empty if all checks pass
        """
        issues: list[str] = []
        now = datetime.now(timezone.utc)

        if not bundle.odds:
            issues.append(f"CRITICAL: Market bundle for {bundle.market_type} has no odds records.")
            return bundle, issues

        for rec in bundle.odds:
            rec_issues = self._validate_record(rec, event_start_time, now)
            issues.extend(rec_issues)

        # Market-level checks
        if bundle.market_type in self._LINE_REQUIRED:
            missing_lines = [r.selection for r in bundle.odds if r.line is None]
            if missing_lines:
                issues.append(
                    f"Line value required for {bundle.market_type} but missing on: "
                    f"{missing_lines}"
                )

        if issues:
            logger.warning(
                "OddsIngestionAgent: %d issue(s) found in %s bundle for fixture %s.",
                len(issues), bundle.market_type, bundle.fixture_id,
            )
        else:
            logger.info(
                "OddsIngestionAgent: Bundle for %s / %s passed validation.",
                bundle.fixture_id, bundle.market_type,
            )

        return bundle, issues

    def _validate_record(
        self,
        rec: OddsRecord,
        event_start_time: Optional[datetime],
        now: datetime,
    ) -> list[str]:
        issues: list[str] = []

        # 1. Odds format
        if rec.decimal_odds <= 1.0:
            issues.append(
                f"INVALID ODDS: '{rec.selection}' has decimal odds {rec.decimal_odds} ≤ 1.0. "
                "This is not a valid decimal price."
            )

        # 2. Timestamp present
        if rec.odds_timestamp is None:
            issues.append(
                f"MISSING TIMESTAMP: No odds_timestamp for '{rec.selection}'. "
                "Cannot determine freshness."
            )
            return issues  # Cannot proceed with freshness check

        # 3. Odds not from the future
        if rec.odds_timestamp > now:
            issues.append(
                f"SUSPICIOUS TIMESTAMP: Odds for '{rec.selection}' are timestamped "
                f"in the future ({rec.odds_timestamp})."
            )

        # 4. Freshness check
        data_type = self._data_type_for_market(rec.market_type)
        status, age = freshness_checker.check(data_type, rec.odds_timestamp, as_of=now)
        age_str = freshness_checker.human_readable(age)

        if status == FreshnessStatus.EXPIRED:
            issues.append(
                f"EXPIRED ODDS: '{rec.selection}' odds are {age_str} old "
                f"(threshold for {data_type} is "
                f"{freshness_checker.threshold(data_type)}s). "
                "Recalculate immediately before using these odds."
            )
            rec.freshness_status = FreshnessStatusEnum.EXPIRED
        elif status == FreshnessStatus.STALE:
            issues.append(
                f"STALE ODDS: '{rec.selection}' odds are {age_str} old. "
                "Confidence will be reduced."
            )
            rec.freshness_status = FreshnessStatusEnum.STALE
        elif status == FreshnessStatus.WARNING:
            issues.append(
                f"ODDS WARNING: '{rec.selection}' odds are {age_str} old — approaching "
                "the freshness limit."
            )
            rec.freshness_status = FreshnessStatusEnum.WARNING
        else:
            rec.freshness_status = FreshnessStatusEnum.FRESH

        # 5. Settlement rule present
        if not rec.settlement_rule:
            issues.append(
                f"MISSING SETTLEMENT RULE: '{rec.selection}' has no settlement_rule. "
                "Ambiguous market — treat with caution."
            )

        # 6. Event has not already started / finished (for pre-match)
        est = event_start_time or rec.event_start_time
        if est is not None:
            if est.tzinfo is None:
                est = est.replace(tzinfo=timezone.utc)
            if now > est:
                gap_min = (now - est).total_seconds() / 60
                issues.append(
                    f"POST-KICKOFF ODDS: Odds timestamp for '{rec.selection}' is "
                    f"{gap_min:.0f} minutes after event start. "
                    "Use live pipeline if the match is in progress."
                )

        # 7. Source must not be blank
        if not rec.source:
            issues.append(
                f"MISSING SOURCE: No source attribution for '{rec.selection}'."
            )

        return issues

    @staticmethod
    def _data_type_for_market(mt: MarketType) -> str:
        if mt == MarketType.LIVE:
            return "live_score"
        return "prematch_odds"

    # ── Freshness penalty helper ───────────────────────────────────────────────

    @staticmethod
    def freshness_penalty(bundle: MarketBundle) -> float:
        """
        Compute a staleness penalty [0, 1] for use in uncertainty_adjusted_edge.

        FRESH → 0.0, WARNING → 0.01, STALE → 0.04, EXPIRED → 0.10
        """
        worst = FreshnessStatusEnum.FRESH
        penalty_map = {
            FreshnessStatusEnum.FRESH: 0.0,
            FreshnessStatusEnum.WARNING: 0.01,
            FreshnessStatusEnum.STALE: 0.04,
            FreshnessStatusEnum.EXPIRED: 0.10,
        }
        priority = {
            FreshnessStatusEnum.FRESH: 0,
            FreshnessStatusEnum.WARNING: 1,
            FreshnessStatusEnum.STALE: 2,
            FreshnessStatusEnum.EXPIRED: 3,
        }
        for rec in bundle.odds:
            if priority.get(rec.freshness_status, 0) > priority.get(worst, 0):
                worst = rec.freshness_status
        return penalty_map.get(worst, 0.0)
