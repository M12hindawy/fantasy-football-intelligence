"""
markets/adapter.py
==================
Abstract base class for bookmaker market adapters.

Every bookmaker integration (1xBet, Bet365, etc.) must implement this
interface.  The system routes all odds through this adapter layer so
that the rest of the pipeline remains provider-agnostic.
"""

from __future__ import annotations

import abc
from typing import Any, Optional

from data.schemas import MarketBundle, MarketType, OddsRecord


class BookmakerMarketAdapter(abc.ABC):
    """
    Provider-agnostic interface for bookmaker market data.

    Concrete implementations must override all abstract methods.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable bookmaker name (e.g., '1xBet')."""

    @property
    @abc.abstractmethod
    def supported_markets(self) -> list[MarketType]:
        """List of MarketType values supported by this adapter."""

    # ── Odds ingestion ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def parse_odds(self, raw_payload: Any) -> list[OddsRecord]:
        """
        Parse raw odds data (API response, dict, screenshot export) into
        a list of OddsRecord objects.

        Implementations must populate ALL mandatory OddsRecord fields,
        including odds_timestamp and settlement_rule.

        Parameters
        ----------
        raw_payload : provider-specific data structure

        Returns
        -------
        list[OddsRecord]
        """

    @abc.abstractmethod
    def get_settlement_rule(self, market_type: MarketType, line: Optional[float]) -> str:
        """
        Return the bookmaker's settlement rule for a given market.

        Used to populate OddsRecord.settlement_rule.
        """

    # ── Market normalisation ──────────────────────────────────────────────────

    @abc.abstractmethod
    def normalize_market(
        self,
        bundle: MarketBundle,
        method: str = "shin",
    ) -> MarketBundle:
        """
        Remove the bookmaker margin from a MarketBundle.

        Must:
        - compute overround
        - apply the requested margin-removal method
        - record method_used in the returned bundle
        - NOT silently average incompatible methods

        Returns
        -------
        MarketBundle with overround and margin_method_used populated
        """

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_market_bundle(self, bundle: MarketBundle) -> list[str]:
        """
        Validate a MarketBundle and return a list of warning strings.

        Default checks (subclasses may extend):
        - All odds are > 1.0
        - Odds timestamp is present
        - Settlement rule is non-empty
        - At least one OddsRecord in the bundle
        """
        warnings: list[str] = []

        if not bundle.odds:
            warnings.append("Market bundle contains no odds records.")
            return warnings

        for rec in bundle.odds:
            if rec.decimal_odds <= 1.0:
                warnings.append(
                    f"Invalid odds for '{rec.selection}': {rec.decimal_odds} (must be > 1.0)"
                )
            if not rec.settlement_rule:
                warnings.append(
                    f"Missing settlement_rule for '{rec.selection}' in {rec.market_type}"
                )

        return warnings

    def is_market_supported(self, market_type: MarketType) -> bool:
        """Return True if this adapter supports the given market type."""
        return market_type in self.supported_markets

    # ── Utility ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider='{self.provider_name}')"
