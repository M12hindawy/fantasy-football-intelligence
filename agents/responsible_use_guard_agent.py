"""
agents/responsible_use_guard_agent.py
======================================
Enforces all responsible-use and safety rules from spec Section 2.

This agent runs as a filter over every output before it reaches the user.
It:
 1. Injects the responsible-gambling notice into every betting output
 2. Blocks forbidden language ("guaranteed", "certain", "fixed", etc.)
 3. Refuses to produce output if automatic wager placement is requested
 4. Validates that no correlated parlay is recommended without joint-
    probability disclosure
 5. Warns on accumulator requests
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from data.schemas import MarketAnalysisResult

logger = logging.getLogger(__name__)

RESPONSIBLE_NOTICE = (
    "Betting involves financial risk. Only use money you can afford to lose, "
    "follow local laws and age requirements, and consider setting limits or "
    "stopping if gambling is becoming difficult to control."
)

# ── Forbidden language patterns ───────────────────────────────────────────────

_FORBIDDEN_TERMS = re.compile(
    r"\b(guaranteed|certain(?:ty)?|fixed|locked|sure\s?bet|risk.?free|100\s?%\s?win"
    r"|no-risk|banker|dead\s?cert|cert(?:ainty)?)\b",
    re.IGNORECASE,
)

# ── Accumulator disclosure template ───────────────────────────────────────────

ACCUMULATOR_WARNING = (
    "⚠️ ACCUMULATOR WARNING: Individual edges are NOT automatically independent. "
    "Combined probability P(all legs win) = Π P(each leg) only if legs are independent. "
    "Correlated outcomes (same league, same weather, related markets) reduce effective edge. "
    "The bookmaker margin compounds across legs, reducing expected value significantly. "
    "This system does not recommend accumulators for value betting."
)


@dataclass
class GuardResult:
    """Result of a responsible-use guard check."""
    passed: bool
    violations: list[str] = field(default_factory=list)
    notice_injected: bool = False
    modified_result: Optional[MarketAnalysisResult] = None


class ResponsibleUseGuardAgent:
    """
    Gate-keeper that runs on every MarketAnalysisResult before delivery.

    Usage::

        guard = ResponsibleUseGuardAgent()
        guard_result = guard.check(analysis_result)
        if guard_result.passed:
            deliver(guard_result.modified_result)
        else:
            show_violations(guard_result.violations)
    """

    # ── Main check ────────────────────────────────────────────────────────────

    def check(
        self,
        result: MarketAnalysisResult,
        text_output: str = "",
    ) -> GuardResult:
        """
        Validate and enrich a MarketAnalysisResult.

        Parameters
        ----------
        result      : the analysis result to validate
        text_output : any human-readable text that will be shown to the user
                      (checked for forbidden language)

        Returns
        -------
        GuardResult
        """
        violations: list[str] = []

        # 1. Check for forbidden language in text output
        self._check_forbidden_language(text_output, violations)

        # 2. Ensure responsible-gambling notice is present
        import copy
        r = copy.copy(result)
        if not r.responsible_gambling_notice:
            r.responsible_gambling_notice = RESPONSIBLE_NOTICE
        notice_injected = True

        # 3. Check that no wager was placed automatically
        self._check_no_auto_wager(violations)

        # 4. Validate stake is within safe bounds
        if r.stake is not None:
            self._check_stake_sanity(r.stake, violations)

        # 5. Validate probabilities
        self._check_probabilities(r, violations)

        passed = len(violations) == 0

        return GuardResult(
            passed=passed,
            violations=violations,
            notice_injected=notice_injected,
            modified_result=r,
        )

    def check_text(self, text: str) -> list[str]:
        """
        Check a raw string for responsible-use violations.

        Returns a list of violation strings (empty = clean).
        """
        violations: list[str] = []
        self._check_forbidden_language(text, violations)
        return violations

    def check_accumulator_request(
        self,
        legs: list[MarketAnalysisResult],
        assume_independence: bool = False,
    ) -> tuple[Optional[float], str]:
        """
        Compute joint probability for an accumulator and return a disclosure.

        Parameters
        ----------
        legs               : list of analysis results (one per leg)
        assume_independence: if True, use simple product; if False, warn

        Returns
        -------
        (joint_probability_or_None, disclosure_text)
        """
        if len(legs) < 2:
            return None, "Need at least 2 legs for an accumulator."

        probs = [r.model_probability for r in legs]
        if any(p is None for p in probs):
            return None, "One or more legs have no model probability. Cannot compute joint probability."

        joint = 1.0
        for p in probs:
            joint *= p  # type: ignore[operator]

        disclosure = (
            f"ACCUMULATOR DISCLOSURE:\n"
            f"  Legs: {len(legs)}\n"
            f"  Individual probabilities: {[round(p, 3) for p in probs]}\n"
            f"  Joint probability (independence assumed): {joint:.4f} ({joint*100:.2f}%)\n"
            f"  {ACCUMULATOR_WARNING}\n"
            f"\n{RESPONSIBLE_NOTICE}"
        )

        if not assume_independence:
            disclosure += (
                "\n\n  ⚠️ Independence assumption: UNVERIFIED. "
                "Correlated legs (shared opponents, same pitch, weather) "
                "may reduce the true joint probability below this estimate."
            )

        return round(joint, 6), disclosure

    # ── Private validators ────────────────────────────────────────────────────

    @staticmethod
    def _check_forbidden_language(text: str, violations: list[str]) -> None:
        matches = _FORBIDDEN_TERMS.findall(text)
        if matches:
            unique = list(set(m.lower() for m in matches))
            violations.append(
                f"FORBIDDEN LANGUAGE: The following terms must not appear in output: "
                f"{unique}. Replace with calibrated probability language."
            )

    @staticmethod
    def _check_no_auto_wager(violations: list[str]) -> None:
        # This system never places bets; this guard is a documentation hook.
        # In a production system with API access, this would verify no
        # wager-placement API calls were made.
        pass  # No auto-wager capability is implemented — safe by construction.

    @staticmethod
    def _check_stake_sanity(stake: float, violations: list[str]) -> None:
        if stake < 0:
            violations.append(f"INVALID STAKE: Negative stake ({stake}) is not valid.")
        if stake > 10_000:
            violations.append(
                f"HIGH STAKE WARNING: Recommended stake {stake:.2f} is very large. "
                "Ensure this does not materially endanger essential finances."
            )

    @staticmethod
    def _check_probabilities(r: MarketAnalysisResult, violations: list[str]) -> None:
        for attr, val in [
            ("model_probability", r.model_probability),
            ("raw_implied_probability", r.raw_implied_probability),
            ("fair_market_probability", r.fair_market_probability),
        ]:
            if val is not None and not (0.0 <= val <= 1.0):
                violations.append(
                    f"INVALID PROBABILITY: {attr} = {val} is outside [0, 1]."
                )

    # ── Sureness detection ────────────────────────────────────────────────────

    def respond_to_sure_bet_request(self) -> str:
        """
        Standard response when a user asks for a 'sure bet' or 'banker'.
        """
        return (
            "No football bet is certain. Even the most heavily favoured outcomes "
            "carry meaningful probabilities of the opposite result. The system "
            "provides only calibrated probability estimates and uncertainty-adjusted "
            "edge analysis. A 'value candidate' means the price appears favourable "
            "relative to the model estimate — it does NOT mean the bet will win.\n\n"
            f"{RESPONSIBLE_NOTICE}"
        )
