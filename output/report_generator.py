"""
output/report_generator.py
===========================
Converts an AnalysisResult into human-readable text or structured JSON reports.
Supports three output formats: TEXT, JSON, MARKDOWN.

Designed to be the final stage before the dashboard/CLI/API renders output.
Never fabricates probabilities — only formats what the orchestrator computed.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional

from agents.orchestrator import AnalysisResult


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"

def _stars(strength: float, max_stars: int = 5) -> str:
    filled = max(0, min(max_stars, round(strength * max_stars)))
    return "★" * filled + "☆" * (max_stars - filled)


class ReportGenerator:
    """
    Formats AnalysisResult into human-readable outputs.

    Usage
    -----
        rg = ReportGenerator()
        print(rg.text(result))
        print(rg.markdown(result))
        data = rg.json(result)
    """

    # ── TEXT ──────────────────────────────────────────────────────────────

    def text(self, result: AnalysisResult, team_names: Optional[dict] = None) -> str:
        tn = team_names or {}
        home = tn.get(result.home_team_id, result.home_team_id)
        away = tn.get(result.away_team_id, result.away_team_id)
        lines = [
            "=" * 60,
            f"  MATCH ANALYSIS: {home} vs {away}",
            f"  Fixture ID : {result.fixture_id}",
            f"  Generated  : {result.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Model      : {result.model_name}",
            f"  Data Grade : {result.dq_grade}",
            "=" * 60,
            "",
            "  PROBABILITIES",
            f"    {home} Win : {_pct(result.p_home_win)}",
            f"    Draw        : {_pct(result.p_draw)}",
            f"    {away} Win : {_pct(result.p_away_win)}",
            "",
            "  GOALS",
            f"    xG {home} : {result.expected_home_goals:.2f}",
            f"    xG {away} : {result.expected_away_goals:.2f}",
            f"    BTTS        : {_pct(result.p_btts)}",
            f"    Over 2.5    : {_pct(result.p_over_25)}",
            "",
        ]

        # Top scorelines from simulation
        if result.simulation and result.simulation.get("top_scorelines"):
            lines.append("  TOP SCORELINES")
            for sc in result.simulation["top_scorelines"][:5]:
                lines.append(f"    {sc['label']:6s}  {_pct(sc['probability'])}")
            lines.append("")

        # Team analyses
        if result.home_analysis:
            ha = result.home_analysis
            lines += [
                f"  {home.upper()} CONTEXT",
                f"    Form trend   : {ha.get('form_trend','?')}",
                f"    Fatigue risk : {ha.get('fatigue_risk','?')}",
                f"    Strengths    : {', '.join(ha.get('strengths', [])) or 'None identified'}",
                f"    Weaknesses   : {', '.join(ha.get('weaknesses', [])) or 'None identified'}",
                "",
            ]
        if result.away_analysis:
            aa = result.away_analysis
            lines += [
                f"  {away.upper()} CONTEXT",
                f"    Form trend   : {aa.get('form_trend','?')}",
                f"    Fatigue risk : {aa.get('fatigue_risk','?')}",
                f"    Strengths    : {', '.join(aa.get('strengths', [])) or 'None identified'}",
                f"    Weaknesses   : {', '.join(aa.get('weaknesses', [])) or 'None identified'}",
                "",
            ]

        # Lineup alerts
        if result.home_missing_key_players:
            lines.append(f"  ⚠️  {home} missing key players: {', '.join(result.home_missing_key_players)}")
        if result.away_missing_key_players:
            lines.append(f"  ⚠️  {away} missing key players: {', '.join(result.away_missing_key_players)}")
        if result.home_missing_key_players or result.away_missing_key_players:
            lines.append("")

        # What-if
        if result.what_if_applied and result.what_if_adjustments:
            wi = result.what_if_adjustments
            orig = wi["original"]
            adj  = wi["adjusted"]
            lines += [
                "  WHAT-IF SCENARIOS APPLIED",
                f"    xG adjusted: {orig['mu_home']:.2f}→{adj['mu_home']:.2f} (H)  "
                f"{orig['mu_away']:.2f}→{adj['mu_away']:.2f} (A)",
                "",
            ]

        # Recommendation
        lines += [
            "  RECOMMENDATION",
            f"    Signal     : {result.recommendation}",
            f"    Strength   : {_stars(result.recommendation_strength)}",
            "",
        ]

        # Warnings
        if result.warnings:
            lines.append("  WARNINGS")
            for w in result.warnings[:5]:
                lines.append(f"    ⚠  {w}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("  ⚠  ANALYTICAL TOOL ONLY — NOT A GUARANTEE OF OUTCOME.")
        lines.append("     NEVER bet more than you can afford to lose.")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ── MARKDOWN ──────────────────────────────────────────────────────────

    def markdown(self, result: AnalysisResult, team_names: Optional[dict] = None) -> str:
        tn = team_names or {}
        home = tn.get(result.home_team_id, result.home_team_id)
        away = tn.get(result.away_team_id, result.away_team_id)
        md = [
            f"## ⚽ {home} vs {away}",
            f"> *Generated: {result.generated_at.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Model: `{result.model_name}` | DQ: **{result.dq_grade}***",
            "",
            "### Probabilities",
            "| Outcome | Probability |",
            "|---------|------------|",
            f"| 🏠 {home} Win | **{_pct(result.p_home_win)}** |",
            f"| 🤝 Draw       | **{_pct(result.p_draw)}** |",
            f"| ✈️ {away} Win  | **{_pct(result.p_away_win)}** |",
            "",
            "### Expected Goals",
            f"- **{home}** xG: `{result.expected_home_goals:.2f}`",
            f"- **{away}** xG: `{result.expected_away_goals:.2f}`",
            f"- BTTS: `{_pct(result.p_btts)}` | Over 2.5: `{_pct(result.p_over_25)}`",
            "",
        ]

        if result.simulation and result.simulation.get("top_scorelines"):
            md.append("### Top Scorelines (Monte Carlo)")
            md.append("| Score | Probability |")
            md.append("|-------|------------|")
            for sc in result.simulation["top_scorelines"][:5]:
                md.append(f"| `{sc['label']}` | {_pct(sc['probability'])} |")
            md.append("")

        md += [
            "### Recommendation",
            f"**{result.recommendation}** {_stars(result.recommendation_strength)}",
            "",
            "> ⚠️ *Analytical tool only — not a guarantee of outcome.*",
        ]
        return "\n".join(md)

    # ── JSON ──────────────────────────────────────────────────────────────

    def json(self, result: AnalysisResult, indent: int = 2) -> str:
        return json.dumps(result.as_dict(), indent=indent, default=str)


report_generator = ReportGenerator()
