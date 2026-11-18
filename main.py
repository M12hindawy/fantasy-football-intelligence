"""
main.py
=======
CLI entry point for the Football Value & Probability Analyzer.

Commands
--------
  python main.py analyze   — Analyse a specific market
  python main.py backtest  — Run walk-forward backtest on historical data
  python main.py odds-check — Quick overround/fair-prob check from manual odds

Usage examples
--------------
  # Analyse Over 2.5 goals with manually typed 1xBet odds
  python main.py analyze \\
      --competition PL \\
      --home "Arsenal" \\
      --away "Chelsea" \\
      --kickoff "2025-12-15T15:00:00Z" \\
      --market TOTALS \\
      --line 2.5 \\
      --selection "Over 2.5" \\
      --odds 1.85 \\
      --odds-time "2025-12-14T20:00:00Z"

  # Quick overround check
  python main.py odds-check --odds 2.10 3.30 3.50

  # Analyse with stake sizing
  python main.py analyze ... \\
      --bankroll 1000 --currency USD --risk-profile moderate \\
      --jurisdiction "United States"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

# Ensure UTF-8 output on Windows consoles (prevents emoji encoding errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config.settings import settings
from data.schemas import (
    BankrollContext,
    MatchStatus,
    MarketType,
    TeamRecord,
)
from data.fetchers.odds_api_fetcher import OddsApiFetcher
from markets.onebet_adapter import OneBetAdapter
from agents.odds_ingestion_agent import OddsIngestionAgent
from agents.margin_removal_agent import MarginRemovalAgent
from agents.market_normalization_agent import MarketNormalizationAgent
from agents.market_probability_agent import MarketProbabilityAgent
from agents.value_assessment_agent import ValueAssessmentAgent
from agents.risk_and_stake_agent import RiskAndStakeAgent
from agents.responsible_use_guard_agent import ResponsibleUseGuardAgent
from output.formatter import format_report, format_no_data_report
from output.json_contract import AnalysisOutputJSON
from markets.market_math import overround, normalize_shin, raw_implied_probability

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("main")


# ── Shared pipeline ───────────────────────────────────────────────────────────

def run_analyze_pipeline(args: argparse.Namespace) -> int:
    """
    Full analysis pipeline for a single market selection.

    Returns exit code (0 = success, 1 = error).
    """
    guard = ResponsibleUseGuardAgent()

    # ── Handle 'sure bet' requests ────────────────────────────────────────────
    if getattr(args, "sure_bet", False):
        print(guard.respond_to_sure_bet_request())
        return 0

    # ── Validate required fields ──────────────────────────────────────────────
    if not args.home or not args.away:
        print("ERROR: --home and --away team names are required.", file=sys.stderr)
        return 1

    if not args.odds:
        print(
            "ERROR: --odds is required. The system cannot claim value without current odds.\n"
            "You can enter decimal odds manually, e.g.: --odds 1.85",
            file=sys.stderr
        )
        return 1

    # ── Parse kickoff ─────────────────────────────────────────────────────────
    try:
        kickoff = datetime.fromisoformat(args.kickoff.replace("Z", "+00:00"))
    except Exception:
        kickoff = datetime.now(timezone.utc)
        logger.warning("Could not parse kickoff time '%s'; using now.", args.kickoff)

    # ── Parse odds timestamp ──────────────────────────────────────────────────
    try:
        odds_ts = datetime.fromisoformat(args.odds_time.replace("Z", "+00:00"))
    except Exception:
        odds_ts = datetime.now(timezone.utc)
        logger.warning("Could not parse --odds-time; using current time.")

    # ── Determine market type ─────────────────────────────────────────────────
    market_str = (args.market or "1X2").upper().replace("-", "_").replace(" ", "_")
    alias = {
        "1X2": "ONE_X_TWO",
        "OVER_UNDER": "TOTALS",
        "HANDICAP": "ASIAN_HANDICAP",
        "AH": "ASIAN_HANDICAP",
        "EH": "EUROPEAN_HANDICAP",
    }
    market_str = alias.get(market_str, market_str)
    try:
        market_type = MarketType[market_str]
    except KeyError:
        print(f"ERROR: Unknown market type '{args.market}'.", file=sys.stderr)
        print(f"Valid options: {[m.name for m in MarketType]}", file=sys.stderr)
        return 1

    # ── Parse selections and odds ─────────────────────────────────────────────
    # For 1X2, the user may supply three odds values
    # For other markets, one selection + one odds value
    selection = args.selection or "Home"

    if args.all_odds:
        # Multi-odds: e.g. --all-odds "Home:2.10,Draw:3.30,Away:3.50"
        try:
            parts = [p.strip() for p in args.all_odds.split(",")]
            selections_list = []
            for part in parts:
                name, price = part.split(":")
                selections_list.append((name.strip(), float(price.strip())))
        except Exception as exc:
            print(f"ERROR: Could not parse --all-odds '{args.all_odds}': {exc}", file=sys.stderr)
            return 1
    else:
        selections_list = [(selection, float(args.odds))]

    # ── Build MarketBundle ────────────────────────────────────────────────────
    fixture_id = f"{args.home.replace(' ', '_')}_{args.away.replace(' ', '_')}"

    try:
        bundle = OddsApiFetcher.from_manual_input(
            fixture_id=fixture_id,
            bookmaker="1xBet",
            market_type=market_type,
            selections=selections_list,
            line=float(args.line) if args.line else None,
            odds_timestamp=odds_ts,
            event_start_time=kickoff,
        )
    except Exception as exc:
        print(f"ERROR: Building odds bundle failed: {exc}", file=sys.stderr)
        return 1

    # ── Odds ingestion validation ─────────────────────────────────────────────
    ingestion_agent = OddsIngestionAgent()
    bundle, ingestion_issues = ingestion_agent.validate(bundle, kickoff)
    if ingestion_issues:
        print("\n[ODDS VALIDATION ISSUES]")
        for issue in ingestion_issues:
            print(f"  • {issue}")
        print()

    # ── Margin removal ────────────────────────────────────────────────────────
    margin_agent = MarginRemovalAgent()
    bundle, method_used, margin_warnings = margin_agent.apply(bundle, method="auto")
    if margin_warnings:
        for w in margin_warnings:
            print(f"  ⚠️  {w}")

    # ── Normalisation validation ───────────────────────────────────────────────
    norm_agent = MarketNormalizationAgent()
    bundle, norm_warnings = norm_agent.validate(bundle)
    if norm_warnings:
        for w in norm_warnings:
            logger.info("Normalisation: %s", w)

    # ── Get fair probability for the target selection ─────────────────────────
    # Find the target odds record
    target_record = next(
        (r for r in bundle.odds if r.selection.lower() == selection.lower()),
        bundle.odds[0] if bundle.odds else None,
    )
    if target_record is None:
        print("ERROR: Selection not found in bundle.", file=sys.stderr)
        return 1

    # ── Model prediction ──────────────────────────────────────────────────────
    # Build a simple prediction using goal-averages if team stats are available
    # For MVP: use naive league-average Poisson when no historical data loaded
    try:
        from models.dixon_coles import DixonColesModel, score_probability_matrix, matrix_to_1x2, matrix_to_btts, matrix_to_over_under
        import numpy as np

        # Placeholder: without fitted model, use league average xG
        # In production: load fitted model from disk
        print("\n[MODEL] No fitted model loaded — using league-average baseline (xG: 1.35 / 1.10).")
        print("        Run 'python main.py backtest' after loading historical data for calibrated output.\n")

        mu_h = 1.35   # league-average home xG
        mu_a = 1.10   # league-average away xG
        rho = -0.10

        mat = score_probability_matrix(mu_h, mu_a, rho)
        p_home, p_draw, p_away = matrix_to_1x2(mat)
        p_btts = matrix_to_btts(mat)
        p_o25, _ = matrix_to_over_under(mat, 2.5)

        from data.schemas import PredictionOutput, ScoreMatrix
        prediction = PredictionOutput(
            fixture_id=fixture_id,
            model_name="BaselineDixonColes",
            model_version="baseline-league-avg",
            generated_at=datetime.now(timezone.utc),
            p_home_win=round(p_home, 6),
            p_draw=round(p_draw, 6),
            p_away_win=round(p_away, 6),
            expected_goals_home=mu_h,
            expected_goals_away=mu_a,
            p_btts=round(p_btts, 6),
            p_over_25=round(p_o25, 6),
            score_matrix=ScoreMatrix(matrix=mat.tolist()),
        )

    except Exception as exc:
        print(f"ERROR: Prediction engine failed: {exc}", file=sys.stderr)
        return 1

    # ── Market probability ────────────────────────────────────────────────────
    mp_agent = MarketProbabilityAgent()
    model_prob, mp_notes = mp_agent.get_model_probability(prediction, bundle, selection)
    if mp_notes:
        for note in mp_notes:
            print(f"  ℹ️  {note}")

    # ── Value assessment ──────────────────────────────────────────────────────
    va_agent = ValueAssessmentAgent()
    result = va_agent.assess(
        home_team=args.home,
        away_team=args.away,
        competition=args.competition or "Unknown",
        kickoff=kickoff,
        status=MatchStatus.PRE_MATCH,
        bundle=bundle,
        odds_record=target_record,
        prediction=prediction,
        model_probability=model_prob,
        key_reasons=[],
        main_risks=[],
    )

    # ── Stake sizing (optional) ───────────────────────────────────────────────
    if args.bankroll and args.currency and args.risk_profile and args.jurisdiction:
        try:
            bankroll_ctx = BankrollContext(
                bankroll=float(args.bankroll),
                currency=args.currency.upper(),
                risk_profile=args.risk_profile.lower(),
                legal_jurisdiction=args.jurisdiction,
            )
            stake_agent = RiskAndStakeAgent()
            stake_rec = stake_agent.recommend(result, bankroll_ctx)
            result = stake_agent.apply_to_result(result, stake_rec)
            print(f"\n[STAKE] {stake_rec.reasoning}")
        except Exception as exc:
            print(f"⚠️  Stake calculation failed: {exc}")

    # ── Responsible use guard ─────────────────────────────────────────────────
    guard_result = guard.check(result)
    if not guard_result.passed:
        print("\n[GUARD VIOLATIONS]")
        for v in guard_result.violations:
            print(f"  ❌ {v}")
    result = guard_result.modified_result or result

    # ── Output ────────────────────────────────────────────────────────────────
    human_report = format_report(result)
    print(human_report)

    if args.json:
        output_json = AnalysisOutputJSON.from_result(result)
        print("\n[JSON OUTPUT]")
        print(output_json.serialize())

    return 0


def run_odds_check(args: argparse.Namespace) -> int:
    """Quick overround and fair-probability check from CLI odds."""
    if not args.odds:
        print("ERROR: Provide odds with --odds 2.10 3.30 3.50", file=sys.stderr)
        return 1
    odds_list = [float(o) for o in args.odds]
    vig = overround(odds_list)
    fair = normalize_shin(odds_list)
    print(f"\nODDS        : {odds_list}")
    print(f"OVERROUND   : {vig * 100:.2f}%")
    print(f"FAIR PROBS  : {[f'{p * 100:.2f}%' for p in fair]} (Shin method)")
    raw = [raw_implied_probability(o) for o in odds_list]
    print(f"RAW PROBS   : {[f'{p * 100:.2f}%' for p in raw]}")
    return 0


# ── CLI parser ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-agent",
        description="Football Value & Probability Analyzer for 1xBet Markets",
    )
    sub = parser.add_subparsers(dest="command")

    # ── analyze ───────────────────────────────────────────────────────────────
    p_analyze = sub.add_parser("analyze", help="Analyse a single market selection")
    p_analyze.add_argument("--competition", default="", help="Competition code, e.g. PL")
    p_analyze.add_argument("--home", required=True, help="Home team name")
    p_analyze.add_argument("--away", required=True, help="Away team name")
    p_analyze.add_argument("--kickoff", default="", help="Kickoff ISO-8601 timestamp")
    p_analyze.add_argument("--market", default="1X2", help="Market type, e.g. 1X2, TOTALS")
    p_analyze.add_argument("--line", default=None, help="Market line, e.g. 2.5")
    p_analyze.add_argument("--selection", default="Home", help="Selection name, e.g. 'Over 2.5'")
    p_analyze.add_argument("--odds", default=None, help="Decimal odds for the selection")
    p_analyze.add_argument("--all-odds", default=None, help="All odds as 'Home:2.10,Draw:3.30,Away:3.50'")
    p_analyze.add_argument("--odds-time", default="", help="Odds timestamp ISO-8601")
    p_analyze.add_argument("--bankroll", default=None, help="Bankroll amount (activates stake sizing)")
    p_analyze.add_argument("--currency", default=None, help="ISO-4217 currency code, e.g. USD")
    p_analyze.add_argument("--risk-profile", default=None, help="conservative|moderate|aggressive")
    p_analyze.add_argument("--jurisdiction", default=None, help="Legal jurisdiction")
    p_analyze.add_argument("--json", action="store_true", help="Also output JSON")
    p_analyze.add_argument("--sure-bet", action="store_true", help="Request a sure-bet response")

    # ── odds-check ────────────────────────────────────────────────────────────
    p_odds = sub.add_parser("odds-check", help="Quick overround check from decimal odds")
    p_odds.add_argument("--odds", nargs="+", help="Decimal odds e.g. 2.10 3.30 3.50")

    # ── backtest ──────────────────────────────────────────────────────────────
    p_bt = sub.add_parser("backtest", help="Run walk-forward backtest (requires match data file)")
    p_bt.add_argument("--data", required=True, help="Path to JSON match records file")
    p_bt.add_argument("--train-seasons", type=int, default=2)
    p_bt.add_argument("--json", action="store_true", help="Output results as JSON")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "analyze":
        return run_analyze_pipeline(args)
    elif args.command == "odds-check":
        return run_odds_check(args)
    elif args.command == "backtest":
        return run_backtest(args)
    else:
        parser.print_help()
        return 1


def run_backtest(args: argparse.Namespace) -> int:
    """Load match records from JSON file and run walk-forward backtest."""
    import json
    from pathlib import Path
    path = Path(args.data)
    if not path.exists():
        print(f"ERROR: Data file not found: {path}", file=sys.stderr)
        return 1
    with open(path) as f:
        records = json.load(f)
    # Parse dates
    for r in records:
        if isinstance(r.get("date"), str):
            r["date"] = datetime.fromisoformat(r["date"].replace("Z", "+00:00"))
    from backtesting.walk_forward import WalkForwardBacktester
    bt = WalkForwardBacktester(train_seasons=args.train_seasons)
    result = bt.run(records)
    print(f"\nWalk-Forward Backtest Results ({result.n_folds} folds)")
    print(f"  Mean Brier Score : {result.mean_brier}")
    print(f"  Mean Log Loss    : {result.mean_log_loss}")
    print(f"  Mean ECE         : {result.mean_ece}")
    print(f"  Mean ROI         : {result.mean_roi}")
    print(f"  Mean CLV         : {result.mean_clv}")
    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), default=str, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
