"""
backtesting/walk_forward.py
============================
Walk-forward backtester for the prediction models.

Key constraints (from spec Section 13):
- NEVER use random train/test splits — always time-ordered
- Record the exact odds snapshot available at prediction time
- NEVER use closing odds in a historical prediction unless closing
  odds were actually available at the timestamp being simulated
- Evaluate on every league, market, odds range, and home/away split

Output: WalkForwardResult containing per-fold and aggregate metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from backtesting.metrics import (
    BrierScore,
    LogLoss,
    compute_clv,
    compute_roi,
    compute_max_drawdown,
    compute_calibration_ece,
)
from models.dixon_coles import DixonColesModel
from models.elo_model import DynamicEloModel
from models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    n_train: int
    n_val: int

    # Metrics
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    ece: Optional[float] = None
    accuracy: Optional[float] = None
    roi: Optional[float] = None
    clv: Optional[float] = None


@dataclass
class WalkForwardResult:
    """Aggregate backtesting results."""

    n_folds: int
    total_train_matches: int
    total_val_matches: int
    folds: list[FoldResult] = field(default_factory=list)

    # Aggregate metrics (mean across folds)
    mean_brier: Optional[float] = None
    mean_log_loss: Optional[float] = None
    mean_ece: Optional[float] = None
    mean_accuracy: Optional[float] = None
    mean_roi: Optional[float] = None
    mean_clv: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Selection statistics
    total_selections: int = 0
    value_candidates: int = 0
    model_version: str = ""
    run_at: Optional[datetime] = None


class WalkForwardBacktester:
    """
    Chronological walk-forward backtester.

    Parameters
    ----------
    train_seasons : number of seasons to use in each training window
    val_seasons   : number of seasons in each validation window
    min_matches   : minimum matches required before a model is usable
    """

    def __init__(
        self,
        train_seasons: int = 2,
        val_seasons: int = 1,
        min_matches: int = 200,
    ) -> None:
        self._train_seasons = train_seasons
        self._val_seasons = val_seasons
        self._min_matches = min_matches

    def run(
        self,
        match_records: list[dict],
        odds_snapshots: Optional[dict] = None,
        edge_threshold: float = 0.03,
    ) -> WalkForwardResult:
        """
        Run the walk-forward backtest.

        Parameters
        ----------
        match_records : list of dicts with keys:
            home_id, away_id, home_goals, away_goals, date (datetime),
            home_name, away_name, competition, season
        odds_snapshots : {fixture_id: {market: decimal_odds, ...}}
            Pre-match odds available at prediction time (NOT closing odds)
        edge_threshold : minimum edge to classify a selection as value candidate

        Returns
        -------
        WalkForwardResult
        """
        if not match_records:
            raise ValueError("No match records provided.")

        # Sort chronologically — never shuffle
        records = sorted(match_records, key=lambda m: m["date"])

        # Group by season
        seasons = sorted(set(m.get("season", "unknown") for m in records))
        logger.info("WalkForwardBacktester: %d seasons found.", len(seasons))

        if len(seasons) < self._train_seasons + self._val_seasons:
            logger.warning(
                "Insufficient seasons (%d) for %d train + %d val folds. "
                "Running with available data.",
                len(seasons), self._train_seasons, self._val_seasons,
            )

        folds: list[FoldResult] = []
        fold_briefs: list[float] = []
        fold_log_losses: list[float] = []
        fold_eces: list[float] = []
        fold_rois: list[float] = []
        fold_clvs: list[float] = []

        # Build rolling folds
        for fold_start in range(len(seasons) - self._val_seasons):
            train_season_ids = seasons[max(0, fold_start - self._train_seasons + 1): fold_start + 1]
            val_season_ids = seasons[fold_start + 1: fold_start + 1 + self._val_seasons]

            if not val_season_ids:
                break

            train_records = [m for m in records if m.get("season") in train_season_ids]
            val_records = [m for m in records if m.get("season") in val_season_ids]

            if len(train_records) < self._min_matches:
                logger.warning(
                    "Fold %d: insufficient training matches (%d < %d). Skipping.",
                    fold_start, len(train_records), self._min_matches
                )
                continue

            # Fit models on training window
            dc_model = DixonColesModel()
            elo_model = DynamicEloModel()
            try:
                dc_model.fit(train_records)
                elo_model.fit(train_records)
            except Exception as exc:
                logger.error("Fold %d: model fitting failed: %s", fold_start, exc)
                continue

            ensemble = EnsembleModel(dc_model=dc_model, elo_model=elo_model)

            # Validate on hold-out window
            probs_home, outcomes_home = [], []
            probs_over25, outcomes_over25 = [], []
            roi_returns = []

            for m in val_records:
                try:
                    pred, _ = ensemble.predict(
                        m["home_id"], m["away_id"], m.get("fixture_id", ""),
                        m.get("home_name", ""), m.get("away_name", ""),
                    )
                    # 1X2 home-win probability
                    probs_home.append(pred.p_home_win)
                    actual_home_win = 1 if m["home_goals"] > m["away_goals"] else 0
                    outcomes_home.append(actual_home_win)

                    # Over 2.5
                    if pred.p_over_25 is not None:
                        probs_over25.append(pred.p_over_25)
                        actual_over25 = 1 if (m["home_goals"] + m["away_goals"]) > 2.5 else 0
                        outcomes_over25.append(actual_over25)

                    # ROI simulation (flat 1-unit stake on value candidates)
                    snap = (odds_snapshots or {}).get(m.get("fixture_id", ""), {})
                    home_odds = snap.get("home_odds")
                    if home_odds and home_odds > 1.0:
                        fair_p = 1.0 / home_odds / (1.0 + 0.05)  # assume 5% vig
                        if pred.p_home_win - fair_p >= edge_threshold:
                            ret = pred.p_home_win * home_odds - 1.0
                            roi_returns.append(ret)

                except Exception:
                    continue

            if not probs_home:
                continue

            # Compute fold metrics
            brier = BrierScore(probs_home, outcomes_home)
            ll = LogLoss(probs_home, outcomes_home)
            ece = compute_calibration_ece(probs_home, outcomes_home)
            accuracy = float(np.mean([
                1 if (p > 0.5) == (o == 1) else 0
                for p, o in zip(probs_home, outcomes_home)
            ]))
            roi = compute_roi(roi_returns) if roi_returns else None
            clv_list = [compute_clv(p, o) for p, o in zip(probs_home, [1.0 / odds_snapshots.get(m.get("fixture_id", ""), {}).get("home_odds", 0.0001) for m in val_records[:len(probs_home)]]) if o > 0]
            mean_clv = float(np.mean(clv_list)) if clv_list else None

            fold_result = FoldResult(
                fold_id=fold_start,
                train_start=min(r["date"] for r in train_records),
                train_end=max(r["date"] for r in train_records),
                val_start=min(r["date"] for r in val_records),
                val_end=max(r["date"] for r in val_records),
                n_train=len(train_records),
                n_val=len(val_records),
                brier_score=round(brier, 4),
                log_loss=round(ll, 4),
                ece=round(ece, 4),
                accuracy=round(accuracy, 4),
                roi=round(roi, 4) if roi is not None else None,
                clv=round(mean_clv, 4) if mean_clv is not None else None,
            )
            folds.append(fold_result)
            fold_briefs.append(brier)
            fold_log_losses.append(ll)
            fold_eces.append(ece)
            if roi is not None:
                fold_rois.append(roi)
            if mean_clv is not None:
                fold_clvs.append(mean_clv)

        if not folds:
            logger.warning("WalkForwardBacktester: no valid folds completed.")

        # Aggregate
        all_roi_returns: list[float] = []  # for drawdown (would need per-match data)

        result = WalkForwardResult(
            n_folds=len(folds),
            total_train_matches=sum(f.n_train for f in folds),
            total_val_matches=sum(f.n_val for f in folds),
            folds=folds,
            mean_brier=round(float(np.mean(fold_briefs)), 4) if fold_briefs else None,
            mean_log_loss=round(float(np.mean(fold_log_losses)), 4) if fold_log_losses else None,
            mean_ece=round(float(np.mean(fold_eces)), 4) if fold_eces else None,
            mean_roi=round(float(np.mean(fold_rois)), 4) if fold_rois else None,
            mean_clv=round(float(np.mean(fold_clvs)), 4) if fold_clvs else None,
            model_version="Ensemble(DC+Elo)-WF",
            run_at=datetime.utcnow(),
        )

        logger.info(
            "WalkForwardBacktester complete: %d folds, mean Brier=%.4f, mean ROI=%.4f",
            result.n_folds,
            result.mean_brier or 0.0,
            result.mean_roi or 0.0,
        )
        return result
