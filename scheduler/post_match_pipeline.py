"""
scheduler/post_match_pipeline.py
================================
Post-match evaluation pipeline.
After a match finishes: store result, compare to prediction, update model metrics.
"""
from __future__ import annotations
import logging
import urllib.request
import json
from datetime import datetime, timezone

from db.repository import get_session
from db.models import Match, TeamMatchStats, Prediction

logger = logging.getLogger(__name__)


class PostMatchPipeline:
    """
    Runs after every completed match:
      1. Check FPL for finished fixtures
      2. Store final result in Match table
      3. Compare against stored Prediction
      4. Log model accuracy
      5. Trigger Elo store refresh
    """

    def run(self) -> dict:
        summary = {"processed": 0, "skipped": 0, "errors": 0}
        try:
            finished = self._get_finished_fpl_fixtures()
            for fixture in finished:
                try:
                    result = self._process_fixture(fixture)
                    if result == "processed":
                        summary["processed"] += 1
                    else:
                        summary["skipped"] += 1
                except Exception as e:
                    logger.error(f"Error processing fixture {fixture.get('id')}: {e}")
                    summary["errors"] += 1
        except Exception as e:
            logger.error(f"PostMatchPipeline.run() error: {e}")
            summary["errors"] += 1

        if summary["processed"] > 0:
            # Refresh the Elo store with new match data
            try:
                from data.elo_store import elo_store
                elo_store.refresh()
                logger.info("Elo store refreshed after post-match pipeline.")
            except Exception as e:
                logger.warning(f"Could not refresh Elo store: {e}")

        logger.info(f"PostMatchPipeline complete: {summary}")
        return summary

    def _get_finished_fpl_fixtures(self) -> list[dict]:
        url = "https://fantasy.premierleague.com/api/fixtures/"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            fixtures = json.loads(r.read())
        return [f for f in fixtures if f.get("finished") and f.get("team_h_score") is not None]

    def _process_fixture(self, fixture: dict) -> str:
        fixture_id = str(fixture["id"])

        with get_session() as session:
            # Check if already stored
            match = session.query(Match).filter(Match.provider_id == fixture_id).first()
            if not match:
                return "skipped"  # Not tracked yet

            # Update final score if missing
            if match.home_score is None:
                match.home_score = fixture.get("team_h_score", 0)
                match.away_score = fixture.get("team_a_score", 0)
                match.status = "FINISHED"
                match.updated_at = datetime.now(timezone.utc)
                session.commit()
                logger.info(f"Updated result for match {match.id}: {match.home_score}-{match.away_score}")

            # Find stored prediction and log accuracy
            pred = session.query(Prediction).filter(Prediction.match_id == match.id).order_by(Prediction.created_at.desc()).first()
            if pred and pred.actual_home_score is None:
                pred.actual_home_score = match.home_score
                pred.actual_away_score = match.away_score
                # Determine actual outcome
                if match.home_score > match.away_score:
                    pred.actual_outcome = "HOME_WIN"
                    predicted_prob = pred.home_win_probability
                elif match.home_score < match.away_score:
                    pred.actual_outcome = "AWAY_WIN"
                    predicted_prob = pred.away_win_probability
                else:
                    pred.actual_outcome = "DRAW"
                    predicted_prob = pred.draw_probability

                # Brier score for this match
                brier = round((1.0 - (predicted_prob or 0.5)) ** 2, 4)
                pred.brier_score = brier
                session.commit()
                logger.info(f"Prediction evaluated: outcome={pred.actual_outcome} brier={brier:.4f}")

        return "processed"


post_match_pipeline = PostMatchPipeline()
