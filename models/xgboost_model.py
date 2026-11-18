"""
models/xgboost_model.py
=======================
XGBoost match outcome classifier. Disabled by default (ENABLE_XGBOOST=False).
Predicts 1X2 probabilities from engineered team features.
Falls back gracefully to None if xgboost is not installed or model not trained.
"""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.info("xgboost not installed — XGBoostModel disabled.")


def _features_to_vector(home_features: dict, away_features: dict) -> "np.ndarray":
    """Convert home/away feature dicts to a flat numpy feature vector."""
    import numpy as np
    home = home_features
    away = away_features
    return np.array([
        home.get("form_w3", 0.0),
        home.get("form_w5", 0.0),
        home.get("form_w10", 0.0),
        home.get("goals_scored_per_game", 0.0),
        home.get("goals_conceded_per_game", 0.0),
        home.get("xg_per_game") or 0.0,
        home.get("xga_per_game") or 0.0,
        home.get("elo_rating", 1500.0) / 1000.0,
        home.get("days_since_last_match") or 7.0,
        home.get("matches_in_last_14_days", 2),

        away.get("form_w3", 0.0),
        away.get("form_w5", 0.0),
        away.get("form_w10", 0.0),
        away.get("goals_scored_per_game", 0.0),
        away.get("goals_conceded_per_game", 0.0),
        away.get("xg_per_game") or 0.0,
        away.get("xga_per_game") or 0.0,
        away.get("elo_rating", 1500.0) / 1000.0,
        away.get("days_since_last_match") or 7.0,
        away.get("matches_in_last_14_days", 2),

        # Relative
        home.get("elo_rating", 1500.0) - away.get("elo_rating", 1500.0),
        home.get("goals_scored_per_game", 0.0) - away.get("goals_conceded_per_game", 0.0),
        away.get("goals_scored_per_game", 0.0) - home.get("goals_conceded_per_game", 0.0),
    ], dtype=float)


class XGBoostMatchModel:
    """
    XGBoost 1X2 classifier.
    Labels: 0=Home Win, 1=Draw, 2=Away Win

    Disabled when:
      - ENABLE_XGBOOST=False in settings
      - xgboost package not installed
      - No trained model exists at MODEL_REGISTRY_PATH

    Returns None in all disabled cases — EnsembleModel must handle None.
    """

    FEATURE_NAMES = [
        "home_form_w3", "home_form_w5", "home_form_w10",
        "home_goals_scored", "home_goals_conceded",
        "home_xg", "home_xga", "home_elo_norm", "home_rest_days", "home_matches_14d",
        "away_form_w3", "away_form_w5", "away_form_w10",
        "away_goals_scored", "away_goals_conceded",
        "away_xg", "away_xga", "away_elo_norm", "away_rest_days", "away_matches_14d",
        "elo_diff", "home_atk_vs_away_def", "away_atk_vs_home_def",
    ]

    def __init__(self) -> None:
        self._model = None
        self._trained = False
        self._registry = Path(settings.model_registry_path)

    @property
    def is_enabled(self) -> bool:
        return settings.enable_xgboost and _XGB_AVAILABLE

    def train(self, X: "np.ndarray", y: "np.ndarray") -> None:
        if not self.is_enabled:
            logger.warning("XGBoostMatchModel.train() called but model is disabled.")
            return

        import xgboost as xgb
        import numpy as np

        self._model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y, verbose=False)
        self._trained = True
        logger.info("XGBoostMatchModel trained on %d samples.", len(y))

    def save(self, name: str = "xgboost_match.pkl") -> None:
        if not self._trained:
            raise RuntimeError("Model not trained — cannot save.")
        self._registry.mkdir(parents=True, exist_ok=True)
        path = self._registry / name
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("XGBoostMatchModel saved to %s", path)

    def load(self, name: str = "xgboost_match.pkl") -> bool:
        if not _XGB_AVAILABLE:
            return False
        path = self._registry / name
        if not path.exists():
            logger.info("No saved XGBoost model at %s — skipping.", path)
            return False
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._trained = True
        logger.info("XGBoostMatchModel loaded from %s", path)
        return True

    def predict(self, home_features: dict, away_features: dict) -> Optional[tuple[float, float, float]]:
        """Returns (p_home, p_draw, p_away) or None if disabled/not trained."""
        if not self.is_enabled or not self._trained:
            return None
        import numpy as np
        vec = _features_to_vector(home_features, away_features).reshape(1, -1)
        proba = self._model.predict_proba(vec)[0]
        return float(proba[0]), float(proba[1]), float(proba[2])

    def predict_batch(self, feature_pairs: list[tuple[dict, dict]]) -> list[Optional[tuple[float, float, float]]]:
        if not self.is_enabled or not self._trained:
            return [None] * len(feature_pairs)
        import numpy as np
        X = np.stack([_features_to_vector(h, a) for h, a in feature_pairs])
        probas = self._model.predict_proba(X)
        return [(float(p[0]), float(p[1]), float(p[2])) for p in probas]
