# ⚽ Fantasy Football Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?style=for-the-badge)
![Dash](https://img.shields.io/badge/Plotly-Dash-blue?style=for-the-badge&logo=plotly)
![Multi-Agent](https://img.shields.io/badge/Multi--Agent-System-purple?style=for-the-badge)
![Dixon-Coles](https://img.shields.io/badge/Dixon--Coles-Statistical-green?style=for-the-badge)

**A production-grade multi-agent AI system for Fantasy Premier League optimization, combining statistical football modeling (Dixon-Coles), ML predictions (XGBoost), live odds integration, and an interactive Plotly Dash dashboard.**

</div>

---

## 🧠 System Overview

This system combines **15+ specialized AI agents** working in concert to provide:
- **Match outcome predictions** using the Dixon-Coles statistical model
- **Player value assessment** via XGBoost ML models
- **Live betting odds** analysis and market normalization
- **FPL team optimization** using linear programming
- **Multi-gameweek planning** with risk management
- **Real-time monitoring** and drift detection

## 🤖 Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                           │
└───────────────┬─────────────────────────────────────────────────┘
                │
    ┌───────────┼───────────────────────────────┐
    │           │                               │
    ▼           ▼                               ▼
┌───────┐  ┌──────────┐                  ┌──────────────┐
│ Data  │  │  Model   │                  │   Fantasy    │
│ Layer │  │  Layer   │                  │   Layer      │
│       │  │          │                  │              │
│• Data │  │• Dixon-  │                  │• FPL Client  │
│  Ing. │  │  Coles   │                  │• Gameweek    │
│• Odds │  │• XGBoost │                  │  Engine      │
│  API  │  │• ELO     │                  │• Optimizer   │
│• FPL  │  │• Ensemble│                  │• Multi-GW    │
│  API  │  │• Calib.  │                  │  Planner     │
└───────┘  └──────────┘                  └──────────────┘
```

### 🤖 All 15+ Agents

| Agent | Role |
|-------|------|
| `orchestrator.py` | Master coordinator — routes tasks to specialized agents |
| `data_ingestion_agent.py` | Fetches match data from Football-Data.org API |
| `odds_ingestion_agent.py` | Pulls live betting odds from multiple sources |
| `global_scraper.py` | Web scraping for match stats and news |
| `market_probability_agent.py` | Converts odds to true probabilities (margin removal) |
| `margin_removal_agent.py` | Removes bookmaker margin from odds |
| `market_normalization_agent.py` | Normalizes odds across different markets |
| `team_analysis_agent.py` | Analyzes team form, home/away records |
| `match_analysis_agent.py` | Generates match-level insights |
| `value_assessment_agent.py` | Finds value bets (model vs market disagreement) |
| `availability_agent.py` | Tracks player injuries, suspensions |
| `fantasy_agent.py` | FPL-specific recommendations |
| `lineup_agent.py` | Optimal lineup selection |
| `live_agent.py` | Real-time match monitoring |
| `league_context_agent.py` | League standings context |
| `responsible_use_guard_agent.py` | Responsible gambling safeguards |
| `risk_and_stake_agent.py` | Bankroll management and stake sizing |

## 📊 Statistical Models

### Dixon-Coles Model
The core match prediction engine uses the **Dixon-Coles (1997)** statistical model:
- **Poisson-based** goal scoring model for home and away teams
- **Rho correction** for low-score scoreline dependencies (0-0, 1-0, 0-1, 1-1)
- **Time-decay weighting** — recent matches weighted more heavily
- Outputs: Home win / Draw / Away win probabilities

### XGBoost Player Projections
- Feature engineering: form, home/away, opponent strength, fixture difficulty
- **Walk-forward validation** to prevent data leakage
- Calibrated outputs using isotonic regression

### ELO Rating System
- Dynamic team strength ratings updated after each match
- Home advantage modeled explicitly
- Used as features for the ML model

## 🚀 Quick Start

```bash
git clone https://github.com/M12hindawy/fantasy-football-intelligence.git
cd fantasy-football-intelligence
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
python main.py
```

### Run the Dashboard
```bash
python dashboard.py
# Open http://localhost:8050
```

### Run Predictions
```bash
python main.py --mode predict --gameweek 15
```

## 📁 Project Structure

```
fantasy-football-intelligence/
├── main.py                   # Entry point
├── dashboard.py              # Plotly Dash interactive dashboard
├── api.py                    # REST API endpoints
├── requirements.txt
├── .env.example              # API key template
├── agents/                   # 15+ specialized AI agents
│   ├── orchestrator.py       # Master coordinator
│   ├── data_ingestion_agent.py
│   ├── market_probability_agent.py
│   ├── value_assessment_agent.py
│   ├── risk_and_stake_agent.py
│   └── ...
├── models/                   # Statistical and ML models
│   ├── dixon_coles.py        # Dixon-Coles implementation
│   ├── xgboost_model.py      # XGBoost player projections
│   ├── elo_model.py          # ELO rating system
│   ├── ensemble.py           # Model ensemble voting
│   └── calibration.py        # Probability calibration
├── data/                     # Data management
│   ├── fetchers/             # API clients
│   ├── schemas.py            # Data validation
│   └── cache.py              # Caching layer
├── fantasy/                  # FPL-specific modules
│   ├── fpl_client.py         # FPL API client
│   ├── optimizer.py          # Linear programming optimizer
│   └── multi_gw_engine.py    # Multi-gameweek planner
├── backtesting/              # Walk-forward validation
│   ├── walk_forward.py
│   └── metrics.py
├── simulation/               # Monte Carlo simulations
│   ├── monte_carlo.py
│   └── season_simulator.py
├── monitoring/               # Production monitoring
│   ├── drift_detector.py
│   └── model_monitor.py
└── tests/                    # Test suite
```

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Statistical Modeling** | Dixon-Coles, Poisson regression |
| **Machine Learning** | XGBoost, Scikit-learn |
| **Dashboard** | Plotly Dash |
| **Data Sources** | Football-Data.org API, FPL API, Odds API |
| **Database** | SQLite (SQLAlchemy ORM) |
| **Optimization** | Linear Programming (PuLP) |
| **Simulation** | Monte Carlo |
| **Testing** | pytest |
| **Caching** | SQLite-based cache layer |

## ⚠️ Responsible Use

This system is built for **educational and research purposes** in sports analytics. It includes a dedicated `responsible_use_guard_agent.py` that:
- Flags potentially harmful usage patterns
- Enforces stake size limits
- Provides problem gambling resources

## 👤 Author

**Mohamed Ashraf Abdelhamed**  
📍 Giza, Egypt | AI & Data Engineering Student  
🔗 [LinkedIn](https://linkedin.com/in/mohamed-ashraf-b7b5a72b6/) | [GitHub](https://github.com/M12hindawy) | [Portfolio](https://mohamedhendawy.netlify.app)

## 📄 License

MIT License — See [LICENSE](LICENSE) file.
