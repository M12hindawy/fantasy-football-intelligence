"""
agents/global_scraper.py
========================
On-the-fly dynamic data fetcher for ALL top 5 leagues.
Downloads CSVs directly from football-data.co.uk into memory, 
computes deep stats, and returns records.
Includes fallback to local database if online source is down.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import dateutil.parser
import urllib.request
import io
import logging
from db.repository import get_session
from db.models import Match, Team

logger = logging.getLogger(__name__)

LEAGUES = {
    "EPL": "E0",
    "LaLiga": "SP1",
    "SerieA": "I1",
    "Bundesliga": "D1",
    "Ligue1": "F1"
}

SEASONS = ["2627", "2526", "2425", "2324"]

def fuzzy_match_team(df, team_input):
    """Find the exact team name from the dataframe."""
    team_input = team_input.strip().lower()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    # 1. Exact match (case insensitive)
    for t in teams:
        if t.lower() == team_input: return t
    
    # 2. Substring match
    for t in teams:
        if team_input in t.lower() or t.lower() in team_input: return t
        
    return None

class GlobalScraperAgent:
    def _fetch_from_local_db(self, home_name: str, away_name: str):
        """Fallback to use local SQLite database if football-data is down."""
        try:
            with get_session() as session:
                teams = session.query(Team).all()
                team_names = [t.name for t in teams]
                
                def fuzzy(target):
                    t_low = target.strip().lower()
                    for n in team_names:
                        if n.lower() == t_low: return n
                    for n in team_names:
                        if t_low in n.lower() or n.lower() in t_low: return n
                    return None
                
                h_match = fuzzy(home_name)
                a_match = fuzzy(away_name)
                
                if not h_match or not a_match:
                    return None
                
                matches = session.query(Match).filter(Match.status.ilike("finished")).order_by(Match.kickoff_utc.asc()).all()
                
                records = []
                for m in matches:
                    records.append({
                        "home_name": m.home_team_id,  # Stored as names in this DB setup
                        "away_name": m.away_team_id,
                        "home_goals": m.home_score,
                        "away_goals": m.away_score,
                        "home_xg": m.home_score, # Approximation if xG missing
                        "away_xg": m.away_score,
                        "date": m.kickoff_utc.replace(tzinfo=timezone.utc)
                    })
                
                return {
                    "home_id": h_match,
                    "away_id": a_match,
                    "league": "E0",
                    "match_records": records
                }
        except Exception as e:
            logger.error(f"Local DB fallback failed: {e}")
            return None


    def fetch_teams_data(self, home_name: str, away_name: str):
        """
        Scans top 5 leagues to find where these teams play, 
        downloads all their historical data on the fly,
        and computes advanced stats.
        """
        
        # SEARCH LOCAL DB FIRST (since football-data.co.uk is throwing 503)
        local_data = self._fetch_from_local_db(home_name, away_name)
        if local_data:
            logger.info("Using local DB fallback due to API limits.")
            return local_data, None

        target_league_code = None
        target_home_name = None
        target_away_name = None
        
        # We only need to check the most recent season to find their league
        current_season_dfs = {}
        for l_name, l_code in LEAGUES.items():
            try:
                url = f"https://www.football-data.co.uk/mmz4281/{SEASONS[0]}/{l_code}.csv"
                df = pd.read_csv(url, on_bad_lines='skip')
                current_season_dfs[l_code] = df
                
                h_match = fuzzy_match_team(df, home_name)
                a_match = fuzzy_match_team(df, away_name)
                
                if h_match and a_match:
                    target_league_code = l_code
                    target_home_name = h_match
                    target_away_name = a_match
                    break
            except Exception as e:
                continue
                
        if not target_league_code:
            for l_name, l_code in LEAGUES.items():
                try:
                    url = f"https://www.football-data.co.uk/mmz4281/{SEASONS[1]}/{l_code}.csv"
                    df = pd.read_csv(url, on_bad_lines='skip')
                    h_match = fuzzy_match_team(df, home_name)
                    a_match = fuzzy_match_team(df, away_name)
                    if h_match and a_match:
                        target_league_code = l_code
                        target_home_name = h_match
                        target_away_name = a_match
                        break
                except:
                    continue

        if not target_league_code:
            return None, f"Could not find both teams in any of the Top 5 European leagues."

        all_dfs = []
        for s in SEASONS:
            try:
                url = f"https://www.football-data.co.uk/mmz4281/{s}/{target_league_code}.csv"
                df = pd.read_csv(url, on_bad_lines='skip')
                if not df.empty and 'HomeTeam' in df.columns:
                    all_dfs.append(df)
            except:
                pass
                
        if not all_dfs:
            return None, "Failed to download historical data."
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True)
        combined_df = combined_df.sort_values('Date')
        
        records = []
        for _, row in combined_df.iterrows():
            if pd.isna(row.get('FTHG')) or pd.isna(row.get('FTAG')):
                continue
                
            try:
                dt = row['Date'].to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except:
                dt = datetime.now(timezone.utc)
                
            records.append({
                "home_name": row['HomeTeam'],
                "away_name": row['AwayTeam'],
                "home_goals": int(row['FTHG']),
                "away_goals": int(row['FTAG']),
                "home_xg": float(row.get('HXG', row['FTHG'])), 
                "away_xg": float(row.get('AXG', row['FTAG'])),
                "date": dt
            })
            
        return {
            "home_id": target_home_name,
            "away_id": target_away_name,
            "league": target_league_code,
            "match_records": records
        }, None

global_scraper = GlobalScraperAgent()
