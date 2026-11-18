"""
scripts/populate_db.py
======================
Pulls live FPL data (Teams and Fixtures) and populates the local SQLite database.
This provides REAL historical match data for the prediction engine.
"""
from __future__ import annotations
import sys
import os
from datetime import datetime, timezone
import dateutil.parser

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.repository import init_db, get_session
from db.models import Team, Match, TeamMatchStats, Competition
from fantasy.fpl_client import fpl_client

def populate():
    print("Initializing Database...")
    init_db()

    with get_session() as session:
        # Create competition if missing
        comp = session.query(Competition).filter_by(id="PL").first()
        if not comp:
            comp = Competition(id="PL", name="Premier League", country="England", tier=1)
            session.add(comp)
            
        print("Fetching Teams from FPL...")
        teams_data = fpl_client.get_teams()
        
        team_id_map = {}
        for t in teams_data:
            short_name = t["short_name"].upper()
            team_id_map[t["id"]] = short_name
            
            team = session.query(Team).filter_by(id=short_name).first()
            if not team:
                team = Team(id=short_name, name=t["name"])
                session.add(team)
        
        session.commit()
        print(f"Saved {len(teams_data)} teams to database.")

        print("Fetching Fixtures from FPL...")
        fixtures = fpl_client.get_fixtures()
        
        saved_matches = 0
        for f in fixtures:
            # Only process played matches with a valid score
            if not f.get("finished") or f.get("team_h_score") is None:
                continue
                
            home_id = team_id_map.get(f["team_h"])
            away_id = team_id_map.get(f["team_a"])
            
            if not home_id or not away_id:
                continue
                
            match_id = f"PL-2425-{f['id']}"
            match = session.query(Match).filter_by(id=match_id).first()
            
            date_str = f.get("kickoff_time")
            kickoff = dateutil.parser.parse(date_str) if date_str else datetime.now(timezone.utc)
            kickoff = kickoff.replace(tzinfo=None) # sqlite doesn't like tz
            
            if not match:
                match = Match(
                    id=match_id,
                    competition_id="PL",
                    season_year=2024,
                    kickoff_utc=kickoff,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    home_score=f["team_h_score"],
                    away_score=f["team_a_score"],
                    status="finished"
                )
                session.add(match)
                
                # Add stats
                h_stats = TeamMatchStats(
                    match_id=match_id,
                    team_id=home_id,
                    is_home=True,
                    goals=f["team_h_score"],
                    xg=float(f["team_h_score"]) # Use actual goals as proxy for xG
                )
                session.add(h_stats)
                
                a_stats = TeamMatchStats(
                    match_id=match_id,
                    team_id=away_id,
                    is_home=False,
                    goals=f["team_a_score"],
                    xg=float(f["team_a_score"])
                )
                session.add(a_stats)
                saved_matches += 1
                
        session.commit()
        print(f"Saved {saved_matches} historical matches to database.")
        print("Database population complete!")

if __name__ == "__main__":
    populate()
