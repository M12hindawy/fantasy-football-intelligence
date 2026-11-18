"""
scripts/fetch_real_history.py - Updated for 2026
Downloads ALL available PL seasons including current 2026-27
"""
import sys, os, urllib.request, csv
from datetime import datetime
import dateutil.parser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.repository import get_session, init_db
from db.models import Match, TeamMatchStats, Team

TEAM_MAP = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Chelsea": "CHE",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Liverpool": "LIV", "Luton": "LUT", "Man City": "MCI",
    "Man United": "MUN", "Newcastle": "NEW", "Nott'm Forest": "NFO",
    "Sheffield United": "SHU", "Tottenham": "TOT", "West Ham": "WHU",
    "Wolves": "WOL", "Burnley": "BUR", "Ipswich": "IPS",
    "Southampton": "SOU", "Leicester": "LEI",
    # 2025-26 / 2026-27 teams
    "Leeds": "LEE", "Sunderland": "SUN", "Coventry": "COV",
    "Middlesbrough": "MID", "Hull": "HUL", "Norwich": "NOR",
    "Millwall": "MIL", "Swansea": "SWA", "Preston": "PRE",
    "Stoke": "STK", "Bristol City": "BSC", "Derby": "DER",
}

URLS = [
    ("2023-2024", "https://www.football-data.co.uk/mmz4281/2324/E0.csv"),
    ("2024-2025", "https://www.football-data.co.uk/mmz4281/2425/E0.csv"),
    ("2025-2026", "https://www.football-data.co.uk/mmz4281/2526/E0.csv"),
    ("2026-2027", "https://www.football-data.co.uk/mmz4281/2627/E0.csv"),
]

def load_all():
    init_db()
    with get_session() as session:
        existing_teams = {t.id for t in session.query(Team).all()}

        for season, url in URLS:
            print(f"\n--- Downloading {season} ---")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    lines = [l.decode("utf-8", errors="ignore") for l in resp.readlines()]

                reader = csv.DictReader(lines)
                count = 0
                for row in reader:
                    if not row.get("HomeTeam") or not row.get("FTHG"):
                        continue
                    try:
                        home_name = row["HomeTeam"].strip()
                        away_name = row["AwayTeam"].strip()
                        home_id = TEAM_MAP.get(home_name, home_name[:3].upper())
                        away_id = TEAM_MAP.get(away_name, away_name[:3].upper())

                        for tid, tname in [(home_id, home_name), (away_id, away_name)]:
                            if tid not in existing_teams:
                                session.add(Team(id=tid, name=tname))
                                existing_teams.add(tid)

                        date_str = row["Date"]
                        time_str = row.get("Time", "15:00")
                        try:
                            kickoff = dateutil.parser.parse(f"{date_str} {time_str}", dayfirst=True)
                        except:
                            kickoff = datetime.now()

                        match_id = f"PL-{season}-{home_id}-{away_id}-{kickoff.strftime('%Y%m%d')}"
                        if session.query(Match).filter_by(id=match_id).first():
                            continue

                        home_sot = int(row.get("HST") or 0)
                        away_sot = int(row.get("AST") or 0)
                        home_xg = round(home_sot * 0.3, 2)
                        away_xg = round(away_sot * 0.3, 2)

                        # If B365H odds exist, use them too (for future EV calc)
                        session.add(Match(
                            id=match_id,
                            competition_id="PL",
                            season_year=int(season.split("-")[0]),
                            kickoff_utc=kickoff,
                            home_team_id=home_id,
                            away_team_id=away_id,
                            home_score=int(row["FTHG"]),
                            away_score=int(row["FTAG"]),
                            status="finished",
                            data_source="football-data.co.uk"
                        ))
                        session.add(TeamMatchStats(
                            match_id=match_id, team_id=home_id, is_home=True,
                            goals=int(row["FTHG"]),
                            shots=int(row.get("HS") or 0),
                            shots_on_target=home_sot,
                            xg=home_xg
                        ))
                        session.add(TeamMatchStats(
                            match_id=match_id, team_id=away_id, is_home=False,
                            goals=int(row["FTAG"]),
                            shots=int(row.get("AS") or 0),
                            shots_on_target=away_sot,
                            xg=away_xg
                        ))
                        count += 1
                    except Exception as row_err:
                        continue

                session.commit()
                print(f"  Saved {count} new matches for {season}.")
            except Exception as e:
                print(f"  ERROR: {e}")
                session.rollback()

if __name__ == "__main__":
    load_all()
