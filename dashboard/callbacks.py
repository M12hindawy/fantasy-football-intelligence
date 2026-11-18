"""
dashboard/callbacks.py
======================
Dash callbacks - uses GlobalScraperAgent to support ALL Top 5 leagues on the fly!
"""
from dash import Input, Output, State, html, callback
import dash_bootstrap_components as dbc
import numpy as np
from datetime import datetime, timezone

from dashboard.layout import predictions_layout, fantasy_layout, monitoring_layout, today_layout
from agents.orchestrator import orchestrator, OrchestratorInput
from agents.live_agent import live_agent
from agents.global_scraper import global_scraper
from monitoring.alert_manager import alert_manager
from db.repository import get_session
from db.models import Match, TeamMatchStats, Team

def avg(lst, n=10):
    r = lst[:n]
    return round(sum(r) / len(r), 2) if r else 0.0

def form_str(results, n=5):
    colors = {"W": "ðŸŸ¢", "D": "ðŸŸ¡", "L": "ðŸ”´"}
    return " ".join(colors.get(r, r) for r in results[:n]) if results else "—"

def win_pct(results, n=10):
    r = results[:n]
    return round(r.count("W") / len(r), 3) if r else 0.5



@callback(
    Output("predictions-output", "children"),
    Input("analyze-btn", "n_clicks"),
    State("home-team-input", "value"),
    State("away-team-input", "value"),
    prevent_initial_call=True
)
def update_predictions(n_clicks, home_input, away_input):
    if not home_input or not away_input:
        return dbc.Alert("Enter both team names (e.g. 'Barcelona' and 'Vallecano').", color="warning")

    # â”€â”€ GLOBAL SCRAPE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    data_res, err = global_scraper.fetch_teams_data(home_input, away_input)
    
    if err or not data_res:
        return dbc.Alert(err or "Error fetching data.", color="danger")
        
    home_id = data_res["home_id"]
    away_id = data_res["away_id"]
    match_records = data_res["match_records"]
    
    home_name = home_id
    away_name = away_id

    home_home_goals, home_home_conceded, home_home_xg = [], [], []
    away_away_goals, away_away_conceded, away_away_xg = [], [], []
    home_results, away_results, h2h_records = [], [], []

    for rec in match_records:
        hg = rec["home_goals"]
        ag = rec["away_goals"]
        kickoff = rec["date"]
        
        if rec["home_name"] == home_id:
            home_home_goals.append(hg); home_home_conceded.append(ag)
            home_home_xg.append(rec["home_xg"])
            home_results.append("W" if hg > ag else ("D" if hg == ag else "L"))
        elif rec["away_name"] == home_id:
            home_results.append("W" if ag > hg else ("D" if ag == hg else "L"))

        if rec["away_name"] == away_id:
            away_away_goals.append(ag); away_away_conceded.append(hg)
            away_away_xg.append(rec["away_xg"])
            away_results.append("W" if ag > hg else ("D" if ag == hg else "L"))
        elif rec["home_name"] == away_id:
            away_results.append("W" if hg > ag else ("D" if hg == ag else "L"))

        if ((rec["home_name"] == home_id and rec["away_name"] == away_id) or
            (rec["home_name"] == away_id and rec["away_name"] == home_id)):
            h2h_records.append({
                "date": kickoff.strftime("%Y-%m-%d"),
                "home": rec["home_name"], "score": f"{int(hg)}-{int(ag)}", "away": rec["away_name"]
            })

    # Use xG for mu (much more predictive than raw goals)
    # â”€â”€ COMPUTE ADVANCED STATS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # For each team: compute attack & defensive strength separately
    # This is proper Dixon-Coles methodology:
    # mu_home = home_attack_avg Ã— away_defensive_weakness_factor
    # mu_away = away_attack_avg Ã— home_defensive_weakness_factor

    home_conceded_avg = avg(home_home_conceded, 10) or 1.2
    away_conceded_avg = avg(away_away_conceded, 10) or 1.2
    league_avg_goals = 1.5  # Approximate PL/top league average

    # Attack strength = how many goals team scores relative to league avg
    home_attack = (avg(home_home_xg, 10) or league_avg_goals) / league_avg_goals
    away_attack = (avg(away_away_xg, 10) or league_avg_goals) / league_avg_goals

    # Defensive weakness = how many goals team concedes relative to league avg (higher = weaker defense)
    home_defense_weakness = home_conceded_avg / league_avg_goals
    away_defense_weakness = away_conceded_avg / league_avg_goals

    # Proper mu: home team scores based on their attack vs away team's defense (and vice versa)
    mu_home = max(0.3, home_attack * away_defense_weakness * league_avg_goals)
    mu_away = max(0.3, away_attack * home_defense_weakness * league_avg_goals)

    # â”€â”€ DYNAMIC ELO & ENSEMBLE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from data.elo_store import elo_store
    home_elo = elo_store.get_elo(home_id)
    away_elo = elo_store.get_elo(away_id)
    
    # Calculate probabilities from both models
    elo_ph, elo_pd, elo_pa = elo_store.get_home_win_prob(home_id, away_id)

    # â”€â”€ LIVE DETECTION (pass full team names for fuzzy matching) â”€â”€â”€â”€â”€
    is_live = False
    live_match = live_agent.fetch_live_data(home_id, away_id)
    
    if live_match and live_match.is_live:
        is_live = True
        remaining_ratio = max(0.0, 90 - live_match.minute) / 90.0
        rem_mu_home = mu_home * remaining_ratio
        rem_mu_away = mu_away * remaining_ratio
        
        inp = OrchestratorInput(
            fixture_id=f"{home_id}-vs-{away_id}-LIVE",
            home_team_id=home_id, away_team_id=away_id, competition_code="GLOBAL",
            mu_home=rem_mu_home, mu_away=rem_mu_away,
            home_elo=home_elo, away_elo=away_elo, match_records=match_records
        )
        result = orchestrator.analyze(inp)
        
        from simulation.monte_carlo import score_probability_matrix
        mat = score_probability_matrix(rem_mu_home, rem_mu_away, -0.10)
        p_home_win = p_draw = p_away_win = p_btts = p_over25 = 0.0
        c_hg = live_match.home_score
        c_ag = live_match.away_score
        
        for h in range(mat.shape[0]):
            for a in range(mat.shape[1]):
                prob = mat[h, a]
                final_h = h + c_hg; final_a = a + c_ag
                if final_h > final_a: p_home_win += prob
                elif final_h < final_a: p_away_win += prob
                else: p_draw += prob
                if final_h > 0 and final_a > 0: p_btts += prob
                if final_h + final_a > 2.5: p_over25 += prob
                
        result.p_home_win = p_home_win; result.p_draw = p_draw; result.p_away_win = p_away_win
        result.expected_home_goals = c_hg + rem_mu_home
        result.expected_away_goals = c_ag + rem_mu_away
        result.p_btts = p_btts; result.p_over_25 = p_over25
        result.model_name = "In-Play Poisson Simulation (Live)"
    else:
        inp = OrchestratorInput(
            fixture_id=f"{home_id}-vs-{away_id}",
            home_team_id=home_id, away_team_id=away_id, competition_code="GLOBAL",
            mu_home=mu_home, mu_away=mu_away,
            home_elo=home_elo, away_elo=away_elo, match_records=match_records
        )
        result = orchestrator.analyze(inp)

    # â”€â”€ BUILD UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    output = []
    
    if is_live:
        output.append(dbc.Alert([
            html.H4("ðŸ”´ LIVE MATCH IN PROGRESS", className="alert-heading text-danger fw-bold"),
            html.Hr(),
            html.H3(f"{home_name} {live_match.home_score} - {live_match.away_score} {away_name}", className="text-center fw-bold"),
            html.P(f"Minute: {live_match.minute}'", className="text-center text-muted fw-bold"),
            html.P("Probabilities adjusted dynamically based on remaining time!", className="text-center mb-0 small")
        ], color="danger"))
    
    dq_grade = result.dq_grade
    dq_color = {"A":"success","B":"success","C":"warning","D":"warning","F":"danger"}.get(dq_grade,"secondary")
    
    output.append(html.Div([
        dbc.Badge(f"Data Quality: Grade {dq_grade}", color=dq_color, className="me-2 fs-6"),
        dbc.Badge(f"Matches Scraped: {len(match_records)}", color="dark", className="me-2 fs-6"),
        dbc.Badge(f"H2H: {len(h2h_records)} meetings", color="primary", className="me-2 fs-6"),
        dbc.Badge(f"Home Elo: {home_elo}", color="info", className="me-2 fs-6"),
        dbc.Badge(f"Away Elo: {away_elo}", color="info", className="fs-6"),
    ], className="mb-3 d-flex flex-wrap gap-1"))

    # â”€â”€ COMPUTE ENSEMBLE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ens_ph = (result.p_home_win + elo_ph) / 2.0
    ens_pd = (result.p_draw + elo_pd) / 2.0
    ens_pa = (result.p_away_win + elo_pa) / 2.0

    output.append(dbc.Card([
        dbc.CardHeader(html.H5(f"âš½ Match Probabilities — Ensemble (DC + Dynamic Elo)")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.H2(f"{ens_ph*100:.1f}%", className="text-success fw-bold"), html.P(f"ðŸ  {home_name} Win")], className="text-center"),
                dbc.Col([html.H2(f"{ens_pd*100:.1f}%", className="text-warning fw-bold"), html.P("ðŸ¤ Draw")], className="text-center"),
                dbc.Col([html.H2(f"{ens_pa*100:.1f}%", className="text-danger fw-bold"), html.P(f"âœˆï¸ {away_name} Win")], className="text-center"),
            ]),
            html.Hr(),
            html.Div([
                html.Small(f"Dixon-Coles: {result.p_home_win*100:.1f}% | {result.p_draw*100:.1f}% | {result.p_away_win*100:.1f}%", className="text-muted d-block text-center"),
                html.Small(f"Dynamic Elo: {elo_ph*100:.1f}% | {elo_pd*100:.1f}% | {elo_pa*100:.1f}%", className="text-muted d-block text-center mb-3"),
            ]),
            dbc.Row([
                dbc.Col(html.P(f"âš½ Final xG: {result.expected_home_goals:.2f}")),
                dbc.Col(html.P(f"âš½ Final xG: {result.expected_away_goals:.2f}")),
                dbc.Col(html.P(f"ðŸŽ¯ BTTS: {result.p_btts*100:.1f}%")),
                dbc.Col(html.P(f"ðŸ“ˆ Over 2.5: {result.p_over_25*100:.1f}%")),
            ]),
        ])
    ], className="mb-3"))

    # Advanced Stats Display
    # Calculate some advanced stats to show
    def sum_stat(records, field, team_id, is_home):
        return sum(r[field] for r in records if (r["home_name"] == team_id if is_home else r["away_name"] == team_id))

    output.append(dbc.Card([
        dbc.CardHeader(html.H5("ðŸ“Š Team Stats (Computed from Real Match History)")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6(f"ðŸ  {home_name} (Home Stats)"),
                    html.P(f"Last 5 overall: {form_str(home_results, 5)}"),
                    html.P(f"Avg goals scored: {avg(home_home_goals)} / game"),
                    html.P(f"Avg xG computed: {avg([r['home_xg'] for r in match_records if r['home_name'] == home_id])} / game"),
                    html.P(f"Avg shots on target: {avg([r['home_shots_target'] for r in match_records if r['home_name'] == home_id])} / game"),
                ]),
                dbc.Col([
                    html.H6(f"âœˆï¸ {away_name} (Away Stats)"),
                    html.P(f"Last 5 overall: {form_str(away_results, 5)}"),
                    html.P(f"Avg goals scored: {avg(away_away_goals)} / game"),
                    html.P(f"Avg xG computed: {avg([r['away_xg'] for r in match_records if r['away_name'] == away_id])} / game"),
                    html.P(f"Avg shots on target: {avg([r['away_shots_target'] for r in match_records if r['away_name'] == away_id])} / game"),
                ]),
            ]),
        ])
    ], className="mb-3"))

    if h2h_records:
        h2h_home_w = sum(1 for r in h2h_records if r["home"] == home_id and int(r["score"].split("-")[0]) > int(r["score"].split("-")[1]))
        h2h_away_w = sum(1 for r in h2h_records if r["home"] == away_id and int(r["score"].split("-")[0]) > int(r["score"].split("-")[1]))
        h2h_draws = len(h2h_records) - h2h_home_w - h2h_away_w
        h2h_rows = [html.Tr([html.Td(r["date"]), html.Td(r["home"]), html.Td(r["score"], className="fw-bold text-center"), html.Td(r["away"])]) for r in h2h_records[:6]]
        output.append(dbc.Card([
            dbc.CardHeader(html.H5(f"ðŸ”„ Head-to-Head (last {len(h2h_records)} meetings)")),
            dbc.CardBody([
                html.P(f"{home_name}: {h2h_home_w}W  |  Draws: {h2h_draws}  |  {away_name}: {h2h_away_w}W", className="fw-bold"),
                dbc.Table([html.Thead(html.Tr([html.Th("Date"), html.Th("Home"), html.Th("Score"), html.Th("Away")])), html.Tbody(h2h_rows)], bordered=True, hover=True, size="sm")
            ])
        ], className="mb-3"))

    rec_color = {"HOME_WIN":"success","AWAY_WIN":"danger","DRAW":"warning"}.get(result.recommendation,"secondary")
    output.append(dbc.Alert([html.H5(f"ðŸŽ¯ Recommendation: {result.recommendation}"), html.P(f"Confidence Strength: {result.recommendation_strength:.2f}")], color=rec_color))

    return html.Div(output)

from fantasy.fpl_client import fpl_client
from fantasy.optimizer import FPLOptimizer, PlayerOption

@callback(
    Output("fantasy-output", "children"),
    Input("optimize-btn", "n_clicks"),
    State("gw-input", "value"),
    prevent_initial_call=True
)
def update_fantasy(n_clicks, gameweek):
    gw = gameweek or fpl_client.get_next_gameweek() or 1
    players_data = fpl_client.get_players()
    teams_data = fpl_client.get_teams()
    team_map = {t["id"]: t["short_name"] for t in teams_data}
    
    # Team strength mapping for custom adjustments
    team_strength = {t["id"]: t["strength"] for t in teams_data}
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    
    candidates = []
    for p in players_data:
        # 1. Player Status: Only consider available players
        if p.get("status") not in ("a", "d"): continue
        chance = p.get("chance_of_playing_next_round")
        if chance is not None and int(chance) < 75: continue
        
        pid = str(p["id"])
        name = f'{p.get("first_name","")} {p.get("second_name","")}'.strip()
        price = p.get("now_cost", 50) / 10.0
        team_id = p.get("team")
        club = team_map.get(team_id, "UNK")
        pos = pos_map.get(p.get("element_type"), "MID")
        
        minutes = p.get("minutes", 0)
        
        # 2. Extract crucial underlying stats (as requested by user)
        # ep_next: FPL's internal prediction based on FIXTURE DIFFICULTY (Ø§Ù„ÙØ±ÙŠÙ‚ Ø§Ù„Ù„ÙŠ Ù‡ØªÙ‚Ø§Ø¨Ù„ Ø§Ù„ÙØ±ÙŠÙ‚)
        ep_next = float(p.get("ep_next", 0.0) or 0.0)
        
        # form: Player's recent performance (Ø§Ø¯Ø§Ø¡ Ø§Ù„Ù„Ø§Ø¹Ø¨ÙŠÙ†)
        form = float(p.get("form", 0.0) or 0.0)
        
        # ict_index: Influence, Creativity, Threat (underlying quality, Ø§Ø¶Ø§ÙØ© Ù…Ù† Ø¹Ù†Ø¯ÙŠ)
        ict = float(p.get("ict_index", 0.0) or 0.0)
        
        # xGI/90: Expected Goal Involvements per 90 (Ø§Ø¶Ø§ÙØ© Ù…ØªÙ‚Ø¯Ù…Ø© Ù…Ù† Ø¹Ù†Ø¯ÙŠ)
        xgi_90 = float(p.get("expected_goal_involvements_per_90", 0.0) or 0.0)
        
        # Team Level (Ù…Ø³ØªÙˆÙŠ Ø§Ù„ÙØ±Ù‚Ø©)
        t_strength = team_strength.get(team_id) or 3
        strength_multiplier = t_strength / 3.0 # Normalize around 3
        
        # 3. Composite xP Formula calculation
        # We heavily weight ep_next (since it has fixture logic), but add our own adjustments
        # for form, underlying stats, and team strength.
        custom_xp = (ep_next * 0.5) + (form * 0.2) + ((ict / 10.0) * 0.15) + (xgi_90 * 1.5 * 0.15)
        
        # Adjust slightly for overall team strength
        custom_xp *= strength_multiplier
        
        # Filter out players who barely play to prevent algorithm gaming
        if minutes > 90 or form > 2.0 or ep_next > 2.0:
            candidates.append(PlayerOption(
                player_id=pid, name=name, position=pos, club_id=club, 
                price=price, expected_points=round(custom_xp, 2), 
                is_nailed=(minutes > 500 or form > 2.0)
            ))

    optimizer = FPLOptimizer()
    squad = optimizer.optimize(candidates)
    if not squad: return html.Div("Could not form 15-man squad. Check budget or constraints.")

    sq = squad.summary()
    
    # Render rich UI with the new stats
    players_list = [html.Tr([
        html.Td(p["position"], className="fw-bold"), 
        html.Td(p["name"]), 
        html.Td(p["club"]), 
        html.Td(f"Â£{p['price']}m"), 
        html.Td(f"{p['xp']} pts", className="text-success fw-bold")
    ]) for p in sq["squad"]]
    
    return html.Div([
        html.H4(f"ðŸ¤– AI Optimized Squad — GW{gw}", className="mb-3"),
        dbc.Alert("Optimization based on: Fixture Difficulty + Player Form + Team Strength + Expected Goal Involvements (xGI) + ICT Index.", color="info", className="small"),
        dbc.Table([
            html.Thead(html.Tr([html.Th("Pos"), html.Th("Player"), html.Th("Club"), html.Th("Price"), html.Th("Projected xP")])), 
            html.Tbody(players_list)
        ], bordered=True, hover=True, size="sm")
    ])

@callback(
    Output("monitoring-output", "children"),
    Input("refresh-alerts-btn", "n_clicks")
)
def update_monitoring(n_clicks):
    active_alerts = alert_manager.get_recent_alerts()
    if not active_alerts: return dbc.Alert("System healthy.", color="success")
    return html.Div([dbc.Alert("Active Alerts:", color="warning"), html.Ul([html.Li(f"[{a.severity}] {a.message}") for a in active_alerts])])

@callback(
    Output("today-matches-output", "children"),
    Input("live-refresh-interval", "n_intervals")
)
def update_today_matches(n):
    from data.providers.fpl_provider import fpl_provider
    from data.elo_store import elo_store
    
    fixtures = fpl_provider.get_fixtures()
    teams = fpl_provider.get_teams()
    team_map = {t["id"]: t["name"] for t in teams}
    
    today = datetime.now(timezone.utc).date()
    
    today_fixtures = []
    for f in fixtures:
        if not f.get("kickoff_time"): continue
        kickoff = datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))
        is_today = kickoff.date() == today
        is_live = f.get("started") and not f.get("finished")
        if is_today or is_live:
            today_fixtures.append((f, kickoff))
            
    if not today_fixtures:
        return html.Div(html.H4("No Premier League matches scheduled for today.", className="text-center text-muted mt-5"))
        
    def pct_str(v): return f"{v * 100:.1f}%"
    
    cards = []
    for f, kickoff in sorted(today_fixtures, key=lambda x: x[1]):
        h_id = f["team_h"]
        a_id = f["team_a"]
        h_name = team_map.get(h_id, "Unknown")
        a_name = team_map.get(a_id, "Unknown")
        
        try:
            elo_ph, elo_pd, elo_pa = elo_store.get_home_win_prob(h_name, a_name)
        except Exception:
            elo_ph, elo_pd, elo_pa = 0.45, 0.28, 0.27
            
        is_live = f.get("started") and not f.get("finished")
        is_finished = f.get("finished")
        
        if is_live:
            badge_color, badge_text = "danger", f"LIVE {f.get('minutes', 0)}'"
            score_text = f"{f.get('team_h_score', 0)} - {f.get('team_a_score', 0)}"
        elif is_finished:
            badge_color, badge_text = "secondary", "FT"
            score_text = f"{f.get('team_h_score', 0)} - {f.get('team_a_score', 0)}"
        else:
            badge_color, badge_text = "primary", kickoff.strftime("%H:%M UTC")
            score_text = "vs"
            
        status_badge = dbc.Badge(badge_text, color=badge_color, className="float-end fs-6")
        
        bar = html.Div([
            html.Div(pct_str(elo_ph), style={"width": pct_str(elo_ph), "height": "26px", "lineHeight": "26px", "fontSize": "12px", "textAlign": "center", "fontWeight": "bold", "background": "#198754", "color": "white"}),
            html.Div(pct_str(elo_pd), style={"width": pct_str(elo_pd), "height": "26px", "lineHeight": "26px", "fontSize": "12px", "textAlign": "center", "fontWeight": "bold", "background": "#ffc107", "color": "#212529"}),
            html.Div(pct_str(elo_pa), style={"width": pct_str(elo_pa), "height": "26px", "lineHeight": "26px", "fontSize": "12px", "textAlign": "center", "fontWeight": "bold", "background": "#dc3545", "color": "white"}),
        ], style={"display": "flex", "width": "100%", "borderRadius": "6px", "overflow": "hidden"})
        
        cards.append(dbc.Card(dbc.CardBody([
            status_badge,
            html.H5(f"{h_name}  {score_text}  {a_name}", className="mb-2 text-center fw-bold"),
            bar,
            html.Div([
                html.Small(f"Home {pct_str(elo_ph)}", className="text-success fw-bold"),
                html.Small(f"Draw {pct_str(elo_pd)}", className="text-warning fw-bold"),
                html.Small(f"Away {pct_str(elo_pa)}", className="text-danger fw-bold"),
            ], className="d-flex justify-content-between mt-1 small")
        ]), className="mb-3 shadow-sm"))
        
    return html.Div([html.H4(f"Matches for {today.strftime('%A, %b %d %Y')} (Premier League)", className="mb-4"), *cards])

# ── NEW CALLBACKS: Injuries, Multi-GW, Tab routing for new tabs ──────────────

from dashboard.layout import injuries_layout, multi_gw_fantasy_layout

@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab_content_v2(tab):
    if tab == "tab-today": return today_layout()
    elif tab == "tab-predictions": return predictions_layout()
    elif tab == "tab-fantasy": return multi_gw_fantasy_layout()
    elif tab == "tab-injuries": return injuries_layout()
    elif tab == "tab-monitoring": return monitoring_layout()
    return html.Div("Select a tab above.")

@callback(
    Output("injuries-output", "children"),
    Input("refresh-injuries-btn", "n_clicks"),
    prevent_initial_call=False
)
def update_injuries(n_clicks):
    from agents.availability_agent import availability_agent
    from data.providers.fpl_provider import fpl_provider

    try:
        availability_agent.invalidate_cache()
        injured = availability_agent.get_all_injuries()
        teams_data = fpl_provider.get_teams()
        team_map = {t["id"]: t["name"] for t in teams_data}

        if not injured:
            return dbc.Alert("No injury alerts — all players available!", color="success")

        status_colors = {
            "INJURED": "danger",
            "DOUBTFUL": "warning",
            "SUSPENDED": "danger",
            "UNAVAILABLE": "secondary",
        }

        rows = []
        for p in sorted(injured, key=lambda x: x.status):
            color = status_colors.get(p.status, "secondary")
            rows.append(html.Tr([
                html.Td(dbc.Badge(p.status, color=color)),
                html.Td(p.name, className="fw-bold"),
                html.Td(f"{p.chance_next_round}%" if p.chance_next_round is not None else "—"),
                html.Td(p.news[:80] if p.news else "—", className="text-muted small"),
            ]))

        return html.Div([
            dbc.Alert(f"Found {len(injured)} non-available players across all PL clubs.", color="warning", className="small"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Status"), html.Th("Player"), html.Th("Chance Next GW"), html.Th("News")])),
                html.Tbody(rows)
            ], bordered=True, hover=True, size="sm", responsive=True)
        ])
    except Exception as e:
        return dbc.Alert(f"Error fetching injury data: {e}", color="danger")


@callback(
    Output("mgw-output", "children"),
    Input("mgw-btn", "n_clicks"),
    State("mgw-gws-slider", "value"),
    prevent_initial_call=True
)
def update_multi_gw(n_clicks, n_gws):
    from fantasy.multi_gw_engine import multi_gw_engine

    try:
        multi_gw_engine._bootstrap = None
        multi_gw_engine._fixtures = None
        
        players = multi_gw_engine.analyze(n_gws=n_gws or 5)[:30]
        captain_picks = sorted(players, key=lambda x: x.captain_potential, reverse=True)[:5]

        fdr_badge_color = lambda fdr: "success" if fdr < 2.5 else ("warning" if fdr < 3.5 else "danger")

        def player_row(p):
            fixture_badges = [
                dbc.Badge(f"GW{f['gw']} {f['opponent']} ({'H' if f['home'] else 'A'}) FDR{f['fdr']}",
                          color=fdr_badge_color(f['fdr']), className="me-1", pill=True)
                for f in p.fixtures
            ]
            return html.Tr([
                html.Td(p.position, className="fw-bold"),
                html.Td(p.name),
                html.Td(p.team_name),
                html.Td(f"{p.price:.1f}m"),
                html.Td(f"{p.form:.1f}"),
                html.Td(f"{p.multi_gw_score:.2f}", className="text-success fw-bold"),
                html.Td(dbc.Badge(p.risk, color="danger" if p.risk=="HIGH" else ("success" if p.risk=="LOW" else "warning"))),
                html.Td(html.Div(fixture_badges)),
            ])

        captain_rows = [html.Tr([
            html.Td(f"#{i+1}"),
            html.Td(p.name, className="fw-bold"),
            html.Td(p.team_name),
            html.Td(f"{p.captain_potential:.2f}", className="text-warning fw-bold"),
            html.Td(f"Form: {p.form:.1f}"),
        ]) for i, p in enumerate(captain_picks)]

        return html.Div([
            html.H4(f"Multi-GW Analysis — Next {n_gws} Gameweeks", className="mb-3"),
            dbc.Alert("Players ranked by composite score: Fixture difficulty + Form + xGI + ICT + Upcoming FDR", color="info", className="small mb-3"),

            html.H5("â­ Captain Picks", className="mt-3 text-warning"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Rank"), html.Th("Player"), html.Th("Club"), html.Th("Captain Score"), html.Th("Form")])),
                html.Tbody(captain_rows)
            ], bordered=True, size="sm", color="warning", className="mb-4"),

            html.H5("ðŸ“‹ Top Transfer Targets", className="mt-3"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Pos"), html.Th("Player"), html.Th("Club"), html.Th("Price"), html.Th("Form"), html.Th("Score"), html.Th("Risk"), html.Th(f"Next {n_gws} Fixtures")])),
                html.Tbody([player_row(p) for p in players])
            ], bordered=True, hover=True, size="sm", responsive=True),
        ])
    except Exception as e:
        import traceback
        return dbc.Alert(f"Error: {e}\n{traceback.format_exc()}", color="danger")





