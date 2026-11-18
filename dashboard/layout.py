"""
dashboard/layout.py
===================
Defines the UI structure of the Dash application.
"""
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(html.H1("Football Intelligence & Fantasy v3", className="text-center my-4"), width=12)
        ]),
        
        dcc.Tabs(id="tabs", value="tab-today", children=[
            dcc.Tab(label="📅 Today", value="tab-today"),
            dcc.Tab(label="⚽ Match Analysis", value="tab-predictions"),
            dcc.Tab(label="🏆 Multi-GW Fantasy", value="tab-fantasy"),
            dcc.Tab(label="🚑 Injuries & Availability", value="tab-injuries"),
            dcc.Tab(label="📊 Model Monitoring", value="tab-monitoring"),
        ]),
        
        html.Div(id="tab-content", className="mt-4")
    ], className="container-fluid mt-3")

def today_layout():
    return dbc.Card([
        dbc.CardHeader("Today's Matches"),
        dbc.CardBody([
            dcc.Interval(id='live-refresh-interval', interval=60*1000, n_intervals=0),
            dcc.Loading(
                id="loading-today",
                children=[html.Div(id="today-matches-output")]
            )
        ])
    ])

def predictions_layout():
    return dbc.Card([
        dbc.CardHeader("Match Analysis"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Input(id="home-team-input", placeholder="Home Team ID (e.g. ARS)", type="text"), width=4),
                dbc.Col(dbc.Input(id="away-team-input", placeholder="Away Team ID (e.g. TOT)", type="text"), width=4),
                dbc.Col(dbc.Button("Analyze Match", id="analyze-btn", color="primary", className="w-100"), width=4)
            ], className="mb-4"),
            dcc.Loading(
                id="loading-predictions",
                children=[html.Div(id="predictions-output")]
            )
        ])
    ])

def fantasy_layout():
    return dbc.Card([
        dbc.CardHeader("FPL Squad Optimizer"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Input(id="gw-input", placeholder="Target Gameweek (e.g. 1)", type="number", min=1, max=38), width=8),
                dbc.Col(dbc.Button("Optimize Squad", id="optimize-btn", color="success", className="w-100"), width=4)
            ], className="mb-4"),
            dcc.Loading(
                id="loading-fantasy",
                children=[html.Div(id="fantasy-output")]
            )
        ])
    ])

def monitoring_layout():
    return dbc.Card([
        dbc.CardHeader("Model Drift & Quality Alerts"),
        dbc.CardBody([
            dbc.Button("Refresh Alerts", id="refresh-alerts-btn", color="info", className="mb-4"),
            dcc.Loading(
                id="loading-monitoring",
                children=[html.Div(id="monitoring-output")]
            )
        ])
    ])

def injuries_layout():
    return dbc.Card([
        dbc.CardHeader("🚑 Injuries & Player Availability"),
        dbc.CardBody([
            dbc.Button("Refresh Injury List", id="refresh-injuries-btn", color="warning", className="mb-4"),
            dcc.Loading(id="loading-injuries", children=[html.Div(id="injuries-output")])
        ])
    ])

def multi_gw_fantasy_layout():
    return dbc.Card([
        dbc.CardHeader("🏆 Multi-Gameweek Fantasy Intelligence"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Number of GWs to analyze:"),
                    dcc.Slider(id="mgw-gws-slider", min=1, max=10, step=1, value=5,
                               marks={i: str(i) for i in range(1, 11)})
                ], width=8),
                dbc.Col([
                    dbc.Button("Analyze Multi-GW", id="mgw-btn", color="success", className="w-100 mt-4")
                ], width=4),
            ], className="mb-4"),
            dcc.Loading(id="loading-mgw", children=[html.Div(id="mgw-output")])
        ])
    ])
