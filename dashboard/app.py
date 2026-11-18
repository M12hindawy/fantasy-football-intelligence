"""
dashboard/app.py
================
Main Dash application initialization.
Run with: python -m dashboard.app
"""
import dash
import dash_bootstrap_components as dbc

# Initialize the Dash app with a Bootstrap theme for quick styling
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Football Intelligence v3"
)
server = app.server

# Import layout and callbacks to register them
from dashboard.layout import create_layout
app.layout = create_layout()

import dashboard.callbacks  # noqa: F401

if __name__ == "__main__":
    app.run(debug=True, port=8050)
