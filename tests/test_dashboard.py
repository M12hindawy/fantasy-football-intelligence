"""
tests/test_dashboard.py
=======================
Tests to ensure the dashboard components can be imported and initialized.
"""
from dashboard.app import app
from dashboard.layout import create_layout

def test_dashboard_initializes():
    # Just checking that no import errors or setup errors occur
    layout = create_layout()
    assert layout is not None
    assert app.title == "Football Intelligence v3"
