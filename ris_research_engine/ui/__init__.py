"""UI layer for the RIS Auto-Research Engine.

This module provides multiple interfaces for interacting with the RIS research engine:
- RISEngine: Simple API for Jupyter notebooks with minimal configuration
- RISDashboard: Interactive ipywidgets dashboard with 5 tabs
- cli: Command-line interface (use ris-cli command or python -m ris_research_engine.ui.cli)
"""

from .jupyter_minimal import RISEngine

try:
    from .jupyter_dashboard import RISDashboard
except ImportError:
    RISDashboard = None

__all__ = ['RISEngine', 'RISDashboard']
