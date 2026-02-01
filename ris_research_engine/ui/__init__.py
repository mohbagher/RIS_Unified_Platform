"""UI layer for the RIS Auto-Research Engine.

This module provides multiple interfaces for interacting with the RIS research engine:
- RISEngine: Simple API for Jupyter notebooks
- RISDashboard: Interactive ipywidgets dashboard
- CLI: Command-line interface
"""

from .jupyter_minimal import RISEngine
from .jupyter_dashboard import RISDashboard
from .cli import main as cli_main

__all__ = [
    'RISEngine',
    'RISDashboard',
    'cli_main',
]
