"""
SmartStay AI — Streamlit Cloud Production Entrypoint
Serves the complete 3-tab AI Student Accommodation & Roommate Allocation Platform.
"""

import os
import sys

# Ensure root directory is at the top of sys.path for Streamlit Cloud
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import and execute the frontend app module
from frontend.app import *
