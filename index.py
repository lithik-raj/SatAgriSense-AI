"""
Root Entrypoint for SatAgriSense-AI Flask Backend.
Exports top-level 'app' Flask instance for @vercel/python WSGI runtime.
"""
import sys
import os

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.join(_cur_dir, "backend")

for _p in [_backend_dir, _cur_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.app import app

# Explicit top-level Flask instance assignment for @vercel/python detection
app = app
handler = app
