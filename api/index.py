"""
Vercel Serverless Entry Point for SatAgriSense-AI Flask Backend.
Exports top-level 'app' Flask instance for @vercel/python WSGI runtime.
"""
import sys
import os

# Ensure VERCEL environment flag is set
os.environ["VERCEL"] = "1"

# Directory resolution
cur_dir = os.path.dirname(os.path.abspath(__file__))
api_backend_dir = os.path.join(cur_dir, "backend")
root_dir = os.path.abspath(os.path.join(cur_dir, ".."))
root_backend_dir = os.path.join(root_dir, "backend")

for p in [api_backend_dir, root_backend_dir, cur_dir, root_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from backend.app import app

# Explicit top-level Flask instance assignment for @vercel/python detection
app = app
handler = app
