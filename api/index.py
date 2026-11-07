"""
Vercel Serverless Entry Point for SatAgriSense-AI Flask Backend.
Imports the Flask app from the backend directory and exposes it as 'app' for Vercel WSGI.
"""
import sys
import os

# Robust directory resolution for Vercel Lambda environment
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(cur_dir, ".."))
backend_dir = os.path.join(root_dir, "backend")

for p in [backend_dir, root_dir, cur_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Ensure VERCEL environment flag is set so /tmp is used for SQLite database
if "VERCEL" not in os.environ and "AWS_LAMBDA_FUNCTION_NAME" not in os.environ:
    os.environ["VERCEL"] = "1"

try:
    from app import app
except ImportError:
    try:
        from backend.app import app
    except Exception as e:
        from flask import Flask, jsonify
        from flask_cors import CORS
        app = Flask(__name__)
        CORS(app)
        @app.route("/api/<path:subpath>", methods=["GET", "POST"])
        @app.route("/<path:subpath>", methods=["GET", "POST"])
        def fallback_route(subpath):
            return jsonify({"status": "FALLBACK_ONLINE", "message": "Vercel backend active", "detail": str(e), "path": subpath})

# WSGI handler for Vercel
handler = app
