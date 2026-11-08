"""
Vercel Serverless Entry Point for SatAgriSense-AI Flask Backend.
Imports the Flask app from the backend directory and exposes it as 'app' and 'handler' for Vercel WSGI.
"""
import sys
import os
import shutil

# Robust directory resolution for Vercel Lambda environment
cur_dir = os.path.dirname(os.path.abspath(__file__))
api_backend_dir = os.path.join(cur_dir, "backend")
root_dir = os.path.abspath(os.path.join(cur_dir, ".."))
root_backend_dir = os.path.join(root_dir, "backend")

# Ensure all backend resolution paths are top priority in sys.path
for p in [api_backend_dir, root_backend_dir, cur_dir, root_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Ensure VERCEL environment flag is set so /tmp is used for SQLite database
if "VERCEL" not in os.environ and "AWS_LAMBDA_FUNCTION_NAME" not in os.environ:
    os.environ["VERCEL"] = "1"

# On Vercel /tmp is writable; seed initial sqlite db if not present
try:
    tmp_db = "/tmp/agri_sentinel.db"
    if not os.path.exists(tmp_db):
        for candidate_db in [
            os.path.join(api_backend_dir, "agri_sentinel.db"),
            os.path.join(root_backend_dir, "agri_sentinel.db")
        ]:
            if os.path.exists(candidate_db):
                shutil.copy2(candidate_db, tmp_db)
                break
except Exception as e:
    pass

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
            return jsonify({
                "status": "FALLBACK_ONLINE",
                "message": "Vercel backend active",
                "detail": str(e),
                "path": subpath
            })

# Both 'app' and 'handler' are supported by Vercel WSGI
handler = app
