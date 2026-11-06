"""
Vercel Serverless Entry Point for SatAgriSense-AI Flask Backend.
Imports the Flask app from the backend directory and exposes it as 'app' for Vercel.
"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app  # noqa: F401

# Vercel uses 'app' as the WSGI handler
handler = app
