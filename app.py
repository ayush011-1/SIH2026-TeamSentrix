#!/usr/bin/env python3
"""
Launcher for Sentrix AI Model Testing Platform.
Usage:
    python app.py
    python app.py --port 8080 --host 0.0.0.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai.server import run_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentrix AI Model Testing Web Platform")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
