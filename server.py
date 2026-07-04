"""
CaseClose Dashboard Server

A lightweight HTTP server that:
  1. Serves the dashboard UI (dashboard.html)
  2. Exposes /api/cases — returns all case audit logs from output/case_logs/
  3. Exposes /api/tickets — returns raw ticket data from data/tickets.json

This connects the frontend to live agent data instead of hardcoded mockups.

Usage:
    python server.py                  # Start on port 8080
    python server.py --port 3000      # Custom port
"""

import argparse
import json
import sys
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CASE_LOGS_DIR = PROJECT_ROOT / "output" / "case_logs"
DATA_DIR = PROJECT_ROOT / "data"


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler that serves the dashboard and API endpoints."""

    def do_GET(self):
        # API: Return all case audit logs
        if self.path == "/api/cases":
            self._serve_json(self._load_all_cases())
            return

        # API: Return raw tickets
        if self.path == "/api/tickets":
            self._serve_json(self._load_tickets())
            return

        # Serve dashboard.html as the root page
        if self.path == "/" or self.path == "/index.html":
            self.path = "/dashboard.html"

        # Default: serve static files from project root
        super().do_GET()

    def _load_all_cases(self) -> list:
        """Load all JSON files from output/case_logs/."""
        cases = []
        if not CASE_LOGS_DIR.exists():
            return cases

        for file in sorted(CASE_LOGS_DIR.glob("*.json")):
            try:
                with open(file, "r") as f:
                    case = json.load(f)
                    cases.append(case)
            except Exception as e:
                cases.append({"error": str(e), "file": file.name})

        return cases

    def _load_tickets(self) -> list:
        """Load tickets from data/tickets.json."""
        tickets_path = DATA_DIR / "tickets.json"
        if not tickets_path.exists():
            return []
        with open(tickets_path, "r") as f:
            return json.load(f).get("tickets", [])

    def _serve_json(self, data):
        """Send a JSON response with CORS headers."""
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Suppress default request logging for cleaner output."""
        pass


def main():
    parser = argparse.ArgumentParser(description="CaseClose Dashboard Server")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to serve on")
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))  # Serve files relative to project root

    server = HTTPServer(("localhost", args.port), DashboardHandler)

    print(f"\n  CaseClose Dashboard")
    print(f"  ────────────────────────".encode('ascii', 'replace').decode())
    print(f"  Dashboard:   http://localhost:{args.port}")
    print(f"  API Cases:   http://localhost:{args.port}/api/cases")
    print(f"  API Tickets: http://localhost:{args.port}/api/tickets")
    print(f"\n  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
