"""Web server launcher for the Ahmedabad Urban Heat Island Interactive Dashboard.

Serves the standalone dashboard client locally using Python's built-in HTTP server.
No external web frameworks (like Flask or Streamlit) required.
"""
import functools
import http.server
import logging
import os
import socketserver
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dashboard.data_builder import build_dashboard_data

log = logging.getLogger("dashboard.serve")


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler that suppresses noisy request logs while serving assets."""
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
    }

    def log_message(self, format, *args):
        # Only log errors, suppress standard 200 GET logs
        if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            log.warning("%s - %s", self.address_string(), format % args)


def run_dashboard(port: int = 8000, open_browser: bool = True, blocking: bool = True):
    """Ensure dashboard data is built and start local HTTP server."""
    dashboard_dir = os.path.join(config.BASE_DIR, "dashboard")
    js_data = os.path.join(dashboard_dir, "data.js")

    # If data bundle missing, build it
    if not os.path.exists(js_data):
        log.info("Dashboard data bundle missing. Building data bundle now...")
        build_dashboard_data()

    # Change working directory context for SimpleHTTPRequestHandler
    handler_factory = functools.partial(QuietHTTPHandler, directory=dashboard_dir)

    # Search for an available port starting at `port`
    server = None
    actual_port = port
    for p in range(port, port + 20):
        try:
            server = socketserver.TCPServer(("127.0.0.1", p), handler_factory)
            actual_port = p
            break
        except OSError as e:
            if "already in use" in str(e).lower() or e.errno in (48, 98, 10048):
                continue
            raise

    if server is None:
        raise RuntimeError(f"Could not bind dashboard server to any port in range {port}-{port+20}")

    url = f"http://127.0.0.1:{actual_port}/"
    print("=" * 72)
    print("AHMEDABAD URBAN HEAT ISLAND & PINN INTERACTIVE RESEARCH DASHBOARD")
    print("=" * 72)
    print(f" Dashboard URL: {url}")
    print(f" Directory:     {dashboard_dir}")
    print(" Press Ctrl+C in this terminal to stop the dashboard server.")
    print("=" * 72)

    if open_browser:
        # Give server a moment to start before browser opens
        def _open():
            time.sleep(0.6)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    if blocking:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server...")
        finally:
            try:
                server.shutdown()
            except Exception:
                pass
            server.server_close()
            print("Dashboard server stopped.")
    else:
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        return server, actual_port


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch interactive Ahmedabad Urban Heat Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open default web browser")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding dashboard data bundle")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.rebuild:
        build_dashboard_data()

    run_dashboard(port=args.port, open_browser=not args.no_browser, blocking=True)
