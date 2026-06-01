"""agent/health.py — health + metrics HTTP server."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional

from loguru import logger


class _Handler(BaseHTTPRequestHandler):
    status_fn: Callable[[], dict] = lambda: {"status": "ok"}
    metrics_fn: Callable[[], dict] = lambda: {}

    def log_message(self, format, *args):  # noqa: A002
        pass

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/healthz", "/"):
            self._send_json(200, _Handler.status_fn())
        elif self.path == "/metrics":
            self._send_json(200, _Handler.metrics_fn())
        else:
            self.send_response(404)
            self.end_headers()


def start_health_server(
    port: int,
    status_fn: Callable[[], dict],
    metrics_fn: Optional[Callable[[], dict]] = None,
) -> Optional[HTTPServer]:
    if port <= 0:
        return None
    _Handler.status_fn = status_fn
    _Handler.metrics_fn = metrics_fn or (lambda: {})
    server = HTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health server listening on :{port} (/health, /metrics)")
    return server
