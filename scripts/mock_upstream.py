#!/usr/bin/env python3
"""A minimal OpenAI-compatible mock upstream for smoke tests.

It returns a fixed chat completion for any request.
It uses only the Python standard library.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "127.0.0.1"
PORT = 18099


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "smoke response"},
                    "finish_reason": "stop",
                }
            ],
        }
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        # Keep the mock output quiet.
        pass


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    server.serve_forever()
