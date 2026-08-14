#!/usr/bin/env python3
"""A minimal OpenAI-compatible mock upstream for smoke tests.

It returns a fixed chat completion for any request.
It supports streaming responses when the request asks for them.
It uses only the Python standard library.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "127.0.0.1"
PORT = 18099


def stream_payload():
    """Build the SSE body for a streaming completion."""
    chunks = [
        {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "smoke "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock",
            "choices": [
                {"index": 0, "delta": {"content": "response"}, "finish_reason": "stop"}
            ],
        },
        {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    lines = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (lines + "data: [DONE]\n\n").encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except (ValueError, TypeError):
            body = {}

        if body.get("stream"):
            payload = stream_payload()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

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
