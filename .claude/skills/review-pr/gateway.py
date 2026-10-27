#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Model gateway for one headless review, started and stopped by gateway.sh.

The agents read untrusted PR text with permissions skipped, so they never get the API key: they
call this gateway, which adds the key to each Messages API request and relays the reply. It
listens only on the address it has at start-up, which gateway.sh puts on the agents' network.

  API_KEY   sent upstream as Ocp-Apim-Subscription-Key
  API_URL   upstream base URL, https only, e.g. https://llm-api.amd.com/Anthropic
  API_USER  optional `user` header; the AMD LLM gateway requires one with shared keys
"""

import http.client
import http.server
import os
import socket
import ssl
import threading
import urllib.parse

KEY = os.environ["API_KEY"].strip()
USER = os.environ.get("API_USER", "").strip()
UPSTREAM = urllib.parse.urlsplit(os.environ["API_URL"].rstrip("/"))
PORT = 8080
PATHS = {"/v1/messages", "/v1/messages/count_tokens"}
FORWARDED = {"accept", "anthropic-beta", "anthropic-version", "content-type"}
# The response headers the CLI reads; it parses bodies by content type and retries by these hints.
RELAYED = ("content-encoding", "content-type", "request-id", "retry-after", "retry-after-ms", "x-should-retry")
# Longer than the review job's timeout, so a gateway whose job died still exits and is removed.
LIFETIME = 3 * 3600
TLS = ssl.create_default_context()


class Gateway(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        if urllib.parse.urlsplit(self.path).path not in PATHS:
            self.send_error(404)
            return
        if self.headers.get("Content-Length") is None:
            self.send_error(411)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        headers = {k: v for k, v in self.headers.items() if k.lower() in FORWARDED}
        headers["Ocp-Apim-Subscription-Key"] = KEY
        if USER:
            headers["user"] = USER
        upstream = http.client.HTTPSConnection(UPSTREAM.hostname, UPSTREAM.port or 443, timeout=900, context=TLS)
        try:
            upstream.request("POST", UPSTREAM.path + self.path, body, headers)
            reply = upstream.getresponse()
        # ValueError means the HTTP client rejected a header value; its message holds that value, maybe the key.
        except (OSError, http.client.HTTPException, ValueError) as exc:
            upstream.close()
            self.send_error(502, f"upstream request failed: {type(exc).__name__}")
            return
        try:
            self.send_response(reply.status)
            for name in RELAYED:
                value = reply.getheader(name)
                if value is not None:
                    self.send_header(name, value.replace("\r", "").replace("\n", ""))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            logged = reply.status < 400
            # read1 returns what has arrived, so server-sent events reach the agent as they stream.
            while chunk := reply.read1(65536):
                if not logged:
                    text = chunk[:200].decode("utf-8", "replace").replace("\r", " ").replace("\n", " ")
                    self.log_message("upstream %d: %s", reply.status, text)
                    logged = True
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            upstream.close()


def main():
    if UPSTREAM.scheme != "https":
        raise SystemExit("API_URL must be an https URL")
    host = socket.gethostbyname(socket.gethostname())
    server = http.server.ThreadingHTTPServer((host, PORT), Gateway)
    threading.Timer(LIFETIME, os._exit, args=(0,)).start()
    if KEY != os.environ["API_KEY"]:
        print("API_KEY had surrounding whitespace, which was removed", flush=True)
    print(f"listening on {host}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
