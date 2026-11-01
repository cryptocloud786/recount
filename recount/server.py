"""A tiny web UI on the standard library: GET / serves web/index.html,
POST /api/audit runs the pipeline (live or replay) and returns JSON."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .pipeline import Recount
from .replay import DEFAULT_SAMPLE, SAMPLES_DIR, Gateway, slugify

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def make_handler(replay: bool, sample: str, max_spend: float = 0.10):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/config":
                gw_dir = SAMPLES_DIR / sample
                claim = (gw_dir / "claim.txt").read_text().strip() if (gw_dir / "claim.txt").exists() else ""
                return self._send(200, json.dumps({"replay": replay, "sample": sample, "claim": claim}).encode(),
                                  "application/json")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/audit":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send(400, b'{"error":"bad json"}', "application/json")
            claim = (req.get("claim") or "").strip()[:1000]
            use_replay = replay or bool(req.get("replay"))
            if use_replay:
                gw = Gateway.replay(sample)
                if not claim and (gw.run_dir / "claim.txt").exists():
                    claim = (gw.run_dir / "claim.txt").read_text().strip()
            else:
                if not claim:
                    return self._send(400, b'{"error":"claim is required"}', "application/json")
                run_dir = SAMPLES_DIR / slugify(claim)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "claim.txt").write_text(claim + "\n")
                gw = Gateway("live", run_dir)
            audit = Recount(gw, max_spend_usd=max_spend).run(claim)
            self._send(200, json.dumps(audit.to_dict(), ensure_ascii=False).encode(), "application/json")

    return Handler


def serve(port: int = 8787, replay: bool = False, sample: str = DEFAULT_SAMPLE) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(replay, sample))
    print(f"Recount UI on http://127.0.0.1:{port}  ({'replay' if replay else 'live via Otari'})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
