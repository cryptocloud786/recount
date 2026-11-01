"""Record every Otari response to disk, and replay them offline.

Live runs save each step's response (body + Otari routing headers, never the
request's API key) to samples/<run>/<step>.json. `--replay` reads them back,
so the whole pipeline, the CLI and the web UI work with no key and no spend.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .otari import OtariClient, OtariError, OtariResult

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
DEFAULT_SAMPLE = "india-population"


def slugify(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:limit].rstrip("-")) or "claim"


class ReplayMissing(OtariError):
    def __init__(self, path: Path):
        super().__init__(0, "replay_missing", f"no recorded response at {path}",
                         "Record it first with a live run, or pick a sample with --sample.")


class Gateway:
    """What the pipeline talks to: a live Otari client that records, or a replayer."""

    def __init__(self, mode: str = "live", run_dir: Path | None = None, client: OtariClient | None = None,
                 resume: bool = False):
        if mode not in ("live", "replay"):
            raise ValueError(mode)
        self.mode = mode
        self.resume = resume  # live mode: reuse steps already recorded in run_dir, pay only for the rest
        self.run_dir = Path(run_dir) if run_dir else SAMPLES_DIR / DEFAULT_SAMPLE
        self.client = client if client is not None else (OtariClient() if mode == "live" else None)
        self.spent_usd = 0.0
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def replay(cls, sample: str = DEFAULT_SAMPLE) -> "Gateway":
        p = Path(sample)
        return cls("replay", p if p.is_dir() else SAMPLES_DIR / sample)

    def _path(self, step: str) -> Path:
        return self.run_dir / f"{step}.json"

    def call(self, step: str, kind: str, request: dict[str, Any]) -> OtariResult:
        """kind is 'chat' or 'responses'. request holds model + body fields."""
        path = self._path(step)
        if self.mode == "replay" or (self.resume and path.exists()):
            if not path.exists():
                raise ReplayMissing(path)
            data = json.loads(path.read_text())
            result = OtariResult.from_json(data)
        else:
            fn: Callable[..., OtariResult] = getattr(self.client, kind)
            req = dict(request)
            model = req.pop("model")
            first = req.pop("messages" if kind == "chat" else "input")
            result = fn(model, first, **req)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            safe_request = {k: v for k, v in request.items() if k.lower() not in ("api_key", "authorization")}
            path.write_text(json.dumps({"step": step, "kind": kind, "request": safe_request, **result.to_json()},
                                       indent=2, ensure_ascii=False))
        self.spent_usd += result.cost_usd
        self.calls.append({"step": step, "served_by": result.served_by, "fallback": result.fell_back,
                           "cost_usd": result.cost_usd, "request_id": result.headers.get("Otari-Request-Id")})
        return result
