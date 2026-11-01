"""Thin client for the Otari gateway.

Every model call Recount makes goes through this module, so there is exactly
one place that knows the base URL, the auth header, and how Otari reports
errors, fallbacks and cost.

No third-party dependencies: urllib only.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Note: Otari serves its API under /api/v1, not /v1 (the hosted quickstart
# once said /v1; the gateway answers with a helpful error if you get it wrong).
DEFAULT_BASE_URL = "https://api.otari.ai/api/v1"

# Response headers worth surfacing to the user.
OTARI_HEADERS = (
    "Otari-Model",
    "Otari-Provider",
    "Otari-Fallback",
    "Otari-Attempt-Count",
    "Otari-Request-Id",
    "Otari-Container-Id",
    "Otari-Error-Code",
    "Otari-Response-Duration-Ms",
)


class OtariError(RuntimeError):
    """Any non-2xx answer from Otari, with its stable error code."""

    def __init__(self, status: int, code: str | None, detail: str, hint: str = ""):
        self.status = status
        self.code = code
        self.detail = detail
        self.hint = hint
        msg = f"Otari answered {status}"
        if code:
            msg += f" ({code})"
        msg += f": {detail}"
        if hint:
            msg += f"\n  -> {hint}"
        super().__init__(msg)


class BudgetExceeded(OtariError):
    """A budget, member cap, or workspace funds stopped the request."""


class RateLimited(OtariError):
    pass


class RoutingRejected(OtariError):
    """Every candidate in a routing policy refused the request."""


BUDGET_HINT = (
    "A spend cap stopped this request before any model ran, so nothing was charged. "
    "Daily Otari budgets reset at 00:00 UTC (05:30 IST). Meanwhile run "
    "`recount --replay` to see the recorded audit, or raise the cap in the Otari dashboard."
)


def classify_error(status: int, body: bytes | str | None) -> OtariError:
    """Turn an HTTP error from Otari into the right exception with a human hint."""
    text = body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else (body or "")
    code: str | None = None
    detail = text.strip() or "no body"
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            code = data.get("code")
            if not code and isinstance(data.get("error"), dict):
                code = data["error"].get("code")
            d = data.get("detail") or data.get("message") or data.get("error")
            if isinstance(d, (dict, list)):
                d = json.dumps(d)
            if d:
                detail = str(d)
    except (ValueError, TypeError):
        pass

    low = f"{code or ''} {detail}".lower()
    if status == 402 or "budget" in low or "insufficient_funds" in low or "spend cap" in low:
        return BudgetExceeded(status, code, detail, BUDGET_HINT)
    if status == 429:
        return RateLimited(status, code, detail, "Rate limited by Otari or the provider; wait a minute, or use --replay.")
    if code == "all_candidates_rejected":
        return RoutingRejected(
            status, code, detail,
            "No model in the routing policy accepted the request. Check the policy name and the request parameters.",
        )
    if status in (401, 403):
        return OtariError(status, code, detail, "Check OTARI_API_KEY (a workspace key from the Otari dashboard).")
    return OtariError(status, code, detail)


@dataclass
class OtariResult:
    """A parsed Otari response plus the routing metadata from its headers."""

    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        # cost_usd includes gateway tool calls (search, sandbox); estimated_cost does not.
        try:
            return float((self.body.get("usage") or {}).get("cost_usd") or 0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def served_by(self) -> str:
        return self.headers.get("Otari-Model") or str(self.body.get("model", "?"))

    @property
    def fell_back(self) -> bool:
        return str(self.headers.get("Otari-Fallback", "false")).lower() == "true"

    def to_json(self) -> dict[str, Any]:
        return {"headers": self.headers, "response": self.body}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "OtariResult":
        return cls(body=data.get("response") or {}, headers=data.get("headers") or {})


class OtariClient:
    """Calls Otari's OpenAI-compatible endpoints."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("OTARI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OTARI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        if self.base_url.endswith("otari.ai/v1"):
            # Common mistake; fix it rather than burn a request on a 404.
            self.base_url = self.base_url[: -len("/v1")] + "/api/v1"

    def _post(self, path: str, payload: dict[str, Any], tags: dict[str, str] | None = None) -> OtariResult:
        if not self.api_key:
            raise OtariError(0, "no_api_key", "OTARI_API_KEY is not set",
                             "Export OTARI_API_KEY, or run with --replay to use recorded responses.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "recount/0.1",
        }
        if tags:
            headers["Otari-Tags"] = ",".join(f"{k}={v}" for k, v in tags.items())
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                hdrs = {h: resp.headers.get(h) for h in OTARI_HEADERS if resp.headers.get(h)}
        except urllib.error.HTTPError as e:
            raise classify_error(e.code, e.read()) from None
        except urllib.error.URLError as e:
            raise OtariError(0, "network", str(e.reason), "Could not reach Otari; check your network or use --replay.") from None
        return OtariResult(body=json.loads(raw or b"{}"), headers=hdrs)

    def chat(self, model: str, messages: list[dict[str, Any]], **extra: Any) -> OtariResult:
        """POST /chat/completions. `extra` may carry Otari fields such as
        mcp_server_ids, max_tool_iterations, or tools=[{"type": "otari_web_search"}]."""
        tags = extra.pop("tags", None)
        return self._post("/chat/completions", {"model": model, "messages": messages, **extra}, tags)

    def responses(self, model: str, input: Any, **extra: Any) -> OtariResult:
        """POST /responses (used for code_interpreter, which returns the executed code)."""
        tags = extra.pop("tags", None)
        return self._post("/responses", {"model": model, "input": input, **extra}, tags)
