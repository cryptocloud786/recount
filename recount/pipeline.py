"""The four-step Recount audit.

1. extract   - openai/gpt-oss-20b pulls the figures and the arithmetic out of the claim.
2. source    - openai/gpt-oss-120b looks up official values with Otari's MCP
               servers (World Bank, Wikipedia) and Otari web search.
3. recompute - openai/gpt-oss-120b re-does the arithmetic in Otari's code
               sandbox (Responses API code_interpreter); we show the code that ran.
4. verdict   - the `recount-auditor` routing policy answers 3 times (n=3);
               the majority vote is the verdict.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .otari import OtariError
from .replay import Gateway

EXTRACT_MODEL = "openai/gpt-oss-20b"
SOURCE_MODEL = "openai/gpt-oss-120b"
RECOMPUTE_MODEL = "openai/gpt-oss-120b"
VERDICT_POLICY = "recount-auditor"

# MCP servers stored in the Otari workspace (Tools > MCP servers).
MCP_WORLDBANK = "242fadad-549d-4da2-97e3-17756f9b618f"
MCP_WIKIPEDIA = "62fb6a2c-17f4-4e32-8365-d045bcbd913b"

VERDICTS = ("ACCURATE", "MISLEADING", "FALSE", "UNVERIFIABLE")


def parse_json(text: str) -> Any:
    """Pull the first JSON object out of model text (tolerates ``` fences and chatter)."""
    if not text:
        raise ValueError("empty model output")
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in: {text[:120]!r}")
    depth = 0
    in_str = esc = False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object")


def chat_text(body: dict[str, Any], index: int = 0) -> str:
    try:
        return body["choices"][index]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def code_runs(body: dict[str, Any]) -> list[dict[str, str]]:
    """The code_interpreter_call items of a Responses answer: what really ran in the sandbox."""
    runs = []
    for item in body.get("output") or []:
        if item.get("type") == "code_interpreter_call":
            logs = "".join(o.get("logs", "") for o in item.get("outputs") or [] if isinstance(o, dict))
            runs.append({"code": item.get("code", ""), "output": logs, "status": item.get("status", ""),
                         "container_id": item.get("container_id", "")})
    return runs


def responses_text(body: dict[str, Any]) -> str:
    parts = []
    for item in body.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "\n".join(parts)


def parse_vote(text: str) -> str:
    up = (text or "").upper()
    m = re.search(r"VERDICT\s*[:=\-]\s*\**\s*(ACCURATE|MISLEADING|FALSE|UNVERIFIABLE)", up)
    if m:
        return m.group(1)
    for v in VERDICTS:
        if re.search(rf"\b{v}\b", up):
            return v
    return "UNVERIFIABLE"


def majority(votes: list[str]) -> tuple[str, int]:
    if not votes:
        return "UNVERIFIABLE", 0
    counts = Counter(votes)
    top, n = counts.most_common(1)[0]
    # A tie with no majority is not a verdict.
    if n * 2 <= len(votes) and len(counts) > 1:
        return "UNVERIFIABLE", n
    return top, n


@dataclass
class StepResult:
    name: str
    ok: bool
    model: str = ""
    data: Any = None
    error: str = ""
    cost_usd: float = 0.0
    fallback: bool = False
    raw_text: str = ""


@dataclass
class Audit:
    claim: str
    steps: list[StepResult] = field(default_factory=list)
    verdict: str = "UNVERIFIABLE"
    agreement: str = ""
    spent_usd: float = 0.0
    mode: str = "live"
    stopped: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim, "verdict": self.verdict, "agreement": self.agreement,
            "spent_usd": round(self.spent_usd, 6), "mode": self.mode, "stopped": self.stopped,
            "steps": [s.__dict__ for s in self.steps],
        }


class Recount:
    def __init__(self, gateway: Gateway, max_spend_usd: float = 0.10, search_fallback: bool = True):
        self.gw = gateway
        self.max_spend = max_spend_usd
        self.search_fallback = search_fallback

    # -- steps ---------------------------------------------------------------
    def extract(self, claim: str) -> StepResult:
        prompt = (
            "Extract every number from this claim and the arithmetic it implies. Reply with JSON only:\n"
            '{"entity": str, "indicator": str, "figures": [{"label": str, "value": number, "unit": str, "year": int|null}], '
            '"derived": {"label": str, "value": number, "unit": str} | null, "formula": str}\n\n'
            f"Claim: {claim}"
        )
        r = self.gw.call("step1_extract", "chat", {
            "model": EXTRACT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "low", "max_tokens": 800,
            "tags": {"app": "recount", "step": "extract"},
        })
        text = chat_text(r.body)
        return StepResult("extract", True, r.served_by, parse_json(text), cost_usd=r.cost_usd, raw_text=text)

    def source(self, claim: str, facts: dict[str, Any]) -> StepResult:
        prompt = (
            "You audit numeric claims against official statistics. Use the World Bank MCP tools first "
            "(pass countries as a JSON list of ISO2 codes, e.g. [\"IN\"] for India, never a bare string; "
            "indicator codes like SP.POP.TOTL), then web search only if needed. "
            "Find the official value for every figure below. Reply with JSON only:\n"
            '{"official": [{"label": str, "value": number, "year": int, "source": str, "url": str}], "notes": str}\n'
            "If you could not retrieve a value from a tool, set value to null and say so in notes. Never guess.\n\n"
            f"Claim: {claim}\nFigures: {json.dumps(facts.get('figures', []))}"
        )
        r = self.gw.call("step2_source", "chat", {
            "model": SOURCE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "mcp_server_ids": [MCP_WORLDBANK, MCP_WIKIPEDIA],
            "tools": [{"type": "otari_web_search", "max_uses": 2}],
            "max_tool_iterations": 6, "max_tokens": 1500,
            "tags": {"app": "recount", "step": "source"},
        })
        text = chat_text(r.body)
        data = parse_json(text)
        cost = r.cost_usd
        if self.search_fallback and not any(o.get("value") is not None for o in data.get("official") or []):
            fb = self._search_fallback(claim, facts)
            if fb is not None:
                fb_data, fb_res = fb
                cost += fb_res.cost_usd
                data["notes"] = (data.get("notes") or "") + " | MCP gave no values; web-search fallback used."
                data["official"] = fb_data.get("official") or data.get("official")
                r.body.setdefault("otari_tool_calls", []).extend(fb_res.body.get("otari_tool_calls") or [])
        # Otari reports its web searches (query + sources) on Chat Completions.
        data["web_searches"] = [
            {"query": c.get("query"), "status": c.get("status"),
             "sources": [s.get("url") for s in c.get("sources") or []][:5]}
            for c in r.body.get("otari_tool_calls") or [] if c.get("type") == "web_search"
        ]
        # Flag any URL the model cites that no search actually returned.
        seen = {u for w in data["web_searches"] for u in w["sources"]}
        for o in data.get("official") or []:
            o["url_seen_in_search"] = bool(o.get("url")) and o.get("url") in seen
        return StepResult("source", True, r.served_by, data, cost_usd=cost, raw_text=text)

    def _search_fallback(self, claim: str, facts: dict[str, Any]):
        """Second chance with Otari web search only, when the MCP route returned nothing."""
        if self.gw.mode == "replay" and not (self.gw.run_dir / "step2b_search.json").exists():
            return None
        prompt = (
            "Search the web for the official value of each figure (prefer data.worldbank.org, government "
            "statistics offices). Reply with JSON only: "
            '{"official": [{"label": str, "value": number|null, "year": int, "source": str, "url": str}]}. '
            "Only report a value you read in a search result, and cite that result's URL. Never guess.\n"
            f"Claim: {claim}\nFigures: {json.dumps(facts.get('figures', []))}"
        )
        r = self.gw.call("step2b_search", "chat", {
            "model": SOURCE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "otari_web_search", "max_uses": 3}],
            "max_tool_iterations": 4, "max_tokens": 1200,
            "tags": {"app": "recount", "step": "source-fallback"},
        })
        return parse_json(chat_text(r.body)), r

    def recompute(self, claim: str, facts: dict[str, Any], official: dict[str, Any]) -> StepResult:
        prompt = (
            "Use the python tool for every calculation; do not compute in your head.\n"
            f"Claim: {claim}\nClaimed figures: {json.dumps(facts)}\n"
            f"Official figures: {json.dumps(official.get('official', []))}\n"
            "1) Recompute the claim's derived number from the CLAIMED figures. "
            "2) Recompute it from the OFFICIAL figures. 3) Print both and the percentage-point gap. "
            "Then answer in one short paragraph."
        )
        r = self.gw.call("step3_recompute", "responses", {
            "model": RECOMPUTE_MODEL,
            "input": prompt,
            "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
            "max_output_tokens": 1200,
            "tags": {"app": "recount", "step": "recompute"},
        })
        runs = code_runs(r.body)
        data = {"runs": runs, "summary": responses_text(r.body), "sandbox_verified": bool(runs)}
        if not runs:
            # A model can answer with plausible "code" it never ran. Without a
            # code_interpreter_call item from the gateway, we do not trust the number.
            data["warning"] = "No sandbox run was reported; recomputed numbers are not trusted."
        return StepResult("recompute", bool(runs), r.served_by, data, cost_usd=r.cost_usd,
                          raw_text=data["summary"])

    def verdict(self, claim: str, facts: Any, official: Any, recompute: Any) -> StepResult:
        evidence = {
            "claimed": facts,
            "official": (official or {}).get("official"),
            "sandbox": [{"code": x["code"], "output": x["output"]} for x in (recompute or {}).get("runs", [])],
            "sandbox_verified": (recompute or {}).get("sandbox_verified", False),
        }
        prompt = (
            "You are a strict fact-check auditor. Using ONLY the evidence, judge the claim.\n"
            f"Claim: {claim}\nEvidence: {json.dumps(evidence)[:6000]}\n"
            "Answer in one line exactly: VERDICT: <ACCURATE|MISLEADING|FALSE|UNVERIFIABLE> - <reason, max 25 words>"
        )
        r = self.gw.call("step4_verdict", "chat", {
            "model": VERDICT_POLICY,  # an Otari routing policy, not a model
            "messages": [{"role": "user", "content": prompt}],
            "n": 3, "max_tokens": 300, "temperature": 0.7,
            "tags": {"app": "recount", "step": "verdict"},
        })
        texts = [chat_text(r.body, i) for i in range(len(r.body.get("choices") or []))]
        votes = [parse_vote(t) for t in texts]
        top, n = majority(votes)
        data = {"votes": votes, "answers": texts, "verdict": top, "agreement": f"{n}/{len(votes)}",
                "served_by": r.served_by, "fallback": r.fell_back,
                "attempts": r.headers.get("Otari-Attempt-Count")}
        return StepResult("verdict", True, r.served_by, data, cost_usd=r.cost_usd, fallback=r.fell_back,
                          raw_text="\n".join(texts))

    # -- orchestration ---------------------------------------------------------
    def run(self, claim: str, on_step=None, max_steps: int = 4) -> Audit:
        audit = Audit(claim=claim, mode=self.gw.mode)
        facts = official = recomputed = None
        plan = [
            ("extract", lambda: self.extract(claim)),
            ("source", lambda: self.source(claim, facts or {})),
            ("recompute", lambda: self.recompute(claim, facts or {}, official or {})),
            ("verdict", lambda: self.verdict(claim, facts, official, recomputed)),
        ]
        for name, fn in plan[:max_steps]:
            if self.gw.mode == "live" and self.gw.spent_usd >= self.max_spend:
                audit.stopped = (f"Stopped before '{name}': this run already spent ${self.gw.spent_usd:.4f}, "
                                 f"at or over --max-spend ${self.max_spend:.2f}.")
                break
            try:
                step = fn()
            except OtariError as e:
                step = StepResult(name, False, error=str(e))
                audit.steps.append(step)
                audit.stopped = str(e)
                if on_step:
                    on_step(step)
                break
            except ValueError as e:
                step = StepResult(name, False, error=f"could not parse model output: {e}")
                audit.steps.append(step)
                audit.stopped = step.error
                if on_step:
                    on_step(step)
                break
            audit.steps.append(step)
            if on_step:
                on_step(step)
            if name == "extract":
                facts = step.data
            elif name == "source":
                official = step.data
            elif name == "recompute":
                recomputed = step.data
            elif name == "verdict":
                audit.verdict = step.data["verdict"]
                audit.agreement = step.data["agreement"]
        audit.spent_usd = self.gw.spent_usd
        return audit
