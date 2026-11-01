"""Command line: `recount "claim"`, `recount --replay`, `recount serve`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import Recount, StepResult
from .replay import DEFAULT_SAMPLE, SAMPLES_DIR, Gateway, slugify

DEFAULT_CLAIM = (
    "India's population grew from 1.32 billion in 2015 to 1.43 billion in 2023, "
    "an increase of more than 15 percent."
)


def _print_step(step: StepResult) -> None:
    head = f"[{step.name}]"
    if not step.ok and step.error:
        print(f"{head} FAILED\n  {step.error}\n")
        return
    meta = f"model={step.model}  cost=${step.cost_usd:.6f}" + ("  (fallback)" if step.fallback else "")
    print(f"{head} {meta}")
    d = step.data or {}
    if step.name == "extract":
        for f in d.get("figures", []):
            print(f"  - {f.get('label')}: {f.get('value')} {f.get('unit', '')} ({f.get('year')})")
        if d.get("derived"):
            print(f"  derived: {d['derived']}")
    elif step.name == "source":
        for o in d.get("official", []):
            print(f"  - {o.get('label')}: {o.get('value')} ({o.get('year')}) - {o.get('source')} {o.get('url', '')}")
        for s in d.get("web_searches", []):
            print(f"  web search: {s['query']!r} -> {len(s['sources'])} sources")
        if d.get("notes"):
            print(f"  notes: {d['notes']}")
    elif step.name == "recompute":
        for i, run in enumerate(d.get("runs", []), 1):
            print(f"  --- sandbox run {i} (executed in Otari code sandbox) ---")
            for line in run["code"].splitlines():
                print(f"  | {line}")
            print(f"  => {run['output'].strip()}")
        if d.get("warning"):
            print(f"  WARNING: {d['warning']}")
    elif step.name == "verdict":
        for v, a in zip(d.get("votes", []), d.get("answers", [])):
            print(f"  vote {v:<12} {a.strip()[:140]}")
        print(f"  served by {d.get('served_by')}  fallback={d.get('fallback')}  attempts={d.get('attempts')}")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="recount", description="Recompute a numeric claim through the Otari gateway.")
    p.add_argument("claim", nargs="?", help="the claim to audit (default: a demo claim)")
    p.add_argument("--replay", action="store_true", help="use recorded Otari responses from samples/ (no key, no spend)")
    p.add_argument("--sample", default=DEFAULT_SAMPLE, help=f"recorded sample to replay (default {DEFAULT_SAMPLE})")
    p.add_argument("--record-as", help="sample name to save a live run under (default: slug of the claim)")
    p.add_argument("--max-spend", type=float, default=0.10, help="stop starting new steps once a run spent this many USD")
    p.add_argument("--resume", action="store_true", help="live: reuse steps already recorded for this claim, pay only for missing ones")
    p.add_argument("--steps", type=int, default=4, help="run only the first N steps (1-4)")
    p.add_argument("--no-search-fallback", action="store_true", help="do not retry with web search when MCP finds nothing")
    p.add_argument("--json", action="store_true", help="print the audit as JSON")
    p.add_argument("--serve", action="store_true", help="start the web UI instead")
    p.add_argument("--port", type=int, default=8787)
    a = p.parse_args(argv)

    if a.serve:
        from .server import serve
        serve(port=a.port, replay=a.replay, sample=a.sample)
        return 0

    if a.replay:
        gw = Gateway.replay(a.sample)
        meta = gw.run_dir / "claim.txt"
        claim = a.claim or (meta.read_text().strip() if meta.exists() else DEFAULT_CLAIM)
    else:
        claim = a.claim or DEFAULT_CLAIM
        run_dir = SAMPLES_DIR / (a.record_as or slugify(claim))
        gw = Gateway("live", run_dir, resume=a.resume)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "claim.txt").write_text(claim + "\n")

    if not a.json:
        print(f"Claim: {claim}\nMode: {gw.mode} ({gw.run_dir.relative_to(SAMPLES_DIR.parent) if gw.run_dir.is_relative_to(SAMPLES_DIR.parent) else gw.run_dir})\n")
    audit = Recount(gw, max_spend_usd=a.max_spend, search_fallback=not a.no_search_fallback).run(claim, on_step=None if a.json else _print_step, max_steps=a.steps)

    if a.json:
        print(json.dumps(audit.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print(f"VERDICT: {audit.verdict}  (agreement {audit.agreement or '-'})")
        print(f"Spent via Otari: ${audit.spent_usd:.6f}" + ("  (recorded)" if audit.mode == "replay" else ""))
        if audit.stopped:
            print(f"Stopped early: {audit.stopped}")
    return 0 if not audit.stopped else 2


if __name__ == "__main__":
    sys.exit(main())
