# Recount

**Paste a numeric claim. Recount recomputes it.**

Recount is a small fact-checking tool for numbers. It pulls the figures out of a claim, looks up the official values, re-does the arithmetic in a real code sandbox (and shows you the code that ran), then asks three model votes for a verdict. Every model call goes through the [Otari](https://github.com/mozilla-ai/otari) gateway.

Built by Abdul AmberNeel for the Mozilla.ai Otari hackathon ([Mozilla Devs Wilding](https://dorahacks.io/hackathon/mozilla-devs-wilding/detail)). Hackathon work lives on the `mozilla-devs-wilding` branch.

```
$ recount --replay
Claim: India's population grew from 1.32 billion in 2015 to 1.43 billion in 2023, an increase of more than 15 percent.

[extract]   model=openai/gpt-oss-20b
  - population: 1320000000.0 (2015)
  - population: 1430000000.0 (2023)
[source]    model=openai/gpt-oss-120b   (World Bank + Wikipedia MCP, web search)
[recompute] model=openai/gpt-oss-120b
  --- sandbox run 1 (executed in Otari code sandbox) ---
  | claimed = (1.43e9 - 1.32e9) / 1.32e9 * 100
  => 8.333333333333332
[verdict]   routing policy recount-auditor -> deepseek-ai/DeepSeek-V4-Flash-0731 (fallback)
  vote FALSE  The claimed 15% increase is wrong; the derived percent increase is 8.33%, not over 15%.
  vote FALSE  ...
  vote FALSE  ...
VERDICT: FALSE  (agreement 3/3)
Spent via Otari: $0.012417  (recorded)
```

## How it works

| Step | What happens | Otari feature |
|---|---|---|
| 1. extract | `openai/gpt-oss-20b` turns the claim into JSON: figures, years, units, the implied formula. | Multiple models: a small, cheap model for a simple job |
| 2. source | `openai/gpt-oss-120b` looks up official values through **MCP** servers stored in the Otari workspace (World Bank, Wikipedia) and **Otari web search** (`otari_web_search`, capped with `max_uses`). Search queries and their sources come back in `otari_tool_calls`, and Recount flags any cited URL that no search returned. If MCP returns nothing, a web-search-only fallback runs. | MCP via `mcp_server_ids`, web search, gateway tool loop (`max_tool_iterations`) |
| 3. recompute | `openai/gpt-oss-120b` on the **Responses API** with `code_interpreter`. Otari runs the code in its sandbox and returns each `code_interpreter_call` with the exact code and its output, which Recount prints. A number with no sandbox run behind it is marked untrusted. | Code execution |
| 4. verdict | The `recount-auditor` **routing policy** answers with `n: 3`; the majority vote wins. The policy falls back Nemotron → DeepSeek V4 Flash → gpt-oss-120b, and Recount shows which model served and whether a fallback happened (`Otari-Model`, `Otari-Fallback`, `Otari-Attempt-Count`). | Routing policy with fallback |
| all | Each step is tagged (`Otari-Tags: app=recount,step=…`) for cost attribution. Recount reads `usage.cost_usd` (which includes tool calls), stops starting new steps once a run passes `--max-spend`, and turns Otari budget refusals into a clear message instead of a stack trace. | Budgets and cost tracking |

### Why "show the executed code" matters

While building this we asked a model to run code through a chat-completions tool with a JSON response format. It replied with tidy `code` and `result` fields but never called the tool: the CAGR it reported (0.08711682932518143) was wrong; the real sandbox gave 0.08706202972134314. Both responses are recorded in [`samples/probes/`](samples/probes). Recount therefore trusts a recomputed number only when the gateway reports a `code_interpreter_call` for it.

## Setup

Python 3.10+, no third-party dependencies.

```bash
git clone -b mozilla-devs-wilding https://github.com/cryptocloud786/recount.git
cd recount
pip install -e .          # optional; gives you the `recount` command. Otherwise use `python -m recount`.
```

### Replay mode (no key, no spend)

Every live response is saved to `samples/<claim>/stepN_*.json` (response body plus Otari routing headers; never the API key). Replay runs the full pipeline from those files:

```bash
python -m recount --replay                 # CLI, recorded india-population audit
python -m recount --serve --replay         # web UI at http://127.0.0.1:8787
```

### Live mode

1. Create a workspace API key in the Otari dashboard and export it:
   ```bash
   export OTARI_API_KEY=...                 # see .env.example
   ```
   The base URL is `https://api.otari.ai/api/v1` (note `/api/v1`, not `/v1`).
2. In the dashboard, add the MCP servers (World Bank, Wikipedia) and put their ids in `recount/pipeline.py` (`MCP_WORLDBANK`, `MCP_WIKIPEDIA`).
3. Create a routing policy named `recount-auditor` with a fallback chain, for example Nemotron → DeepSeek V4 Flash → gpt-oss-120b.
4. Optional but recommended: set a workspace or member budget. Recount handles the refusal when it is hit.

```bash
python -m recount "India's GDP grew 7.2% in 2023, the fastest of any large economy."
python -m recount --serve                  # web UI, live
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--replay` / `--sample NAME` | use recorded responses from `samples/NAME` |
| `--max-spend 0.10` | stop starting new steps once this run has spent this many USD |
| `--resume` | live: reuse steps already recorded for this claim and pay only for the missing ones |
| `--steps N` | run only the first N steps |
| `--no-search-fallback` | skip the web-search retry when MCP finds nothing |
| `--json` | print the whole audit as JSON |

A full live audit of the demo claim cost $0.0124 (most of it the sandbox run).

## When a budget runs out

If Otari refuses a request for budget or funds (402, `insufficient_funds`, or a budget message), Recount stops before the next step, keeps the steps that finished, and prints:

```
Otari answered 402 (budget_exceeded): ...
  -> A spend cap stopped this request before any model ran, so nothing was charged.
     Daily Otari budgets reset at 00:00 UTC (05:30 IST). Meanwhile run `recount --replay` ...
```

Rate limits (429), a routing policy where every candidate refused (`all_candidates_rejected`), a missing key, and a wrong base URL each get their own message.

## Tests

The tests run offline against the recorded responses (no key, no network):

```bash
python -m unittest discover -s tests -t . -v     # or: pytest
```

They cover JSON parsing of model output, vote counting, budget, rate-limit and routing-error handling, the local spend cap, detection of answers that only claim to have run code, the MCP → web search fallback, the full replayed audit, and the web endpoint.

## What we learned about Otari

* All the tools (code execution, web search, MCP) worked on the first request with a workspace key.
* MCP calls aren't reported in `otari_tool_calls` on Chat Completions (web searches are), so the app can't show MCP provenance. In our recorded run the World Bank MCP call failed on a malformed country argument, and the model correctly reported "not found" rather than guessing. That honest gap is visible in the replay.
* `usage.cost_usd` includes gateway tool calls and `estimated_cost` does not, so show `cost_usd` to users.
* `n > 1` is refused by some providers. A routing policy turns that into a transparent fallback instead of an error.

## Layout

```
recount/otari.py     the only module that talks to Otari: auth, base URL, error classification, routing headers
recount/pipeline.py  the four steps and the vote
recount/replay.py    record every response; replay offline
recount/cli.py       command line
recount/server.py    stdlib web server for web/index.html
samples/             recorded Otari responses (demo audit + probe fixtures)
tests/               offline tests
```

## License

MIT. See [LICENSE](LICENSE).
