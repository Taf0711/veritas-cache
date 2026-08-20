# veritas

An OpenAI-compatible cache proxy for LLM calls. A repeated request gets a stored answer
instead of a paid API call.

## The goal

Semantic caches save real money. Production hit rates run 20 to 45 percent, and a hit
saves the full cost of the call. The risk is the false hit: a cached answer that does not
match the new question. Every shipped cache today controls that risk with one static
global similarity threshold. One threshold cannot be right for every entry, so operators
set it high and lose savings.

veritas learns a threshold per cached entry from observed outcomes. Each entry
serves only when its own measured error rate stays under a budget you set. The goal is to
cache more aggressively than a static threshold can safely allow, and to prove the bound
with a benchmark you can rerun yourself.

## Current state

Works today:

- Exact-match and semantic caching for chat completions, streaming and non-streaming.
- The adaptive per-entry policy (`ld3`), live in the proxy with persisted state.
- Shadow mode. Observe your own traffic and count would-be hits before you serve them.
- A bypass header for eval runs, TTL and LRU eviction, exact-only mode per model,
  metrics, and a JSON config file.
- A macOS installer that registers a launchd service, and a Dockerfile.

Measured so far:

- 200,000 replayed queries from a 20,000-prompt Quora trace. The adaptive policy holds
  its error budget at every operating point. The measured false-hit rate stays 10 to 20
  times below the budget. Details in `bench/REPORT.md`.
- An agent-harness experiment suite against a real coding agent. Serving cut upstream
  prompt tokens by 51 to 75 percent across runs. In the final configuration every
  passable task passed. Details in `bench/splice/RESULTS.md`.

Not done yet:

- The benchmark trace is single-turn question pairs. Agent-traffic numbers come from
  smaller experiment suites, not the 200,000-query harness.
- No tenant isolation, no per-entry invalidation API, no cost view in `/metrics`.
- Pre-1.0. The configuration surface may still change.

## Quickstart

Three ways to run it. All need an API key for an upstream LLM provider.

### Install as a service (macOS)

```bash
./scripts/install.sh
```

The installer builds the binary, installs it to `~/.local/bin`, installs the model files,
and registers a launchd agent that starts at login. The service listens on
`127.0.0.1:18091`. Set `VERITAS_PORT` to change the port.

### Run with Docker

```bash
docker build -t veritas .
docker run -p 8080:8080 -v veritas-data:/data veritas
```

The image downloads the model files at build time. The volume keeps the cache database.
The default upstream is the OpenAI API. Point it at another compatible provider with
`-e UPSTREAM_BASE_URL=https://openrouter.ai/api`.

### Run from source

You need Rust 1.96 or newer.

```bash
./scripts/fetch_model.sh
export UPSTREAM_BASE_URL=https://api.openai.com
export OPENAI_API_KEY=your_key_here
cargo run --release
```

The proxy listens on `127.0.0.1:8080`. Do not commit the files in `models/`.

## Point your client at the proxy

Change the `base_url` in the client configuration. Use the same API key. Nothing else
changes in your code.

```python
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="http://127.0.0.1:8080/v1",
)
```

The proxy works with any OpenAI-compatible client. Point several tools at one local
instance and the hit rate compounds across tools.

## See what the cache did

Every response carries headers.

- `x-cache: HIT` means the response came from the cache. `x-cache: MISS` means the proxy
  called the upstream API and stored the response. `x-cache: BYPASS` marks a request that
  skipped the cache.
- `x-cache-match: exact` or `x-cache-match: semantic` names the hit type.
- `x-cache-sim: 0.876543` shows the cosine similarity of a semantic hit.

Send `X-Veritas-Bypass: true` on a request to skip the cache. A bypass request never
serves a stored entry and never writes one. Use it for eval runs and diagnostics.

`GET /metrics` returns counters as JSON: `hits_exact`, `hits_semantic`, `misses`,
`stores`, `evicted`, `bypasses`. The counters reset on restart. The `tokens_avoided`
list persists in the database. It counts the prompt and completion tokens that cache
hits avoided, per model. `scripts/veritas-status.sh` prints these numbers with an
estimated dollar value.

## Configuration

All settings have defaults. Environment variables win over config file values.

- `HOST` sets the bind address. The default is `127.0.0.1`. Containers set `0.0.0.0`.
- `PORT` sets the listen port. The default is `8080`.
- `CACHE_DB_PATH` sets the SQLite path. The default is `cache.db`.
- `UPSTREAM_BASE_URL` sets the upstream API. The default is `https://api.openai.com`.
- `SEMANTIC_THRESHOLD` sets the minimum cosine similarity for a semantic hit. The default is `0.85`.
- `SEMANTIC_POLICY` selects the policy. Values are `static`, `ld3`, and `ld3s`. The default is `static`.
- `ADAPTIVE_DELTA` sets the error budget for the adaptive policies.
- `CACHE_TTL_SECONDS` expires entries older than the limit. The default `0` disables expiry.
- `CACHE_MAX_ENTRIES` evicts the least recently used entries beyond the cap. The default `0` disables the cap.
- `CACHE_EXACT_ONLY_MODELS` lists model names that use exact matching only. Use a comma between names.
- `CACHE_SHADOW=1` enables shadow mode. See the shadow mode section.
- `CACHE_CONFIG` points to a JSON file with any of these settings in snake_case keys.

## Cache behavior

- The proxy checks exact request matches first.
- If the exact match misses, the proxy embeds the prompt and checks approximate nearest
  neighbors against the policy threshold.
- The cache key covers the full request, including `tool_choice` and `prompt_cache_key`.
  It ignores `stream` and `stream_options`.
- Streaming requests pass chunks through live. The proxy caches the assembled completion
  when the stream ends. Streaming hits are served as SSE.
- Cache hits carry synthesized usage. The prompt token count matches the new request.
- Exact-only models skip the semantic path. They still store responses for exact reuse.

## Shadow mode

Set `CACHE_SHADOW=1` to log every decision without serving from cache. The `shadow_log`
table records each decision, its similarity, the cached response, and the fresh upstream
response. Run shadow mode on your own traffic first. Judge the would-be hits offline.
Then switch serving on with evidence.

## Benchmark

The repository contains a benchmark trace, a replay harness, and four cache decision
policies. The full scientific report is in `bench/REPORT.md`.

Method summary: 20,000 prompts in 8,101 equivalence classes from Quora Question Pairs,
replayed 10 times for 200,000 queries. Hit latency is measured. Miss latency uses a
disclosed lognormal model.

Findings at a glance:

- A random wrong entry embeds at 0.05 mean cosine similarity. The nearest wrong entry
  embeds at 0.64. The nearest neighbor is the error source.
- The per-entry adaptive policy holds its error budget at every operating point.
- At matched error, the per-entry policy beats the global adaptive policy by about 20
  points of hit rate. This reproduces the central claim of the vCache paper
  (arXiv 2502.03771). The implementation is original work, written from the paper.
- A tuned static threshold reaches a higher raw hit rate on this trace. It gives no error
  guarantee and needs labeled data to tune.
- The lookup p50 is about 18.6 ms. A hit is about 43 times faster than a modeled miss at
  the median.

Run the measurements and build the charts.

```bash
python3 scripts/build_trace.py
cargo test --release -- --ignored trace_similarity_separation --nocapture
cargo test --release -- --ignored trace_nearest_neighbor_difficulty --nocapture
cargo run --release --bin bench
python3 scripts/make_charts.py
```

## License

MIT. See `LICENSE`.
