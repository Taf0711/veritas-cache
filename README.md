# veritas-cache

veritas-cache is an OpenAI-compatible cache proxy. It receives requests from an OpenAI SDK client. It returns a stored response on a cache hit. It forwards the request on a cache miss.

The proxy supports exact-match and semantic caching. Semantic hits use one static global
threshold or the adaptive ld3 policy. Streaming and non-streaming requests share one cache
entry.

## Requirements

- Rust 1.96 or newer
- An API key for an upstream LLM provider, such as OpenAI
- Downloaded local model files for embeddings

## Download the embedding model

Run the fetch script once before the first build. The script downloads the model files into `models/`.

```bash
./scripts/fetch_model.sh
```

Do not commit the files in `models/`.

## Run the proxy

Set the upstream base URL and API key. Set `SEMANTIC_THRESHOLD` if you want a threshold other than the default `0.85`. Then start the server.

```bash
export UPSTREAM_BASE_URL=https://api.openai.com
export OPENAI_API_KEY=your_key_here
cargo run
```

The proxy listens on `127.0.0.1:8080`.

## Install as a service (macOS)

Run the installer once. It copies the binary to `~/.local/bin`, copies the model files to
`~/.config/veritas-cache/models/`, and registers a launchd agent that starts at login.

```bash
./scripts/install.sh
```

The service listens on `127.0.0.1:18091` by default. Set `VERITAS_PORT` to change it.

## Run with Docker

```bash
docker build -t veritas-cache .
docker run -p 8080:8080 -v veritas-data:/data veritas-cache
```

The image downloads the model files at build time. The volume keeps the cache database.
The default upstream is the OpenAI API. Point it at another compatible provider with
`-e UPSTREAM_BASE_URL=https://openrouter.ai/api`.

## Point an OpenAI SDK client at the proxy

Change the `base_url` in the client configuration. Use the same API key.

Python example:

```python
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="http://127.0.0.1:8080/v1",
)
```

## Configuration

All settings have defaults. Environment variables win over config file values.

- `PORT` sets the listen port. The default is `8080`.
- `CACHE_DB_PATH` sets the SQLite path. The default is `cache.db`.
- `UPSTREAM_BASE_URL` sets the upstream API. The default is `https://api.openai.com`.
- `SEMANTIC_THRESHOLD` sets the minimum cosine similarity for a semantic hit. The default is `0.85`.
- `SEMANTIC_POLICY=ld3` selects the adaptive per-entry policy. `ADAPTIVE_DELTA` sets its error budget. The default policy is `static`.
- `CACHE_TTL_SECONDS` expires entries older than the limit. The default `0` disables expiry.
- `CACHE_MAX_ENTRIES` evicts the least recently used entries beyond the cap. The default `0` disables the cap.
- `CACHE_EXACT_ONLY_MODELS` lists model names that use exact matching only. Use a comma between names.
- `CACHE_SHADOW=1` enables shadow mode. See the shadow mode section.
- `CACHE_CONFIG` points to a JSON file with any of these settings in snake_case keys.

## Cache behavior

- The proxy checks exact request matches first.
- If the exact match misses, the proxy embeds the prompt and checks approximate nearest neighbors with a cosine similarity threshold.
- `x-cache: HIT` means the response came from the cache. `x-cache: MISS` means the proxy called the upstream API and stored the response.
- `x-cache-match: exact` marks an exact hit. `x-cache-match: semantic` marks a semantic hit.
- `x-cache-sim: 0.876543` shows the cosine similarity of a semantic hit.
- The cache key covers the full request, including `tool_choice`. It ignores `stream` and `stream_options`.
- Streaming requests pass chunks through live. The proxy caches the assembled completion when the stream ends. Streaming hits are served as SSE.
- Cache hits carry synthesized usage. The prompt token count matches the new request.
- Exact-only models skip the semantic path. They still store responses for exact reuse.

## Benchmark

The repository contains a benchmark trace, a replay harness, and four cache
decision policies. The full scientific report is in `bench/REPORT.md`.

Method summary: 20,000 prompts in 8,101 equivalence classes from Quora Question
Pairs, replayed 10 times for 200,000 queries. Hit latency is measured. Miss
latency uses a disclosed lognormal model.

Findings at a glance:

- A random wrong entry embeds at 0.05 mean cosine similarity. The nearest wrong
  entry embeds at 0.64. The nearest neighbor is the error source.
- The per-entry adaptive policy holds its error budget at every operating
  point. The measured false-hit rate stays 10 to 20 times below the budget.
- At matched error, the per-entry policy beats the global adaptive policy by
  about 20 points of hit rate. This reproduces the central claim of the vCache
  paper (arXiv 2502.03771).
- A tuned static threshold reaches a higher raw hit rate on this trace. It
  gives no error guarantee and needs labeled data to tune.
- The lookup p50 is about 18.6 ms. A hit is about 43 times faster than a
  modeled miss at the median.

Run the measurements and build the charts.

```bash
python3 scripts/build_trace.py
cargo test --release -- --ignored trace_similarity_separation --nocapture
cargo test --release -- --ignored trace_nearest_neighbor_difficulty --nocapture
cargo run --release --bin bench
python3 scripts/make_charts.py
```

## Metrics

`GET /metrics` returns request counters as JSON. The counters are `hits_exact`, `hits_semantic`, `misses`, `stores`, and `evicted`. The counters reset on restart.

## Shadow mode

Set `CACHE_SHADOW=1` to log every decision without serving from cache. The `shadow_log` table records each decision, its similarity, the cached response, and the fresh upstream response. Use the log to judge decisions offline against real traffic.

## Status

Phase 1: exact-match and semantic cache proxy with one static threshold. Done.
Phase 2: benchmark trace, replay harness, and baseline measurements. Done.
Phase 3: per-entry adaptive thresholds with a measured error bound. Done. The ld3 policy is
wired into the proxy.
Phase 5: productionization. Done. SSE streaming, persisted adaptive state, TTL and LRU
eviction, exact-only mode, metrics, and a JSON config file.
Phase 6: real-traffic evals. In progress. Shadow mode is done. The Splice control-loop
experiment harness is built.
