# veritas-cache

veritas-cache is an OpenAI-compatible cache proxy. It receives requests from an OpenAI SDK client. It returns a stored response on a cache hit. It forwards the request on a cache miss.

The proxy supports exact-match and semantic caching. Semantic hits use one static global
threshold or the adaptive ld3 policy.

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

## Point an OpenAI SDK client at the proxy

Change the `base_url` in the client configuration. Use the same API key.

Python example:

```python
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="http://127.0.0.1:8080/v1",
)
```

## Cache behavior

- The proxy checks exact request matches first.
- If the exact match misses, the proxy embeds the prompt and checks approximate nearest neighbors with a cosine similarity threshold.
- `x-cache: HIT` means the response came from the cache. `x-cache: MISS` means the proxy called the upstream API and stored the response.
- `x-cache-match: exact` marks an exact hit. `x-cache-match: semantic` marks a semantic hit.
- `x-cache-sim: 0.876543` shows the cosine similarity of a semantic hit.
- `SEMANTIC_THRESHOLD` controls the minimum cosine similarity for a semantic hit. The default is `0.85`.
- `SEMANTIC_POLICY=ld3` selects the adaptive per-entry policy. `ADAPTIVE_DELTA` sets its error
  budget. The default policy is `static`.

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

## Status

Phase 1: exact-match and semantic cache proxy with one static threshold. Done. Streaming is
not supported yet.
Phase 2: benchmark trace, replay harness, and baseline measurements. Done.
Phase 3: per-entry adaptive thresholds with a measured error bound. Done. The policies are
benchmarked in the harness. The ld3 policy is wired into the proxy.
