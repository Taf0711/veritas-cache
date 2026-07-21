# veritas-cache

veritas-cache is an OpenAI-compatible cache proxy. It receives requests from an OpenAI SDK client. It returns a stored response on a cache hit. It forwards the request on a cache miss.

Phase 1 implements exact-match and semantic caching with one static global threshold.

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

## Status

Phase 1: exact-match and semantic cache with one static threshold. Streaming is not supported yet.
