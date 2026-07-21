# veritas-cache

veritas-cache is an OpenAI-compatible cache proxy. It receives requests from an OpenAI SDK client. It returns a stored response on a cache hit. It forwards the request on a cache miss.

Phase 1 implements exact-match caching only. Semantic matching comes later.

## Requirements

- Rust 1.96 or newer
- An API key for an upstream LLM provider, such as OpenAI

## Run the proxy

Set the upstream base URL and API key. Then start the server.

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

- The proxy stores exact request matches in a local SQLite database.
- `x-cache: HIT` means the response came from the cache.
- `x-cache: MISS` means the proxy called the upstream API and stored the response.

## Status

Phase 1: exact-match cache only. Streaming is not supported yet.
