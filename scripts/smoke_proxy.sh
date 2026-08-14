#!/bin/sh
# Run an end-to-end smoke test of the proxy without a real API.
set -e

PROXY_URL="http://127.0.0.1:18080"
MOCK_URL="http://127.0.0.1:18099"
PORT=18080
WORK_DIR=$(mktemp -d)
DB_PATH="$WORK_DIR/cache.db"
BODY_ONE="$WORK_DIR/body_one.json"
BODY_TWO="$WORK_DIR/body_two.json"
HEADERS_FILE="$WORK_DIR/headers.txt"
MOCK_PID=""
PROXY_PID=""

cleanup() {
    if [ -n "$PROXY_PID" ]; then
        kill "$PROXY_PID" 2>/dev/null || true
    fi
    if [ -n "$MOCK_PID" ]; then
        kill "$MOCK_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# Build two distinct chat completion bodies.
cat > "$BODY_ONE" <<'EOF'
{"model":"gpt-4o-mini","messages":[{"role":"user","content":"what is 2+2"}]}
EOF
cat > "$BODY_TWO" <<'EOF'
{"model":"gpt-4o-mini","messages":[{"role":"user","content":"who wrote the declaration of independence"}]}
EOF
cat > "$WORK_DIR/body_stream.json" <<'EOF'
{"model":"gpt-4o-mini","messages":[{"role":"user","content":"what is the capital of france"}],"stream":true}
EOF
cat > "$WORK_DIR/body_forced.json" <<'EOF'
{"model":"gpt-4o-mini","messages":[{"role":"user","content":"exact only tool call"}],"tool_choice":{"type":"function","function":{"name":"run"}}}
EOF
cat > "$WORK_DIR/body_auto.json" <<'EOF'
{"model":"gpt-4o-mini","messages":[{"role":"user","content":"exact only tool call"}],"tool_choice":"auto"}
EOF
cat > "$WORK_DIR/body_exact_only.json" <<'EOF'
{"model":"gpt-4o-mini","messages":[{"role":"user","content":"exact only body"}]}
EOF

# Start the mock upstream.
python3 scripts/mock_upstream.py &
MOCK_PID=$!
sleep 1

# Start the proxy against the mock upstream.
CACHE_DB_PATH="$DB_PATH" PORT="$PORT" UPSTREAM_BASE_URL="$MOCK_URL" cargo run --release --bin veritas-cache &
PROXY_PID=$!

# Wait for the proxy health endpoint to return the expected body.
WAITED=0
until [ "$(curl -sS "$PROXY_URL/health" 2>/dev/null)" = "ok" ]; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ "$WAITED" -ge 60 ]; then
        echo "FAIL proxy did not become ready"
        exit 1
    fi
done

fail() {
    echo "FAIL $1"
    exit 1
}

# Request one: a fresh body with a dummy API key. Expect a miss.
# The proxy must ignore inbound auth and pass the key through.
curl -sS -D "$HEADERS_FILE" -o /dev/null -X POST \
    -H "Content-Type: application/json" -H "Authorization: Bearer dummy" \
    --data-binary "@$BODY_ONE" "$PROXY_URL/v1/chat/completions"
if grep -q "^x-cache: MISS" "$HEADERS_FILE"; then
    echo "PASS first request is a miss"
else
    fail "first request was not a miss"
fi

# Request two: the same body. Expect an exact hit.
curl -sS -D "$HEADERS_FILE" -o /dev/null -X POST \
    -H "Content-Type: application/json" --data-binary "@$BODY_ONE" "$PROXY_URL/v1/chat/completions"
if grep -q "^x-cache: HIT" "$HEADERS_FILE" && grep -q "^x-cache-match: exact" "$HEADERS_FILE"; then
    echo "PASS repeated request is an exact hit"
else
    fail "repeated request was not an exact hit"
fi

# Request three: a different body. Expect a miss.
curl -sS -D "$HEADERS_FILE" -o /dev/null -X POST \
    -H "Content-Type: application/json" --data-binary "@$BODY_TWO" "$PROXY_URL/v1/chat/completions"
if grep -q "^x-cache: MISS" "$HEADERS_FILE"; then
    echo "PASS different request is a miss"
else
    fail "different request was not a miss"
fi

# Request four: a streaming body. Expect a miss and an SSE body.
BODY_STREAM="$WORK_DIR/body_stream.json"
curl -sS -D "$HEADERS_FILE" -o "$WORK_DIR/stream_body.txt" -X POST \
    -H "Content-Type: application/json" --data-binary "@$BODY_STREAM" "$PROXY_URL/v1/chat/completions"
if grep -q "^x-cache: MISS" "$HEADERS_FILE" && grep -q "^content-type: text/event-stream" "$HEADERS_FILE"; then
    echo "PASS streaming first request is a miss with SSE"
else
    fail "streaming first request was not an SSE miss"
fi
if tail -c 20 "$WORK_DIR/stream_body.txt" | grep -q "data: \[DONE\]"; then
    echo "PASS streaming miss body ends with DONE"
else
    fail "streaming miss body did not end with DONE"
fi

# Request five: the same streaming body. Expect an exact hit with SSE.
curl -sS -D "$HEADERS_FILE" -o "$WORK_DIR/stream_hit_body.txt" -X POST \
    -H "Content-Type: application/json" --data-binary "@$BODY_STREAM" "$PROXY_URL/v1/chat/completions"
if grep -q "^x-cache: HIT" "$HEADERS_FILE" && grep -q "^x-cache-match: exact" "$HEADERS_FILE" && grep -q "^content-type: text/event-stream" "$HEADERS_FILE"; then
    echo "PASS streaming repeated request is an exact hit with SSE"
else
    fail "streaming repeated request was not an SSE hit"
fi
if tail -c 20 "$WORK_DIR/stream_hit_body.txt" | grep -q "data: \[DONE\]"; then
    echo "PASS streaming hit body ends with DONE"
else
    fail "streaming hit body did not end with DONE"
fi

# Exact-only mode: restart the proxy with the model marked exact-only.
kill "$PROXY_PID" 2>/dev/null || true
wait "$PROXY_PID" 2>/dev/null || true
CACHE_DB_PATH="$DB_PATH" PORT="$PORT" UPSTREAM_BASE_URL="$MOCK_URL" \
    CACHE_EXACT_ONLY_MODELS="gpt-4o-mini" cargo run --release --bin veritas-cache &
PROXY_PID=$!
WAITED=0
until [ "$(curl -sS "$PROXY_URL/health" 2>/dev/null)" = "ok" ]; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ "$WAITED" -ge 60 ]; then
        echo "FAIL proxy did not become ready after the exact-only restart"
        exit 1
    fi
done

# Request eight: a fresh body under exact-only mode. Expect a miss then an exact hit.
curl -sS -D "$HEADERS_FILE" -o /dev/null -X POST \
    -H "Content-Type: application/json" --data-binary "@$WORK_DIR/body_exact_only.json" "$PROXY_URL/v1/chat/completions"
if grep -q "^x-cache: MISS" "$HEADERS_FILE"; then
    echo "PASS exact-only first request is a miss"
else
    fail "exact-only first request was not a miss"
fi
curl -sS -D "$HEADERS_FILE" -o /dev/null -X POST \
    -H "Content-Type: application/json" --data-binary "@$WORK_DIR/body_exact_only.json" "$PROXY_URL/v1/chat/completions"
if grep -q "^x-cache: HIT" "$HEADERS_FILE" && grep -q "^x-cache-match: exact" "$HEADERS_FILE"; then
    echo "PASS exact-only repeated request is an exact hit"
else
    fail "exact-only repeated request was not an exact hit"
fi

# Requests nine and ten: two bodies that differ only in tool_choice.
# Under exact-only mode each variant must miss first and hit on repeat.
for BODY in "$WORK_DIR/body_forced.json" "$WORK_DIR/body_auto.json"; do
    curl -sS -D "$HEADERS_FILE" -o /dev/null -X POST \
        -H "Content-Type: application/json" --data-binary "@$BODY" "$PROXY_URL/v1/chat/completions"
    if grep -q "^x-cache: MISS" "$HEADERS_FILE"; then
        echo "PASS tool_choice variant is a miss"
    else
        fail "tool_choice variant $BODY was not a miss"
    fi
    curl -sS -D "$HEADERS_FILE" -o /dev/null -X POST \
        -H "Content-Type: application/json" --data-binary "@$BODY" "$PROXY_URL/v1/chat/completions"
    if grep -q "^x-cache: HIT" "$HEADERS_FILE" && grep -q "^x-cache-match: exact" "$HEADERS_FILE"; then
        echo "PASS tool_choice variant repeats as an exact hit"
    else
        fail "tool_choice variant $BODY did not repeat as an exact hit"
    fi
done

# Restart once more. The database now holds exact-only entries without embeddings.
# The boot index rebuild must skip them and serve their exact hits.
kill "$PROXY_PID" 2>/dev/null || true
wait "$PROXY_PID" 2>/dev/null || true
CACHE_DB_PATH="$DB_PATH" PORT="$PORT" UPSTREAM_BASE_URL="$MOCK_URL" \
    CACHE_EXACT_ONLY_MODELS="gpt-4o-mini" cargo run --release --bin veritas-cache &
PROXY_PID=$!
WAITED=0
until [ "$(curl -sS "$PROXY_URL/health" 2>/dev/null)" = "ok" ]; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ "$WAITED" -ge 60 ]; then
        echo "FAIL proxy did not become ready after the final restart"
        exit 1
    fi
done
curl -sS -D "$HEADERS_FILE" -o /dev/null -X POST \
    -H "Content-Type: application/json" --data-binary "@$WORK_DIR/body_exact_only.json" "$PROXY_URL/v1/chat/completions"
if grep -q "^x-cache: HIT" "$HEADERS_FILE" && grep -q "^x-cache-match: exact" "$HEADERS_FILE"; then
    echo "PASS boot over exact-only entries serves their exact hits"
else
    fail "boot over exact-only entries did not serve an exact hit"
fi

echo "SMOKE PASS"
