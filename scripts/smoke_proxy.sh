#!/bin/sh
# Run an end-to-end smoke test of the proxy without a real API.
set -e

PROXY_URL="http://127.0.0.1:8080"
MOCK_URL="http://127.0.0.1:18099"
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

# Start the mock upstream.
python3 scripts/mock_upstream.py &
MOCK_PID=$!
sleep 1

# Start the proxy against the mock upstream.
CACHE_DB_PATH="$DB_PATH" UPSTREAM_BASE_URL="$MOCK_URL" cargo run --release --bin veritas-cache &
PROXY_PID=$!

# Wait for the proxy health endpoint.
WAITED=0
until curl -sS "$PROXY_URL/health" > /dev/null 2>&1; do
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

# Request one: a fresh body. Expect a miss.
curl -sS -D "$HEADERS_FILE" -o /dev/null -X POST \
    -H "Content-Type: application/json" --data-binary "@$BODY_ONE" "$PROXY_URL/v1/chat/completions"
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

echo "SMOKE PASS"
