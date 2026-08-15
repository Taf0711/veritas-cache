#!/bin/sh
# Run the Pi harness experiment against veritas-cache.
# Arms: baseline (shadow mode), exact (exact-only), static (static threshold).
# Each arm runs one small task twice in two fresh workspaces.
# The repeat shows whether the second identical request hits the cache.
set -e

PORT=18091
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEMPLATE="$REPO_ROOT/bench/pi/agent-dir/models.json.template"
RUNS_DIR="$REPO_ROOT/bench/pi/runs"
PROXY_BIN="$REPO_ROOT/target/release/veritas-cache"
MODEL="${PI_EXPERIMENT_MODEL:-deepseek/deepseek-v4-flash-0731}"
UPSTREAM_URL="https://openrouter.ai/api"
PROMPT="Create a file math_helpers.js that exports a function double(x) returning x * 2. Use module.exports. Then stop."

echo "info: experiment model is $MODEL"

cd "$REPO_ROOT"

if ! command -v pi >/dev/null 2>&1; then
    echo "FAIL pi is not on PATH"
    exit 1
fi

KEY_FILE="$HOME/.config/veritas-cache/openrouter_key"
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$KEY_FILE" ]; then
    OPENROUTER_API_KEY=$(tr -d '[:space:]' < "$KEY_FILE")
    export OPENROUTER_API_KEY
    echo "info: using the key from $KEY_FILE"
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "FAIL OPENROUTER_API_KEY is not set and $KEY_FILE is missing"
    exit 1
fi

cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --quiet

WORK_DIR="$RUNS_DIR/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$WORK_DIR"
ln -sfn "$WORK_DIR" "$RUNS_DIR/latest"
trap 'kill $PROXY_PID 2>/dev/null || true' EXIT

# Render the fixture agent dir. The key comes from the environment at request time.
AGENT_DIR="$WORK_DIR/agent-dir"
mkdir -p "$AGENT_DIR"
sed "s|__MODEL__|$MODEL|g" "$TEMPLATE" > "$AGENT_DIR/models.json"

start_proxy() {
    ARM="$1"
    shift
    ARM_DIR="$WORK_DIR/$ARM"
    mkdir -p "$ARM_DIR"
    env PORT="$PORT" \
        CACHE_DB_PATH="$ARM_DIR/cache.db" \
        UPSTREAM_BASE_URL="$UPSTREAM_URL" \
        "$@" "$PROXY_BIN" > "$ARM_DIR/proxy.log" 2>&1 &
    PROXY_PID=$!
    WAITED=0
    until [ "$(curl -sS "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "ok" ]; do
        sleep 1
        WAITED=$((WAITED + 1))
        if [ "$WAITED" -ge 60 ]; then
            echo "FAIL proxy for arm $ARM did not become ready"
            exit 1
        fi
    done
    echo "PASS proxy up for arm $ARM"
}

stop_proxy() {
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
}

# Run the task once in a fresh workspace. Print the cache status of the response.
run_task() {
    LABEL="$1"
    TASK_DIR="$WORK_DIR/$LABEL"
    mkdir -p "$TASK_DIR"
    OUT=$(cd "$TASK_DIR" && PI_CODING_AGENT_DIR="$AGENT_DIR" pi --provider veritas --model "$MODEL" -a -p "$PROMPT" 2>&1) || true
    if [ -f "$TASK_DIR/math_helpers.js" ]; then
        if node -e "const m = require('$TASK_DIR/math_helpers.js'); const d = m.double ?? m; if (typeof d !== 'function' || d(4) !== 8) process.exit(1)" 2>/dev/null; then
            echo "PASS $LABEL produced a working double"
        else
            echo "FAIL $LABEL produced a broken double"
        fi
    else
        echo "FAIL $LABEL produced no math_helpers.js"
        echo "$OUT" | tail -3
    fi
}

for ARM in baseline exact static; do
    case "$ARM" in
        baseline) FLAGS="CACHE_SHADOW=1" ;;
        exact)    FLAGS="CACHE_EXACT_ONLY_MODELS=$MODEL" ;;
        static)   FLAGS="SEMANTIC_POLICY=static" ;;
    esac
    start_proxy "$ARM" $FLAGS
    run_task "$ARM-first"
    run_task "$ARM-second"
    curl -sS "http://127.0.0.1:$PORT/metrics" || true
    echo ""
    stop_proxy
    echo "PASS arm $ARM finished"
done

echo "Run dir: $WORK_DIR"
