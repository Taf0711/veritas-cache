#!/bin/sh
# Run the Phase 6.3 control-loop experiment.
# Arms: baseline (shadow mode, pass-through), exact (exact-only mode), static, ld3.
# Every arm shares the proxy path so the arms differ in serving only.
set -e

UPSTREAM_URL="https://openrouter.ai/api"
PORT=18090
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
SUITE="$REPO_ROOT/bench/splice/suite.json"
TEMPLATE_DIR="$REPO_ROOT/bench/splice/xdg"
RUNS_DIR="$REPO_ROOT/bench/splice/runs"
PROXY_BIN="$REPO_ROOT/target/release/veritas-cache"
# Prefer the dev build from bench/splice/bin. Fall back to the splice on PATH.
if [ -z "${SPLICE_BIN:-}" ]; then
    if [ -x "$REPO_ROOT/bench/splice/bin/splice" ]; then
        SPLICE_BIN="$REPO_ROOT/bench/splice/bin/splice"
    else
        SPLICE_BIN="splice"
    fi
fi
MODEL="${SPLICE_EXPERIMENT_MODEL:-openai/gpt-4o-mini}"

# The proxy resolves its model files relative to the working directory.
cd "$REPO_ROOT"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

# Fail fast when the tools are missing.
if ! command -v "$SPLICE_BIN" >/dev/null 2>&1; then
    echo "FAIL splice binary not found: $SPLICE_BIN"
    exit 1
fi
# A real run needs the upstream key. The key never lands in a file.
# The fixture config names the env var. Splice reads it from its own environment.
# Dry-run mode makes no upstream calls, so it skips this check.
if [ "$DRY_RUN" = "0" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "FAIL OPENROUTER_API_KEY is not set"
    exit 1
fi

cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --quiet

# Render the XDG fixture into a fresh directory.
# The model value contains a slash, so sed uses a pipe delimiter.
if [ "$DRY_RUN" = "1" ]; then
    WORK_DIR=$(mktemp -d)
else
    WORK_DIR="$RUNS_DIR/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$WORK_DIR"
fi
XDG_DIR="$WORK_DIR/xdg"
mkdir -p "$XDG_DIR/splice"
for TEMPLATE in config.json stage-models.json; do
    sed "s|__MODEL__|$MODEL|g" "$TEMPLATE_DIR/$TEMPLATE.template" > "$XDG_DIR/splice/$TEMPLATE"
done

if [ "$DRY_RUN" = "1" ]; then
    trap 'kill $PROXY_PID 2>/dev/null || true; rm -rf "$WORK_DIR"' EXIT
else
    ln -sfn "$WORK_DIR" "$RUNS_DIR/latest"
    # Kill the proxy on any failure so the port is free for the next run.
    trap 'kill $PROXY_PID 2>/dev/null || true' EXIT
fi

# Validate the suite offline before any run.
"$SPLICE_BIN" eval --suite "$SUITE"
echo "PASS suite validates"

# Start the proxy for one arm. Wait for health. Run nothing without health.
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

if [ "$DRY_RUN" = "1" ]; then
    start_proxy baseline CACHE_SHADOW=1
    stop_proxy
    echo "DRY RUN OK"
    exit 0
fi

# Preflight: prove the upstream answers through the proxy before the arms run.
# Use a throwaway database so the probe response never pollutes an arm.
if [ "$DRY_RUN" = "0" ]; then
    start_proxy preflight CACHE_SHADOW=1
    cat > "$WORK_DIR/probe.json" <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"say ok"}],"max_tokens":1}
EOF
    PROBE_CODE=$(curl -sS -o /dev/null -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        --data-binary "@$WORK_DIR/probe.json" "http://127.0.0.1:$PORT/v1/chat/completions")
    stop_proxy
    rm -rf "$WORK_DIR/preflight"
    if [ "$PROBE_CODE" != "200" ]; then
        echo "FAIL upstream probe returned HTTP $PROBE_CODE"
        echo "Check OPENROUTER_API_KEY and the model id $MODEL."
        exit 1
    fi
    echo "PASS upstream probe"
fi

# Models expect a python command. Shim it to python3 for the agent shells.
SHIM_DIR="$WORK_DIR/shim"
mkdir -p "$SHIM_DIR"
ln -sf "$(command -v python3)" "$SHIM_DIR/python"

# Each arm runs the same suite against the same proxy port.
for ARM in baseline exact static ld3; do
    case "$ARM" in
        baseline) FLAGS="CACHE_SHADOW=1" ;;
        exact)    FLAGS="CACHE_EXACT_ONLY_MODELS=$MODEL" ;;
        static)   FLAGS="SEMANTIC_POLICY=static" ;;
        ld3)      FLAGS="SEMANTIC_POLICY=ld3" ;;
    esac
    start_proxy "$ARM" $FLAGS
    # The cycle-forcing tasks fail by design, so eval bench can exit non-zero.
    # The report is the output.
    env XDG_CONFIG_HOME="$XDG_DIR" PATH="$SHIM_DIR:$PATH" "$SPLICE_BIN" eval bench \
        --suite "$SUITE" \
        --report-dir "$WORK_DIR/$ARM" \
        --timeout 5m \
        --agent-command env XDG_CONFIG_HOME="$XDG_DIR" "$SPLICE_BIN" exec -o stream-json -C {workspace} "{prompt}" || true
    stop_proxy
    echo "PASS arm $ARM finished"
done

echo "Run dir: $WORK_DIR"
echo "Compare with: python3 $REPO_ROOT/bench/splice/diff_arms.py"
