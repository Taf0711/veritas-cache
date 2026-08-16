#!/bin/sh
# Install veritas-cache as a user service.
# Copy the binary to ~/.local/bin. Copy the model files to the config dir.
# Register a launchd agent so the proxy starts at login and restarts on crash.
set -e

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/veritas-cache"
PLIST="$HOME/Library/LaunchAgents/com.veritas.cache.plist"
PORT="${VERITAS_PORT:-18091}"

cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --quiet

mkdir -p "$BIN_DIR" "$CONFIG_DIR/models" "$HOME/Library/LaunchAgents"
cp "$REPO_ROOT/target/release/veritas-cache" "$BIN_DIR/veritas-cache"

if [ ! -f "$CONFIG_DIR/models/model.onnx" ]; then
    if [ ! -f "$REPO_ROOT/models/model.onnx" ]; then
        echo "FAIL model files are missing. Run $REPO_ROOT/scripts/fetch_model.sh first."
        exit 1
    fi
    cp "$REPO_ROOT/models/model.onnx" "$REPO_ROOT/models/tokenizer.json" "$CONFIG_DIR/models/"
fi

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.veritas.cache</string>
    <key>ProgramArguments</key>
    <array>
        <string>$BIN_DIR/veritas-cache</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PORT</key>
        <string>$PORT</string>
        <key>UPSTREAM_BASE_URL</key>
        <string>https://openrouter.ai/api</string>
        <key>SEMANTIC_POLICY</key>
        <string>ld3s</string>
        <key>CACHE_DB_PATH</key>
        <string>$CONFIG_DIR/dogfood.db</string>
        <key>VERITAS_MODEL_DIR</key>
        <string>$CONFIG_DIR/models</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$CONFIG_DIR/proxy.log</string>
    <key>StandardErrorPath</key>
    <string>$CONFIG_DIR/proxy.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

sleep 2
if [ "$(curl -sS "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "ok" ]; then
    echo "PASS veritas-cache is installed and serving on 127.0.0.1:$PORT"
    echo "Logs: $CONFIG_DIR/proxy.log"
    echo "Stop: launchctl unload $PLIST"
else
    echo "FAIL the service did not come up. Check $CONFIG_DIR/proxy.log"
    exit 1
fi
