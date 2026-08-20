#!/bin/sh
# Show veritas-cache status. Metrics come from the running proxy.
PORT="${VERITAS_PORT:-18091}"
DB="${CACHE_DB_PATH:-$HOME/.config/veritas-cache/dogfood.db}"

health=$(curl -s --max-time 3 "http://127.0.0.1:$PORT/health")
if [ "$health" != "ok" ]; then
    echo "DOWN  veritas-cache does not answer on 127.0.0.1:$PORT"
    exit 1
fi

metrics=$(curl -s --max-time 3 "http://127.0.0.1:$PORT/metrics")
METRICS="$metrics" python3 - "$PORT" <<'EOF'
import json, os, sys
port = sys.argv[1]
d = json.loads(os.environ['METRICS'])
hits = d['hits_exact'] + d['hits_semantic']
total = hits + d['misses'] + d['bypasses']
rate = hits / total * 100 if total else 0.0
print(f'UP    127.0.0.1:{port}')
print(f'      hits {hits} (exact {d["hits_exact"]}, semantic {d["hits_semantic"]}), misses {d["misses"]}, stores {d["stores"]}, bypasses {d["bypasses"]}')
print(f'      hit rate {rate:.1f}% over {total} requests since the proxy last started')
if total == 0:
    print('      no traffic seen: pi must run with --provider veritas or a models.json entry that points at this port')

# Dollars avoided per hit: prompt tokens at the cache-read rate plus completion
# tokens at the output rate. Prices are dollars per million tokens, matched by
# model id prefix. Rates change; treat the total as an estimate.
RATES = {
    'deepseek/deepseek-v4-flash-0731': (0.028, 0.28),
    'z-ai/glm-5.2': (0.0945, 1.98),
}
rows = d.get('tokens_avoided', [])
if rows:
    avoided = 0.0
    for model, prompt, completion in rows:
        r = next((v for k, v in RATES.items() if model.startswith(k)), None)
        if r:
            avoided += prompt / 1e6 * r[0] + completion / 1e6 * r[1]
        else:
            print(f'      no rate for {model}: {prompt + completion} tokens avoided, price unknown')
    saved_tokens = sum(p + c for _, p, c in rows)
    print(f'      avoided {saved_tokens} tokens, est. ${avoided:.4f} (all time, persisted)')
EOF

if [ -f "$DB" ]; then
    sqlite3 "$DB" "SELECT '      cached entries: ' || COUNT(*) FROM entries;" 2>/dev/null || true
fi
exit 0
