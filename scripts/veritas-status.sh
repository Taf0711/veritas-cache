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
EOF

if [ -f "$DB" ]; then
    sqlite3 "$DB" "SELECT '      cached entries: ' || COUNT(*) FROM entries;" 2>/dev/null || true
fi
exit 0
