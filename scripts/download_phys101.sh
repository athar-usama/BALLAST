#!/usr/bin/env bash
# Physics 101 (MIT CSAIL) is a flaky, slow, unauthenticated HTTP download:
# a single curl invocation reliably dies partway (exit 18, "partial file")
# on this connection. curl's own --retry only covers connection-establish
# failures, not a mid-transfer drop, so this loop re-invokes curl with
# -C - (resume from the current byte offset) until the local file size
# matches the server's own Content-Length, with a hard cap on attempts.
set -uo pipefail

URL="https://phys101.csail.mit.edu/data/phys101_v1.0.tar"
OUT="data/phys101/phys101_v1.0.tar"
MAX_ATTEMPTS=60

mkdir -p "$(dirname "$OUT")"

expected_size=$(curl -sI --max-time 30 "$URL" | grep -i '^content-length:' | tr -d '\r' | awk '{print $2}')
echo "expected size: ${expected_size:-unknown} bytes"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    current_size=0
    if [ -f "$OUT" ]; then
        current_size=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    fi
    if [ -n "${expected_size:-}" ] && [ "$current_size" -ge "$expected_size" ]; then
        echo "download complete: $current_size bytes"
        exit 0
    fi
    echo "attempt $attempt: resuming from $current_size bytes"
    curl -sL -C - --max-time 600 -o "$OUT" "$URL"
    code=$?
    new_size=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    echo "  curl exit $code, size now $new_size bytes"
    if [ "$new_size" -le "$current_size" ] && [ "$code" -ne 0 ]; then
        sleep 5  # no progress at all this attempt: back off briefly before retrying
    fi
done

echo "gave up after $MAX_ATTEMPTS attempts"
exit 1
