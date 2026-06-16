#!/bin/zsh
# Run occlude on an input video while polling phys_footprint
# (Activity Monitor's "Memory" column) every 5s. Logs the per-poll
# footprint, system memory pressure, and final summary.
#
# Usage: monitor_run.sh <input> <output> [blur]
set -u

INPUT="${1:?input video required}"
OUTPUT="${2:?output path required}"
BLUR="${3:-99}"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY="$ROOT/.venv/bin/python"
OCCLUDE_CLI="$ROOT/.venv/bin/occlude"

FOOT_LOG="${FOOT_LOG:-/tmp/occlude_footprint.log}"
PSUTIL_LOG="${PSUTIL_LOG:-/tmp/occlude_psutil.log}"
STDOUT_LOG="${STDOUT_LOG:-/tmp/occlude_stdout.log}"

: > "$FOOT_LOG"
: > "$PSUTIL_LOG"
: > "$STDOUT_LOG"

echo "input    = $INPUT"
echo "output   = $OUTPUT"
echo "footlog  = $FOOT_LOG"
echo "psutillog= $PSUTIL_LOG"

OCCLUDE_MEM_LOG="$PSUTIL_LOG" "$OCCLUDE_CLI" \
    --input "$INPUT" --output "$OUTPUT" --blur-strength "$BLUR" \
    >"$STDOUT_LOG" 2>&1 &
PID=$!
echo "pid      = $PID"
echo "started  = $(date '+%Y-%m-%d %H:%M:%S')"

T0=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - T0))
    FOOT_KB=$(/usr/bin/footprint --pid "$PID" 2>/dev/null \
        | awk -F'Footprint: ' '/Footprint:/ { gsub(" KB.*","",$2); print $2; exit }')
    PRESSURE=$(/usr/bin/memory_pressure -Q 2>/dev/null \
        | awk -F': ' '/System-wide memory free percentage/ { print $2; exit }')
    if [[ -n "$FOOT_KB" ]]; then
        FOOT_MB=$((FOOT_KB / 1024))
        echo "t=${ELAPSED}s footprint=${FOOT_MB}MB pressure=${PRESSURE}" \
            >> "$FOOT_LOG"
    fi
    sleep 5
done

EXIT=$(wait "$PID"; echo $?)
T1=$(date +%s)
DUR=$((T1 - T0))

echo "exit     = $EXIT"
echo "duration = ${DUR}s"
echo
echo "=== last 5 footprint samples ==="
tail -5 "$FOOT_LOG"
echo
echo "=== peak footprint ==="
sort -t= -k3 -n "$FOOT_LOG" | tail -3
echo
echo "=== last 5 stdout lines ==="
tail -5 "$STDOUT_LOG"

exit $EXIT
