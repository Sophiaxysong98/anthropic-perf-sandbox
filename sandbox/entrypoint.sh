#!/bin/bash
set -e

WORKSPACE="/home/agent/perf_takehome"
SOURCE="/opt/perf_takehome"

if [[ -d "$WORKSPACE/tests" ]]; then
    chattr +i "$WORKSPACE/tests"/*.py 2>/dev/null || true
    chattr +i "$WORKSPACE/tests"/__init__.py 2>/dev/null || true
    chattr +i "$WORKSPACE/tests" 2>/dev/null || true
fi

if [[ -f "$WORKSPACE/problem.py" ]]; then
    chattr +i "$WORKSPACE/problem.py" 2>/dev/null || true
    chattr +i "$WORKSPACE/watch_trace.py" 2>/dev/null || true
fi

if [[ ! -f "$WORKSPACE/perf_takehome.py" ]]; then
    cp -r "$SOURCE/perf_takehome.py" "$WORKSPACE/" 2>/dev/null || true
    cp -r "$SOURCE/problem.py" "$WORKSPACE/" 2>/dev/null || true
    cp -r "$SOURCE/watch_trace.py" "$WORKSPACE/" 2>/dev/null || true
    cp -r "$SOURCE/README.md" "$WORKSPACE/" 2>/dev/null || true
    
    chown agent:agent "$WORKSPACE"/*.py 2>/dev/null || true
    chown agent:agent "$WORKSPACE"/*.md 2>/dev/null || true
    
    chattr +i "$WORKSPACE/problem.py" 2>/dev/null || true
    chattr +i "$WORKSPACE/watch_trace.py" 2>/dev/null || true
fi

echo "[SANDBOX] Verifying immutable protection..."
VERIFY_FILE="$WORKSPACE/tests/submission_tests.py"
if [[ -f "$VERIFY_FILE" ]]; then
    if lsattr "$VERIFY_FILE" 2>/dev/null | grep -q 'i'; then
        echo "[SANDBOX] ✓ Immutable protection verified"
    else
        echo "[SANDBOX] ✗ FATAL: chattr +i failed - files are NOT protected!"
        echo "[SANDBOX] Ensure Docker has cap_add: [LINUX_IMMUTABLE]"
        exit 1
    fi
fi

exec gosu agent "$@"
