#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_ROOT"

HOST=${LOCALTUNE_HOST:-127.0.0.1}
PORT=${LOCALTUNE_PORT:-6543}
OPEN_BROWSER=1
EXTRA_ARGS=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --host)
            HOST=$2
            shift 2
            ;;
        --port)
            PORT=$2
            shift 2
            ;;
        --no-browser)
            OPEN_BROWSER=0
            shift
            ;;
        --no-tensorboard|--no-frontend-build|--skip-training-deps)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            echo "[WARN] Unknown argument ignored: $1"
            shift
            ;;
    esac
done

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PYTHON="uv run python"
else
    echo "[ERROR] Python environment is missing. Install uv and run: uv sync"
    exit 1
fi

BROWSER_HOST=$HOST
if [ "$BROWSER_HOST" = "0.0.0.0" ]; then
    BROWSER_HOST=127.0.0.1
fi
URL="http://$BROWSER_HOST:$PORT"

echo "[INFO] Dashboard: $URL"
if [ "$OPEN_BROWSER" -eq 1 ]; then
    # shellcheck disable=SC2086
    $PYTHON scripts/wait_for_dashboard.py \
        --health-url "$URL/api/status" \
        --browser-url "$URL" &
fi

# shellcheck disable=SC2086
exec $PYTHON scripts/start_dashboard.py --host "$HOST" --port "$PORT" $EXTRA_ARGS
