#!/bin/sh
# `sh run-all.sh` ignores a Bash shebang. Re-exec so that this launcher works
# for both that common invocation and direct `./run-all.sh` execution.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

cd "$(dirname "$0")"

backend_pid=""
frontend_pid=""

cleanup() {
  trap - EXIT
  for pid in "$backend_pid" "$frontend_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait "$backend_pid" 2>/dev/null || true
  wait "$frontend_pid" 2>/dev/null || true
}

on_signal() {
  exit 130
}

trap cleanup EXIT
trap on_signal INT TERM

./run-backend.sh &
backend_pid=$!
./run-frontend.sh &
frontend_pid=$!

while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
  sleep 1
done

if ! kill -0 "$backend_pid" 2>/dev/null; then
  wait "$backend_pid"
else
  wait "$frontend_pid"
fi
