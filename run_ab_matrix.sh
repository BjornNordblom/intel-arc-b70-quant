#!/usr/bin/env bash
# One A/B matrix cell: launch -> bench -> capture -> (optional) teardown.
#
# Usage: ./run_ab_matrix.sh <label> <image> <max_num_seqs> [scenarios] [s2_repeat]
# Env:   KEEP=1 keeps the container running for manual inspection (default: remove)
#
# Examples:
#   ./run_ab_matrix.sh A0-base-seqs8  'vllm/vllm-openai-xpu@sha256:f01e24...' 8 s2 2
#   ./run_ab_matrix.sh A2-split-seqs8 'vllm-xpu-gdn-split:0.1.12.3-p1'        8 s1,s2,s3 3
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LABEL=${1:?usage: run_ab_matrix.sh <label> <image> <max_num_seqs> [scenarios] [s2_repeat]}
IMAGE=${2:?image required}
SEQS=${3:?max_num_seqs required}
SCENARIOS=${4:-s1,s2,s3}
S2_REPEAT=${5:-3}
NAME=${NAME:-swift-b70-mtp-split}
PORT=${PORT:-8080}
KEEP=${KEEP:-0}

cd "$SCRIPT_DIR"
LAUNCH_LOG=$(mktemp)

set +e
IMAGE="$IMAGE" NAME="$NAME" PORT="$PORT" MAX_NUM_SEQS="$SEQS" \
  ./launch.sh 2>&1 | tee "$LAUNCH_LOG"
launch_rc=$?
set -e

OUT=$(sed -n 's/^OUT=//p' "$LAUNCH_LOG" | tail -1)

if [ -z "$OUT" ] || [ ! -d "$OUT" ]; then
  echo "run_ab_matrix: launcher did not report an artifact dir (rc=$launch_rc)" >&2
  rm -f "$LAUNCH_LOG"
  exit 2
fi

{
  echo "label=$LABEL"
  echo "image=$IMAGE"
  echo "max_num_seqs=$SEQS"
  echo "scenarios=$SCENARIOS"
  echo "s2_repeat=$S2_REPEAT"
  echo "launcher_rc=$launch_rc"
} > "$OUT/cell.txt"

bench_rc=1
if [ "$launch_rc" -eq 0 ]; then
  set +e
  python3 bench_mixed.py --outdir "$OUT" --container "$NAME" \
    --scenarios "$SCENARIOS" --s2-repeat "$S2_REPEAT"
  bench_rc=$?
  set -e
else
  echo "launcher failed (rc=$launch_rc) — skipping bench" | tee -a "$OUT/cell.txt"
fi

# evidence: full server log + final state
docker logs "$NAME" > "$OUT/server.log" 2>&1 || true
docker inspect "$NAME" > "$OUT/docker-inspect.json" 2>&1 || true

echo "bench_rc=$bench_rc" >> "$OUT/cell.txt"
echo "cell=$LABEL image=$IMAGE seqs=$SEQS launch_rc=$launch_rc bench_rc=$bench_rc out=$OUT"

if [ "$KEEP" != "1" ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

rm -f "$LAUNCH_LOG"
exit "$bench_rc"
