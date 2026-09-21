#!/usr/bin/env bash
# Launch the Swift-Qwen3.8-27b GPTQ-INT4 server on the Arc Pro B70.
#
# Defaults to the derived image that carries the GDN split-dispatch backport, so
# mixed spec-decode + prefill batches no longer abort the engine.
#
# Env overrides (all optional):
#   MODEL, NAME, PORT, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, MAX_MODEL_LEN,
#   GPU_UTIL, MTP_N, LANGUAGE_MODEL_ONLY, IMAGE, PATCH_DIR, TRITON_CACHE,
#   SERVED_NAME, READY_TIMEOUT, EXTRA_ARGS, PATCHES
#
# PATCHES=0 serves a stock image (e.g. vllm/vllm-openai-xpu:nightly) without
# the runtime MTP patches; use it to bisect upstream behaviour against the
# derived `vllm-xpu-gdn-split` image. Default 1.
#
# MTP_N=0 disables speculative decoding (needed for the vision bring-up,
# MAX_NUM_SEQ_FIX.md 11.1). EXTRA_ARGS is appended verbatim to `vllm serve`.
#
# Defaults: IMAGE=vllm-xpu-gdn-split:0.1.12.3-p1, NAME=swift-b70-mtp-split,
# PORT=8080, MAX_NUM_SEQS=4, MAX_NUM_BATCHED_TOKENS=8192, GPU_UTIL=0.94,
# MTP_N=3, LANGUAGE_MODEL_ONLY=1 (text-only), MAX_MODEL_LEN=131072, fp8 KV,
# prefix caching.
#
# Run the unpatched pinned baseline instead:
#   IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff...' MAX_NUM_SEQS=1 ./launch.sh
#
# Load the vision tower (not validated, see MAX_NUM_SEQ_FIX.md 11.1):
#   LANGUAGE_MODEL_ONLY=0 ./launch.sh
#
# NOTE: it first runs `docker rm -f $NAME`, so do not invoke it while a server
# with the same name is serving traffic. Waits for /v1/models afterwards and
# records run evidence in artifacts/<run-id>/.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:-/opt/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}
PATCH_DIR=${PATCH_DIR:-$SCRIPT_DIR/patches}
TRITON_CACHE=${TRITON_CACHE:-/opt/models/triton_cache}
IMAGE=${IMAGE:-vllm-xpu-gdn-split:0.1.12.3-p1}
NAME=${NAME:-swift-b70-mtp-split}
PORT=${PORT:-8080}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
GPU_UTIL=${GPU_UTIL:-0.94}
MTP_N=${MTP_N:-3}
# 1 = text-only (skip the vision tower, default), 0 = load the vision tower too
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-1}
SERVED_NAME=${SERVED_NAME:-swift38}
READY_TIMEOUT=${READY_TIMEOUT:-420}
PATCHES=${PATCHES:-1}

PATCH_MOUNTS=()
PATCH_PRELUDE=""
if [ "$PATCHES" = "1" ]; then
  PATCH_MOUNTS=(-v "$PATCH_DIR/patch_mtp_nightly.py:/patch_mtp.py:ro"
                -v "$PATCH_DIR/patch_mtp_boundary.py:/patch_boundary.py:ro")
  PATCH_PRELUDE="python /patch_mtp.py; python /patch_boundary.py; "
fi

SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_N}}"
SPEC_ARG="--speculative-config '${SPEC}'"
if [ "$MTP_N" = "0" ]; then SPEC_ARG=""; fi
LMO_ARG=""
if [ "$LANGUAGE_MODEL_ONLY" = "1" ]; then LMO_ARG="--language-model-only"; fi
RENDER_GROUP=$(stat -c '%g' /dev/dri/render* | sort -u | head -1)

# docker silently creates missing bind-mount sources as an empty root-owned dir,
# which only surfaces later as vLLM "no config.json" -> fail before that happens
if [ ! -f "$MODEL/config.json" ]; then
  echo "FAIL: MODEL=$MODEL has no config.json" >&2
  exit 1
fi

RUN_ID=$(date +%F_%H%M%S)
OUT="$SCRIPT_DIR/artifacts/${RUN_ID}-${NAME}-seqs${MAX_NUM_SEQS}"
mkdir -p "$OUT"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# A container under a different name on the same port would make the readiness
# poll below succeed against the wrong server (matters for PATCHES=0 bisects).
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "FAIL: port ${PORT} already in use by another server" >&2
  exit 1
fi

docker run -d --name "$NAME" --network host --ipc host \
  --device /dev/dri --group-add "$RENDER_GROUP" \
  -v /dev/dri:/dev/dri:ro \
  -v "$TRITON_CACHE:/workspace/triton_cache" \
  -e TRITON_CACHE_DIR=/workspace/triton_cache \
  -v "$MODEL:/model:ro" \
  "${PATCH_MOUNTS[@]}" \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  "set -e; ${PATCH_PRELUDE}exec vllm serve /model \
    --quantization gptq --dtype float16 --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_UTIL} --kv-cache-dtype fp8 --port ${PORT} \
    --max-num-seqs ${MAX_NUM_SEQS} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --enable-prefix-caching \
    --served-model-name ${SERVED_NAME} ${LMO_ARG} \
    ${SPEC_ARG} ${EXTRA_ARGS:-} \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml" \
  >/dev/null

{
  echo "run_id=$RUN_ID"
  echo "date=$(date -Is)"
  echo "image=$IMAGE"
  echo "name=$NAME"
  echo "port=$PORT"
  echo "max_num_seqs=$MAX_NUM_SEQS"
  echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "gpu_memory_utilization=$GPU_UTIL"
  echo "mtp_num_speculative_tokens=$MTP_N"
  echo "language_model_only=$LANGUAGE_MODEL_ONLY"
  echo "served_model_name=$SERVED_NAME"
  echo "spec=$SPEC"
  echo "extra_args=${EXTRA_ARGS:-}"
  echo "patches=$PATCHES"
} | tee "$OUT/launcher.out"

deadline=$((SECONDS + READY_TIMEOUT))
while ! curl -sf "http://127.0.0.1:${PORT}/v1/models" -o "$OUT/v1_models.json" 2>/dev/null; do
  if ! docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; then
    echo "FAIL: container exited before becoming ready" | tee -a "$OUT/launcher.out"
    docker logs "$NAME" > "$OUT/server.log" 2>&1 || true
    docker inspect "$NAME" > "$OUT/docker-inspect.json" 2>&1 || true
    echo "OUT=$OUT"
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "FAIL: not ready within ${READY_TIMEOUT}s" | tee -a "$OUT/launcher.out"
    docker logs "$NAME" > "$OUT/server.log" 2>&1 || true
    echo "OUT=$OUT"
    exit 1
  fi
  sleep 5
done

echo "ready after ${SECONDS}s -> http://127.0.0.1:${PORT}/v1" | tee -a "$OUT/launcher.out"
docker logs "$NAME" 2>&1 | grep -m1 "non-default args" > "$OUT/non-default-args.txt" || true
{
  docker exec "$NAME" /opt/venv/bin/pip show vllm-xpu-kernels 2>/dev/null | head -2 || true
  docker exec "$NAME" /opt/venv/bin/pip show vllm 2>/dev/null | head -2 || true
  docker exec "$NAME" sha256sum /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py 2>/dev/null || true
  docker exec "$NAME" grep -c B70_GDN_SPLIT_DISPATCH /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py 2>/dev/null || true
} > "$OUT/versions.txt" 2>&1
docker inspect "$NAME" > "$OUT/docker-inspect.json" 2>&1 || true

echo "OUT=$OUT"
