# intel-arc-b70-quant

GPTQ-INT4 (W4A16, group-128, symmetric) quantization of **Swift-Qwen3.8-27b** for
vLLM on an **Intel Arc Pro B70** (XPU), including a reproducible path from raw
BF16 weights to a served, tool-calling endpoint.

Target contract matches the community reference artifact
`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, which was itself produced with
`gptqmodel 7.3.2` on a B70.

## Result

| | |
|---|---|
| Source model | `ukisai/Swift-Qwen3.8-27b` (Qwen3_5ForConditionalGeneration, 52 GB BF16, 18 shards; local copy used here at `/opt/models/Swift-Qwen3.8-27b`, override with `MODEL_ID`) |
| Output | `Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (19 GB, 5 shards) |
| Quant contract | bits=4, group_size=128, sym=true, desc_act=false, format=gptq |
| Preserved unquantized | 15 `mtp.*` tensors (BF16 draft head), 333 `model.visual.*` tensors |
| Quantized modules | 400 (all 64 `model.language_model.layers.*` linear projections) |
| Quantizer | `gptqmodel:7.3.2` on XPU, 256 calibration samples, 2048-token cap |
| Wall time | **2.40 h** on the Arc Pro B70 |

`quantize_config.json` is field-for-field identical to the reference artifact
(bits/group_size/sym/desc_act/format/method/dynamic/lm_head/pack_dtype and the
`meta.quantizer` string). The only delta is `meta.offload_to_disk`, which is a
quant-time setting with no runtime effect.

Verified with `verify_quant.py` (see [Verification](#4-verification)).

> **Runtime status (2026-09-21).** Serving defaults moved to the stock
> `vllm/vllm-openai-xpu:nightly` (vLLM `0.29.1rc1.dev422`) with the two
> `patches/` applied at container start. That build no longer needs the GDN
> split-dispatch backport: mixed MTP+prefill batches and a 12-request storm pass
> at `--max-num-seqs 4` with MTP-3 (`bench_mixed.py` s1/s2/s3, 0 errors).
> `patch_mtp_boundary.py` **is still required** — an unpatched nightly dies with
> `EngineDeadError` when a spec-decode request runs to exactly `--max-model-len`.
> The derived image `vllm-xpu-gdn-split:0.1.12.3-p1` stays as a legacy fallback
> (`IMAGE=vllm-xpu-gdn-split:0.1.12.3-p1 ./launch.sh`).

## 1. Environment

Host: Ubuntu, Python 3.14.4, i9-13900K, 64 GB RAM, 1x Intel Arc Pro B70 (30.3 GiB VRAM).

Two virtualenvs are used on purpose:

### `.venv` - CPU/dev (smoke tests, verification)

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install gptqmodel==7.5.0 datasets==5.0.1 transformers==5.17.0
```

### `.venv-xpu` - quantization on the B70

```bash
python3 -m venv .venv-xpu
.venv-xpu/bin/pip install --upgrade pip
.venv-xpu/bin/pip install torch==2.9.1+xpu torchvision==0.24.1+xpu \
    --index-url https://download.pytorch.org/whl/xpu
.venv-xpu/bin/pip install gptqmodel==7.3.2 datasets==5.0.1 transformers==5.17.0
```

Resulting stack: `torch 2.9.1+xpu`, `torchvision 0.24.1+xpu`, `gptqmodel 7.3.2`,
`datasets 5.0.1`, `transformers 5.17.0`, `numpy 2.5.2`, Python 3.14.

Pinned inputs: source model revision `048328f4059015b63f860a453bf94834af0db683`;
calibration dataset revision `8049631c405ae6576f93f445c6b8166f76f5505a`.

Two hard-won constraints:

- **Do not install `torchao`.** `gptqmodel` only imports it lazily for FP4/NVFP4
  paths, so it is not needed for GPTQ. Every `torchao` build available for
  Python 3.14 either fails to import against `torch 2.9.1`
  (`AttributeError: 'typing.Union' object has no attribute '__module__'`) or is
  built for a newer torch.
- **Do not use the torch XPU nightly.** `torch 2.15.0.dev*+xpu` has broken
  boolean-mask indexing (`x[mask]` returns an empty tensor for any mask), which
  aborts inside `gptqmodel`'s quantizer at `xmin[tmp] = -xmax[tmp]`. Stable
  `2.9.1+xpu` behaves correctly.

Sanity check:

```bash
.venv-xpu/bin/python -c "
import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0))"
# 2.9.1+xpu True Intel(R) Arc(TM) Pro B70 Graphics
```

## 2. API notes (gptqmodel 7.3.2 / 7.5.0)

The original draft script for this model failed four ways; these are the fixes:

| Draft | Correct |
|---|---|
| `from gptqmodel.adapter import FORMAT` | `from gptqmodel.quantization.config import FORMAT` |
| `dynamic={..., "visual": {"bits": 16}}` | invalid - valid widths are `1..8`; `bits=16` raises `ValueError`. Use the negative-pattern exclusion `"-:.*mtp.*": {}` |
| `GPTQModel.from_pretrained(..., device_map="cpu", offload_buffers=True)` | `device_map`/`device` belong on `QuantizeConfig(device="xpu")`; in 7.5.0 they go through `model_init_kwargs` |
| vision/MTP dynamic entries | unnecessary: `Qwen3_5QModel.module_tree` only walks `model.language_model.layers`, so vision stays BF16 automatically, and `out_of_model_tensors = {"prefixes": ["mtp"]}` copies `mtp.*` verbatim |

Also: this model's `max_position_embeddings` is 262144, and gptqmodel does not
truncate calibration data, so a token-length cap is mandatory for a practical
run (`MAX_SEQ_LEN = 2048` in `quant_swift.py`).

`smoke_test.py` builds a tiny random `Qwen3_5ForConditionalGeneration` (2 layers,
small hidden size, real tokenizer/processor files) and runs the identical
load -> quantize -> save path, so the API surface is validated in ~2 minutes
instead of after loading 52 GB:

```bash
DEVICE=xpu SMOKE_DIR=_tiny_xpu .venv-xpu/bin/python smoke_test.py   # XPU
DEVICE=cpu .venv/bin/python smoke_test.py                           # CPU
```

Expected tail: `SMOKE TEST PASS`, `quantized (qweight) count: 13`,
`mtp keys: []` (the tiny model has no MTP head).

## 3. Quantize

```bash
cd "$(git rev-parse --show-toplevel)"   # repo root
DEVICE=xpu nohup .venv-xpu/bin/python -u quant_swift.py > quant_xpu.log 2>&1 &
```

`MODEL_ID` (source checkpoint) and `OUTPUT_DIR` are env-overridable; the defaults
are the paths used for this run.

What the script does:

1. `AutoProcessor`/tokenizer from the local model dir.
2. Calibration: `HuggingFaceH4/ultrachat_200k` `train_sft[:256]`, each example
   rendered with the model's chat template and truncated to 2048 tokens
   (308,709 tokens total, avg ~1.2k).
3. `QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
   format=FORMAT.GPTQ, device="xpu", dynamic={"-:.*mtp.*": {}})`.
4. `GPTQModel.from_pretrained(MODEL_ID, quantize_config=..., trust_remote_code=True)`
   - weights are streamed layer by layer, quantized results offloaded to disk
   (`offload_to_disk=True` default), so host RAM stays ~4-9 GB.
5. `model.quantize(calibration_dataset, batch_size=1)`, then
   `model.save(OUTPUT_DIR)` and `processor.save_pretrained(OUTPUT_DIR)`.

Progress and per-layer timing land in `quant_xpu.log`; gptqmodel also writes
`logs/gptq_log_*.log` inside the output dir. The run prints at the end:

```
mtp keys preserved: 15
visual keys: 333
quantized (.qweight): 400
done in 2.40 h
```

Reference per-module timings from this run are in `artifacts/quant_log.csv`.

## 4. Verification

```bash
.venv/bin/python verify_quant.py
```

```
bits         = 4        (want 4) OK
group_size   = 128      (want 128) OK
sym          = True     (want True) OK
desc_act     = False    (want False) OK
format       = 'gptq'   (want 'gptq') OK

dynamic      = {"-:.*mtp.*": {}}
mtp exclusion patterns: ['-:.*mtp.*']

mtp tensors preserved : 15 (quantized: 0)
visual tensors        : 333
lm_head quantized     : False
quantized modules     : 400

VERIFY PASS
```

The script fails loudly if MTP were quantized, if `mtp.*` were missing from the
checkpoint, or if any config field drifted.

## 5. Serve (vLLM XPU)

Pinned image from the cookbook recipe:
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
(`vllm 0.27.2rc1.dev77+gac7509e2b`, `vllm-xpu-kernels 0.1.12.3`).

Two runtime patches from
[`intel-arc-pro-b70-inference-cookbook`](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
are mounted and applied in order:

1. `patches/patch_mtp_nightly.py` - adds the `B70_MTP_BF16_DRAFT=1` gate so the
   MTP draft layers build unquantized (matching the preserved BF16 `mtp.*`).
2. `patches/patch_mtp_boundary.py` - handles the partial final speculative group
   at the 131,072-token boundary.

```bash
./launch.sh          # MODEL defaults to <repo>/Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

`launch.sh` (env-overridable: `IMAGE`, `MODEL`, `NAME`, `PORT`, `MAX_NUM_SEQS`,
`GPU_UTIL`, `MTP_N`, `SERVED_NAME`, `LANGUAGE_MODEL_ONLY`, `PATCH_DIR`,
`TRITON_CACHE`) runs the container with host networking (so `/metrics` is
reachable from the LAN without Docker DNAT in the path), `--ipc host`,
`/dev/dri` + render group, and a shared Triton cache (default
`/opt/models/triton_cache` -> `TRITON_CACHE_DIR`), which cuts startup from
~190 s to ~150 s. Paths default to the script's own directory, so a plain clone
works. It removes any container with the same `NAME` first, waits for
`/v1/models`, and writes run evidence to `artifacts/<run-id>/`.

Serve flags (defaults):

```
--quantization gptq --dtype float16 --max-model-len 131072
--gpu-memory-utilization 0.94 --kv-cache-dtype fp8
--max-num-seqs 4 --max-num-batched-tokens 16384 --enable-prefix-caching
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Defaults target the stock nightly (`vllm/vllm-openai-xpu:nightly`) with the
runtime patches from `patches/` applied at container start. Legacy derived
image: `IMAGE=vllm-xpu-gdn-split:0.1.12.3-p1 ./launch.sh`; unpatched pinned base:
`IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff…' PATCHES=0 MAX_NUM_SEQS=1 ./launch.sh`.
Port 8080, one server at a time.

### Serving appendix: `--max-num-seqs > 1` (MTP + GDN fix)

> Historical note (2026-09-21): superseded for daily serving — vLLM `0.29.1`
> nightly passes the same concurrency harness without the backport (see the
> status note at the top). This section documents the original workaround for
> the pinned `0.1.12.3` stack; the derived image remains available as a
> fallback, and `patch_mtp_boundary.py` is still used on nightly.

MTP speculative decoding under concurrency used to kill the engine on XPU
(mixed spec-decode + prefill batches hit a kernel guard). The upstream fix
(`vllm-xpu-kernels` #537, shipped in 0.1.14.1) is not in our pinned base
(0.1.12.3), and it only travels with vLLM PR #48109 plus new op bindings — so
instead of rebasing the runtime it is backported python-only into a derived
image, with its own launcher and a concurrency harness (rationale for that
trade-off over a kernel bump: MAX_NUM_SEQ_FIX.md section 2.3).

```bash
docker build -f docker/Dockerfile.gdn-split -t vllm-xpu-gdn-split:0.1.12.3-p1 .
./launch.sh                                                                    # serve (defaults to the patched image)
./run_ab_matrix.sh A2-split-seqs8 vllm-xpu-gdn-split:0.1.12.3-p1 8 s1,s2,s3 3   # launch -> bench -> evidence
```

Verified at `--max-num-seqs` 1/8/16 (the unpatched base crashes on the first
mixed batch). Root cause, upstream links, image recipe, evidence and measured
numbers are in [MAX_NUM_SEQ_FIX.md](MAX_NUM_SEQ_FIX.md); the pinned base image
stays untouched and is reachable with `IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6…'
MAX_NUM_SEQS=1 ./launch.sh`. `LANGUAGE_MODEL_ONLY=0` loads the vision tower too
(untested — see the model card).

Container start is ~2.5-3 min (weights 17.4 GiB + graph capture). On success the
engine log contains:

```
[B70] MTP draft: forcing unquantized build (env B70_MTP_BF16_DRAFT=1)
```

Launcher knobs:

- `MTP_N` selects speculative tokens; drop `--gpu-memory-utilization` to 0.90 for
  no-spec runs.

Endpoints: `http://127.0.0.1:8080/v1` (model id `swift38`); use host IP for remote
clients. Prometheus metrics at `/metrics`.

## 6. Client checks

```bash
.venv/bin/python test_serve.py    # correctness + streaming smoke test
.venv/bin/python bench_mtp4.py    # p512/g128, n=5, median post-first decode
```

`test_serve.py` sends a non-streaming and a streaming chat completion and fails
if either returns nothing.

`bench_mtp4.py` reproduces the cookbook measurement method (client post-first
rate at p512/g128, median of 5). Measured on this host:

| Model | decode (p512/g128, n=5) | mean acceptance length | draft acceptance |
|---|---|---|---|
| Swift-Qwen3.8-27b (this repo) | 58.8 tok/s | 3.38 | 59.5% |
| Qwen3.8-27B reference artifact | 60.4 tok/s | 3.43 | 60.9% |

Raw decode tok/s is not the metric that matters for an agent workload - Swift
solves tasks in fewer tokens, so end-to-end cost per task is lower than the
per-token rate suggests.

Tool calls (parsed server-side by `qwen3_xml`):

```
finish_reason: tool_calls
TOOL CALL PARSED: [{"name": "get_weather", "args": "{\"city\": \"Paris\"}"}]
```

## 7. Files

```
quant_swift.py            quantization run (XPU by default via DEVICE env)
verify_quant.py           post-quant contract verification
smoke_test.py             tiny-model API/pipeline smoke test (CPU or XPU)
launch.sh                 vLLM XPU container launcher (env-overridable);
                          defaults to the split-dispatch image, MAX_NUM_SEQS=4,
                          GPU_UTIL=0.94
MAX_NUM_SEQ_FIX.md        GDN split-dispatch fix: analysis, image recipe,
                          launcher/harness usage and measurements (serving
                          appendix; not part of the quantization path)
test_serve.py             endpoint smoke test
bench_mtp4.py             p512/g128 n=5 decode benchmark
bench_mixed.py            mixed spec-decode/prefill concurrency harness
run_ab_matrix.sh          launch -> bench -> capture runner (A/B cells)
patches/patch_mtp_nightly.py
                          MTP draft built unquantized (B70_MTP_BF16_DRAFT=1)
patches/patch_mtp_boundary.py
                          max-len speculative-group boundary fix
patches/patch_gdn_split.py + gdn_split_dispatch.diff
                          GDN split-dispatch backport applier (see
                          MAX_NUM_SEQ_FIX.md)
patches/README.md         patch provenance, hashes, how they are applied
docker/Dockerfile.gdn-split
                          derived image with the split-dispatch patch
artifacts/quant_log.csv   per-module quantization timings (full run)
hf/MODEL_CARD.md          Hugging Face model card (publish prep)
hf/UPLOAD_PLAN.md         upload plan, licence compliance + disclosure checklist
```

## 8. Reproduce

```bash
# 1. envs
python3 -m venv .venv-xpu
.venv-xpu/bin/pip install torch==2.9.1+xpu torchvision==0.24.1+xpu --index-url https://download.pytorch.org/whl/xpu
.venv-xpu/bin/pip install gptqmodel==7.3.2 datasets==5.0.1 transformers==5.17.0

# 2. validate API on a tiny model
DEVICE=xpu SMOKE_DIR=_tiny_xpu .venv-xpu/bin/python smoke_test.py

# 3. quantize (~2.4 h on Arc Pro B70)
DEVICE=xpu .venv-xpu/bin/python -u quant_swift.py

# 4. verify
.venv/bin/python verify_quant.py

# 5. serve
./launch.sh
.venv/bin/python test_serve.py
```

## 9. License

Repository code (quantization scripts, launchers, harness, patch appliers,
docs) is **MIT** — see [LICENSE](LICENSE).

The quantized model is **not** covered by the MIT licence and is not stored in
this repository (19 GB, gitignored). Its licence chain is the one of the source
checkpoint and is mirrored in [`hf/`](hf/) for the Hugging Face upload:

- Swift contribution (the fine-tuned weights):
  [Swift Open License v1.0](hf/LICENSE) — free for individuals and organisations
  up to US$1,000,000 gross annual revenue, above which commercial use needs a
  separate Swift Enterprise License from UkisAI.
- Qwen3.8-27B base: [Apache-2.0](hf/LICENSE-APACHE-2.0).
- [`hf/NOTICE`](hf/NOTICE) records what UkisAI changed relative to the base.

Third-party material used here keeps its own licence:

- `patches/gdn_split_dispatch.upstream.md` / `.diff` — patch and commentary from
  the vLLM project issue tracker (Apache-2.0), linked in
  [MAX_NUM_SEQ_FIX.md](MAX_NUM_SEQ_FIX.md) section 2.
- `patches/patch_mtp_nightly.py` / `patches/patch_mtp_boundary.py` — vendored
  verbatim from `SergiioB/intel-arc-pro-b70-inference-cookbook` (MIT,
  Copyright (c) 2026 SergiioB) at commit `3beb704b`; see
  [`patches/README.md`](patches/README.md) for hashes and purpose.
- `gptqmodel` (used for quantization) and the vLLM XPU image are external
dependencies under their own licences.
