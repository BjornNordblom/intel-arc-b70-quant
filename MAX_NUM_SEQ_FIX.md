# MAX_NUM_SEQ_FIX — restore `--max-num-seqs > 1` for MTP + GDN on Arc Pro B70

Status: **EXECUTED 2026-09-17** — plan below, execution log in section 15.
Outcome: the crash is fixed and verified at `--max-num-seqs` 1/8/16 with a
Python-only backport baked into a derived image; no changes were made to the
`swift-b70-mtp` container, its image, or its launch behavior.

Owner: Bjorn
Related: `launch.sh`, container `swift-b70-mtp-split`, pinned base image digest
`f01e24f6c7ff…`

> Sections 1–12 are the plan as written before execution (kept for reasoning
> and review). Section 14 holds the measurements, section 15 the executed work
> and how it was done.

---

## 0. Quickstart — reproduce it

One launcher, `launch.sh`, binding **port 8080** (one B70 fits one server):

| image (default) | `--max-num-seqs` (default) | purpose |
|---|---|---|
| `vllm-xpu-gdn-split:0.1.12.3-p1` | 4 | serving with the mixed-batch fix |

To run the unpatched pinned baseline instead:
```bash
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f' \
  MAX_NUM_SEQS=1 ./launch.sh
```

Text-only by default (`LANGUAGE_MODEL_ONLY=1`). `LANGUAGE_MODEL_ONLY=0` drops
`--language-model-only`, which loads the vision tower as well — see section 11.1
for what vision would still need; it is not validated.

> History: until late 2026-09-17 there were two launchers — `launch_swift_mtp.sh`
> (pinned base, seqs 1) and `launch_swift_mtp_max_num_seq.sh` (patched image,
> `MAX_NUM_SEQS`). They were consolidated into `launch.sh`. The A0–A3
> measurements in section 14 were taken with the two-launcher setup and stay
> valid; the current defaults are `MAX_NUM_SEQS=4`, `GPU_UTIL=0.94`.

```bash
# 1. build the patched image (~3 s, layers on top of the pinned base)
docker build -f docker/Dockerfile.gdn-split -t vllm-xpu-gdn-split:0.1.12.3-p1 .

# 2. run it (waits for /v1/models, records artifacts/<run-id>/, then exits)
./launch.sh

# 3. or run a full measured cell: launch -> s1/s2/s3 bench -> evidence -> teardown
./run_ab_matrix.sh A2-split-seqs8 vllm-xpu-gdn-split:0.1.12.3-p1 8 s1,s2,s3 3
KEEP=1 ./run_ab_matrix.sh A2-split-seqs8 vllm-xpu-gdn-split:0.1.12.3-p1 8 s1,s2,s3 3  # leave it running
```

Confirm the fix is in the running container:

```bash
docker exec swift-b70-mtp-split sha256sum /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py
# 1b69d2b96bb99a3d4b153148af4bb9937abca8cf4a2fb1deac27864018894808
docker exec swift-b70-mtp-split grep -c B70_GDN_SPLIT_DISPATCH /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py
# 1
```

Expected (measured 2026-09-17, MTP 3, seqs 8): solo TTFT ~0.95 s, decode
~59 tok/s, storm 12/12 HTTP 200, MTP acceptance ~62%, no `mutually exclusive`
in the log. The unpatched base with `--max-num-seqs 8` dies on the first mixed
step — that is the bug being fixed.

Evidence per run lives in `artifacts/<run-id>/` (gitignored): `bench.json`,
`server.log`, `non-default-args.txt`, `versions.txt`, `cell.txt`.

Decision: use chunk budget `16384` for 128k-context workloads; §14.2 records the 8192 comparison.

---

## 1. Goal

Test a fix for the mixed-batch crash so MTP speculative decoding can run with
`--max-num-seqs > 1` again, **without touching**:

- the running container `swift-b70-mtp` (user is actively using it),
- the known-good image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
- the existing launcher script's behavior for that image.

Deliverables of this plan:

1. A new derived image carrying the fix (easy rebuild/reinstall).
2. A parameterized launcher script for repeatable runs.
3. A concurrency test harness with pass/fail criteria.
4. An A/B evidence matrix recorded under `artifacts/`.

---

## 2. Root cause (recap, with sources)

Mixed invocation containing spec-decode tokens **and** non-spec (prefill/decode)
tokens is rejected by the XPU `gdn_attention` op:

```
RuntimeError: causal_conv1d does not support spec-decode and non-spec
(prefill + decode) tokens in the same invocation; the spec path and the
non-spec path are mutually exclusive
```

Guard string lives in `vllm_xpu_kernels/_xpu_C.abi3.so` (kernels 0.1.12.3);
called from `vllm/_xpu_ops.py:171` (`_gdn_attention_core_xpu_impl`).
vLLM python fully supports mixed metadata (`gdn_attn.py` partitions tokens via
`argsort` into `spec_token_indx` / `non_spec_token_indx`), so the failure is
kernel-side.

Sources (all read 2026-09-17):

- `vllm-xpu-kernels` issue **#510** — canonical, closed as fixed in 0.1.14 line:
  https://github.com/vllm-project/vllm-xpu-kernels/issues/510
- `vllm` issue **#53928** — same error, still open:
  https://github.com/vllm-project/vllm/issues/53928
- Fix PR `vllm-xpu-kernels` **#537** (merged 2026-08-19, ships in v0.1.14/0.1.14.1):
  https://github.com/vllm-project/vllm-xpu-kernels/pull/537
- Companion PR `vllm` **#48109** (merged 2026-08-19, mamba u64→i64 pointer fix):
  https://github.com/vllm-project/vllm/pull/48109
- Python-only workaround patch (op-level split dispatch), 185 lines:
  https://github.com/vllm-project/vllm-xpu-kernels/issues/510#issuecomment-5301844645

### 2.1 Our build is pre-fix on both sides

| component | ours | fix shipped in |
|---|---|---|
| `vllm-xpu-kernels` | `0.1.12.3` | `0.1.14.1` (PyPI 2026-08-28), latest `0.1.15.1` (2026-09-17) |
| vLLM | `0.27.2rc1.dev77+gac7509e2b.xpu` (files dated 2026-08-14) | needs #48109 (merged 2026-08-19) |

Verified in the running container: `_reinterpret_u64_as_i64` is **absent** from
`vllm/v1/worker/mamba_utils.py` → PR #48109 not included.

### 2.2 Why the Python workaround is the first thing to test

Independently validated on **our exact stack** (`vLLM 0.27.2rc1.dev77+gac7509e2b.xpu`
+ `vllm-xpu-kernels 0.1.12.3` + Qwen3.8-27B GPTQ-Int4 G128 + BF16 MTP head +
MTP4 + Arc Pro B70):

| scenario | before patch | after split-dispatch patch |
|---|---|---|
| solo decode | 33.6 t/s (or crash with 2nd req) | 42.4 t/s |
| staggered 2nd request (2 s) | `RuntimeError` | 5/5 pass |
| 12-request storm | 151.8 t/s aggregate | 206.5 t/s aggregate (+36%) |
| 9.6K prefill + 3 decoders | crash on mixed step | 4/4 pass |
| MTP acceptance | — | 3.14 / 4 tokens |

Also proven: `num_speculative_tokens_per_batch_size=[[1,1,4],[2,64,0]]`
scheduler throttling does **not** prevent the crash, because in-flight draft
verification cannot be drained per scheduler step. Only op-level split works.

Remaining upstream caveat: `vllm-xpu-kernels` #54740 / PR #599 (ragged **n-gram**
draft lengths) is still open on 0.1.14.1 — n-gram only, does not affect MTP.

---

### 2.3 Why a Python backport instead of just bumping the kernels

The proper upstream fix is in `vllm-xpu-kernels` ≥ 0.1.14.1 (PR #537,
"add new split op and refactor logic"), but it is only half of the pair: the same
maintainer thread names vLLM PR **#48109** as required as well. On our pinned
base neither condition holds, which is why the Python backport was chosen:

1. **Our vLLM predates the companion fix.** Base is
   `0.27.2rc1.dev77+gac7509e2b.xpu` (image files dated 2026-08-14); #48109 merged
   2026-08-19 and is absent — verified in the running container:
   `_reinterpret_u64_as_i64` is not present in `vllm/v1/worker/mamba_utils.py`.
   That fix addresses an XPU pointer-overflow in the mamba **align** mode, which
   our config uses (prefix caching on). A kernels-only bump would apply half the
   fix.
2. **PR #537 changed the op bindings.** Diffstat: `csrc/xpu/ops.h +51/-14`,
   `csrc/xpu/torch_bindings.cpp +34/-15`, plus the new split op. An older vLLM
   may call `gdn_attention` with the pre-change signature, so kernel and vLLM
   versions must move together — hence the ABI gate in section 9.
3. **Wheel ↔ torch coupling.** `vllm-xpu-kernels` ships compiled C++ extensions;
   our container is torch `2.13.0+xpu`. A wheel built against a different torch
   XPU ABI is not guaranteed to load.
4. **Release timing.** When this work was planned the newest release was 0.1.14.1
   (2026-08-31) against a vLLM from 2026-08-14; 0.1.15.1 only appeared on
   2026-09-17. The gap between the two halves is inherent, not incidental.
5. **Blast radius vs. evidence.** The Python backport was independently
   validated on exactly this stack (same vLLM hash, kernels 0.1.12.3,
   Qwen3.8-27B GPTQ-Int4, Arc Pro B70; 12-request storm at `--max-num-seqs 16`).
   The kernels route implies a new base image, re-adapting both MTP patches whose
   anchors drift upstream (`patch_mtp_nightly.py` targets `qwen3_5_mtp.py`,
   `patch_mtp_boundary.py` targets `gdn_attn.py`), re-verifying weight/config
   parsing, and re-running the whole matrix — while moving the runtime stack out
   from under the serving baseline that is known-good today.

So the accurate statement is: *kernels ≥ 0.1.14.1 remove the guard, provided the
vLLM side also carries #48109 and matches the new bindings.* For an
August-14 vLLM + 0.1.12.3 kernels, one Python function is the smaller and
reversible change. Section 9 keeps the kernels route available as an isolated
experiment rather than a forced migration.

## 3. Non-negotiables

1. `swift-b70-mtp` container and its image are never removed, renamed or
   re-tagged. New work uses new names/tags only.
2. One B70 → only one vLLM server can hold VRAM at a time. Test runs happen in
   a window where production is stopped; the pinned known-good base stays
   reachable via `IMAGE=… MAX_NUM_SEQS=1 ./launch.sh`.
3. Ports: one server on `8080` at a time — the single GPU means only one fits
   anyway.
4. All patch application must fail loudly (non-zero exit) if the target file
   hash or anchors do not match. No best-effort patching.

Recorded hashes of the **unmodified** pinned image (for patch appliers):

```
47080ce161348db461ac6777f818742a067ef6db5cb64a67165864fbad21d3e1  vllm/_xpu_ops.py
fda86b96ab5daaf50bd02d022518779c220401dbedc7b28cf478f4c48e72d3d3  vllm/v1/attention/backends/gdn_attn.py
08a83fa1f6bd76fee2e0567d8d440fc05191cbb7a019c1a797ccefb63f51346d  vllm/model_executor/models/qwen3_5_mtp.py
vllm-xpu-kernels 0.1.12.3
```

Existing runtime patches (keep as-is, mounted at launch exactly like today):

- `patch_mtp_nightly.py` → patches `model_executor/models/qwen3_5_mtp.py`
  (env gate `B70_MTP_BF16_DRAFT=1`), marker `B70_MTP_NIGHTLY_DRAFT`
- `patch_mtp_boundary.py` → patches `v1/attention/backends/gdn_attn.py`,
  marker `B70_MTP_PARTIAL_FINAL_GROUP`

---

## 4. Phase 0 — offline patch validation (no GPU, no container)

Purpose: decide whether the upstream split patch fits **our** `_xpu_ops.py`
before spending time on an image build. This phase is deliberately GPU-free —
the single B70 stays with `swift-b70-mtp`.

Runtime validation is intentionally **not** done here: A1/A2 (section 8) already
push the baked image through the real server, so a separate GPU smoke run would
duplicate the same test and cost a second test window.

```bash
# 1. vendor the upstream patch comment (provenance)
mkdir -p patches
curl -sL "https://api.github.com/repos/vllm-project/vllm-xpu-kernels/issues/comments/5301844645" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["body"])' > patches/gdn_split_dispatch.upstream.md

# 2. extract the module from the pinned image (no GPU flags, no VRAM)
mkdir -p _work && docker run --rm --entrypoint cat \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py > _work/_xpu_ops.py
sha256sum _work/_xpu_ops.py   # must equal 47080ce1… (pin)

# 3. run the anchor-based applier against the extracted file (host python is fine)
python3 patches/patch_gdn_split.py --target _work/_xpu_ops.py
sha256sum _work/_xpu_ops.py   # patched hash, record it in the run log
python3 -c "import ast,pathlib; ast.parse(pathlib.Path('_work/_xpu_ops.py').read_text())"
grep -c B70_GDN_SPLIT_DISPATCH _work/_xpu_ops.py
```

Exit criteria: applier exits 0, patched hash is stable and recorded, file parses,
marker present. If any of that fails, either fix the applier or go straight to
Phase 2 (kernels upgrade) — do not build the image on a failed patch.

Iteration note: if A2 fails and you need to edit the patch text quickly, the
hot-mount trick (bind-mounting a single patched `_xpu_ops.py` over
site-packages) is still useful — but that is an in-window iteration tool, not a
pre-step. Precedent: upstream reporters validated the patch that way before a
rebuild ever happened.

---

## 5. Phase 1 — derived image with the fix baked in

### 5.1 Repo layout (new files)

```
quant-swift/
├── MAX_NUM_SEQ_FIX.md                  # this plan
├── docker/
│   ├── Dockerfile.gdn-split            # Phase 1 image
│   └── Dockerfile.kernels-upgrade      # Phase 2 image (section 9)
├── patches/
│   ├── gdn_split_dispatch.upstream.md  # vendored upstream comment (provenance)
│   ├── gdn_split_dispatch.diff         # extracted diff, anchor-based
│   └── patch_gdn_split.py              # applier: hash check + anchor replace + compile check
├── launch.sh                           # launcher (section 6)
├── bench_mixed.py                      # concurrency harness (section 7)
├── run_ab_matrix.sh                    # orchestrates A0..A3 (section 8)
└── artifacts/<run-id>/                 # per-run evidence (gitignored)
```

Add to `.gitignore`:

```
_work/
artifacts/
```

### 5.2 Dockerfile sketch

```dockerfile
# docker/Dockerfile.gdn-split
FROM vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f

# provenance
LABEL b70.patch="gdn split dispatch (vllm-xpu-kernels#537 backport)"
LABEL b70.upstream="https://github.com/vllm-project/vllm-xpu-kernels/issues/510#issuecomment-5301844645"

COPY patches/gdn_split_dispatch.diff /opt/b70/patches/
COPY patches/patch_gdn_split.py      /opt/b70/patches/

# fail loudly if the base file hash/anchor drift
RUN /opt/venv/bin/python /opt/b70/patches/patch_gdn_split.py \
 && /opt/venv/bin/python -m py_compile /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py \
 && grep -q B70_GDN_SPLIT_DISPATCH /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py
```

Build/tag:

```bash
docker build -f docker/Dockerfile.gdn-split \
  -t vllm-xpu-gdn-split:0.1.12.3-p1 .
docker image inspect vllm-xpu-gdn-split:0.1.12.3-p1 --format '{{.Id}}'
```

### 5.3 Patch applier requirements (`patches/patch_gdn_split.py`)

1. Assert `sha256(_xpu_ops.py) == 47080ce161348db461ac6777f818742a067ef6db5cb64a67165864fbad21d3e1`
   (the pin for kernels 0.1.12.3). Refuse to patch anything else.
2. Locate the single call block
   `torch.ops._xpu_C.gdn_attention(` … through its matching `)` at 4-space
   indent inside `_gdn_attention_core_xpu_impl`, and replace it with the split
   implementation (two dispatches: non-spec group, spec group).
3. The replacement must keep the current signature and in-place mutation
   semantics (`mutates_args=["core_attn_out", "z"]`).
4. Idempotency marker `B70_GDN_SPLIT_DISPATCH`; second run is a no-op.
5. After write: `py_compile`, re-read, assert marker present, print sha256 of
   the patched file for the run log.
6. Exit non-zero with a clear message on any failure (no partial writes).

Rationale for anchor-based replacement instead of raw `patch(1)`: the upstream
diff was authored against a newer `_xpu_ops.py` and lands at a large offset
(upstream report: "applied at offset −19 lines"); raw context matching is
brittle, the call-block anchor is exact.

Known pitfall to preserve from the upstream patch: it uses input compaction
(`index_select`) + local `torch.arange` indexing. A naive split without
compaction triggered an Inductor index-out-of-bounds at concurrency ≥ 4 — that
is exactly what `A2/A3` (seqs 8/16) must cover.

### 5.4 What is baked vs mounted

- **Baked into the image:** only the new `_xpu_ops.py` split patch.
- **Mounted at runtime, unchanged:** `patch_mtp_nightly.py`,
  `patch_mtp_boundary.py` (same as today) so the validated MTP behavior is
  bit-identical to `swift-b70-mtp`.

This keeps the image delta minimal and reviewable.

---

## 6. Launcher script (`launch.sh`)

Single entrypoint, consolidated from the earlier known-good + patched pair:

```bash
IMAGE=${IMAGE:-vllm-xpu-gdn-split:0.1.12.3-p1}   # override for the pinned base
NAME=${NAME:-swift-b70-mtp-split}
PORT=${PORT:-8080}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
GPU_UTIL=${GPU_UTIL:-0.94}
MTP_N=${MTP_N:-3}
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-1}     # 0 = also load the vision tower
```

It removes any container with the same `NAME` first, waits for `/v1/models`, and
records evidence under `artifacts/<run-id>/`.

Historical note: this section originally described
`launch_swift_mtp_max_num_seq.sh` as a copy of `launch_swift_mtp.sh` with
`MAX_NUM_SEQS` added (defaults then: seqs 8, util 0.92).

Differences in the `vllm serve` line: `--max-num-seqs ${MAX_NUM_SEQS}` only.

Additional requirements:

1. `docker rm -f "$NAME"` only — never any other name.
2. Wait-for-ready loop: poll `http://127.0.0.1:${PORT}/v1/models` until 200 or
   timeout (e.g. 300 s), then exit 0; on timeout dump `docker logs` and exit 1.
3. Write evidence into `artifacts/$(date +%F_%H%M%S)-${NAME}-seqs${MAX_NUM_SEQS}/`:
   - `launcher.out` (echo of full config),
   - `non-default-args.txt` (grep from container log — the authoritative record
     of what the runtime actually used),
   - `docker-inspect.json`,
   - `env.txt` (versions: vllm, vllm-xpu-kernels).
4. Teardown helper: `docker rm -f "$NAME"` only; keep logs before removal.

---

## 7. Test harness (`bench_mixed.py`)

Reuses the style of `bench_mtp4.py` (stdlib only, streaming SSE, TTFT measured
at first non-empty chunk, decode measured after first token).

Scenarios:

- **S1 solo**: single request, ~512-token prompt, 128 new tokens — TTFT + decode
  baseline (comparable to `bench_mtp4.py`).
- **S2 mixed (the bug trigger)**: request A: short prompt, `max_tokens=512`;
  after a 1.5 s stagger, request B: long prompt ~9–12K tokens (guaranteed
  chunked prefill while decoded chunks stay under the 16384-token budget) while
  A is decoding
  with draft tokens in flight. Repeat 5×.
- **S3 storm**: 12 concurrent requests (mixed short/long), aggregate throughput
  + zero non-200 responses.
- **S4 sweep**: S1+S2 for each `MAX_NUM_SEQS` in {1, 2, 4, 8, 16} — one
  container per value (or one container + `/metrics` confirmation; restart is
  cheap because the compiled-graph cache is warm via the mounted triton cache).

Per request record: HTTP status, TTFT (s), decode tok/s, completion tokens.
Aggregate: total tok/s, p50/p95 TTFT.
`/metrics` after each scenario: `vllm:spec_decode_num_accepted_tokens_total` and
draft-token counters → acceptance ratio.

**Abort criteria (any → FAIL, stop container, keep logs):**

1. container not running afterwards (`docker inspect … State.Running != true`),
2. server log matches `mutually exclusive`, `EngineDeadError`, `EngineCore encountered a fatal error`,
3. any non-200/stream error,
4. `Overflow when unpacking long long` (Phase 2 concern).

Output: `artifacts/<run-id>/bench.json` + human-readable table.

---

## 8. A/B matrix and acceptance

| id | image | `--max-num-seqs` | expected | purpose |
|---|---|---|---|---|
| A0 | pinned `f01e24f6` (no patch) | 8 | **crash** (`RuntimeError`, engine dead) | harness sanity: proves the harness reproduces the bug |
| A1 | `gdn-split:0.1.12.3-p1` | 1 | pass | regression check vs `swift-b70-mtp` behavior |
| A2 | `gdn-split:0.1.12.3-p1` | 8 | pass | primary success criterion |
| A3 | `gdn-split:0.1.12.3-p1` | 16 | pass, no Inductor OOB | stress / compaction coverage |
| B1 | `kernels-upgrade` (Phase 2) | 8/16 | pass | upstream route |

Success = A1–A3 all pass with the abort criteria above, and A2/A3 TTFT and
decode are not worse than the A1 baseline (record the delta; upstream reported
+36% aggregate under storm, ~42 t/s solo decode).

A0 is expected to fail — a failure there is a **pass** for the harness and must
be captured verbatim as evidence (it also re-confirms the root cause on our
hardware).

Estimated cost: build 2–5 min (small layers over a cached base), server start
~60–90 s (compiled-graph cache mounted), each A/B run 5–10 min.

---

## 9. Phase 2 — upstream route (kernels 0.1.14.1+ / 0.1.15.1)

Rationale for deferring this is in **section 2.3**: the kernels fix travels with
vLLM PR #48109 and changed op bindings, so it is a stack rebase rather than a
one-line version bump. Only attempt after Phase 1 conclusions, or if Phase 0
shows the Python patch does not fit our file.

A full rebase (if Phase 2 ever becomes the default) would additionally mean:
re-adapting `patch_mtp_nightly.py` (anchor inside `qwen3_5_mtp.py`) and
`patch_mtp_boundary.py` (anchor inside `gdn_attn.py`) to the newer upstream code,
re-checking that the quantized checkpoint still parses under the newer vLLM
config/weight-loading path, re-confirming XPU graph capture and the torch ABI,
and re-running the A0–A3 matrix — i.e. a new baseline, not a patch swap.

Facts:

- kernels fix is in `v0.1.14.1` (PyPI 2026-08-28) and `0.1.15.1`
  (PyPI 2026-09-17); both depend on the companion vLLM PR #48109.
- our vLLM predates #48109 → a kernels-only `pip install` inside the existing
  image is **not** sufficient; hence a new image.

Dockerfile sketch:

```dockerfile
FROM vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
RUN /opt/venv/bin/pip install --no-deps vllm-xpu-kernels==0.1.15.1   # or 0.1.14.1
COPY patches/patch_mamba_u64.py /opt/b70/patches/
RUN /opt/venv/bin/python /opt/b70/patches/patch_mamba_u64.py   # backport PR #48109 (helper + 2 call sites)
```

Gates:

1. **ABI gate** — PR #537 changed bindings (`csrc/xpu/ops.h`, `torch_bindings.cpp`).
   Before running a server, check that our `_xpu_ops.py` call still resolves
   against the new kernel op (arg names/count). Inside a real vLLM init:
   `torch.ops._xpu_C.gdn_attention._schemas`. If it mismatches, this route needs
   a newer vLLM (nightly wheel) instead of the backport — evaluate then.
2. **Harness gate** — A0-style run must pass at seqs 8/16.
3. **Pointer gate** — no `Overflow when unpacking long long`; this is what
   #48109 addresses, and our config uses prefix caching → mamba `align` mode,
   which is the code path involved.
4. Record `vllm-xpu-kernels` version from logs in the run artifacts.

If Phase 2 passes, it becomes the preferred long-term image (no python-only
patch to maintain); Phase 1 remains the faster verified fallback.

---

## 10. Rollback / restore

- Nothing in this plan deletes or re-tags the pinned image. Rollback for a run is
  `docker rm -f swift-b70-mtp-split`.
- Restoring the known-good base: `IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6…' \
  MAX_NUM_SEQS=1 ./launch.sh`.

---

## 11. Risks / open questions

1. **Patch drift** — upstream `_xpu_ops.py` moves; the hash pin makes the
   applier fail loudly rather than patch the wrong code. Re-vendor when the
   base image changes.
2. **Concurrency ≥ 4** — upstream hit an Inductor index-out-of-bounds with a
   naive split; A3 explicitly covers it. If A3 fails, report it back upstream on
   #510 (compaction fix is in the patch we vendor).
3. **XPU graph capture** — `gdn_attention_core_xpu` is registered with
   `eager_break_during_capture`, so the split runs eagerly; no capture variants
   needed. Verify in logs that capture still succeeds at seqs 8/16
   (`cudagraph_capture_sizes` grows with `max_num_seqs`; startup time
   increases accordingly).
4. **VRAM** — max_num_seqs 16 with `max-model-len 131072` raises KV pressure;
   keep `GPU_UTIL=0.92` fixed across A/B and watch for preemptions
   (`vllm:num_preemptions_total`).
5. **Scheduler throttle does not work** — `num_speculative_tokens_per_batch_size`
   was tested upstream and still crashed; do not spend time there.
6. `--max-num-batched-tokens` is `16384` in `launch.sh` for 128k-context work.
   The 8192 comparison had better 10k-prompt storm TTFT, but was not faster for
   the target long-context workload.

### 11.1 Vision / multimodal — present but disabled (not validated)

Both launchers pass `--language-model-only` by default (`LANGUAGE_MODEL_ONLY=1`),
which is what the measured runs used. The checkpoint itself is multimodal:
333 `model.visual.*` tensors, `vision_config` (27-layer ViT, hidden 1152, patch
16, merge 2, temporal 2), image/video token ids, image+video blocks in
`processor_config.json`, and `image_url` handling in the chat template.
`LANGUAGE_MODEL_ONLY=0` removes the flag; that is where the untested work starts:

- **Attention backend.** In text-only mode vLLM waives the mm requirement
  ("Disabled mm_prefix attention mode because multimodal inputs are
  configuration-disabled. Attention backends without mm_prefix support may now
  be selected."). With vision enabled, the full-attention layers require a
  backend that advertises `mm_prefix`; if the auto-selected XPU backend does not,
  one has to be pinned (`--attention-backend TRITON_ATTN` is the usual candidate).
  The GDN layers need no mask change (causal over everything), but the metadata
  builder must handle `scheduled_encoder_inputs` — untested with our two runtime
  patches, especially `patch_mtp_boundary.py`, which rewrites spec/non-spec
  classification in `gdn_attn.py`.
- **Limits and memory.** Needs `--limit-mm-per-prompt` (e.g. image=1, video=0),
  an mm-processor cache size, and a resolution cap: the shipped defaults allow
  images up to 16.7M pixels (~16k visual tokens each) and 768 video frames, which
  will exhaust a 30 GiB card once KV/mamba state and encoder activations are
  counted. Practical: `--mm-processor-kwargs '{"max_pixels":1048576}'`.
- **Artifact completeness.** Add `preprocessor_config.json` and
  `video_preprocessor_config.json` (present in the source checkpoint, not written
  by the quantizer) plus `vocab.json`/`merges.txt` for non-fast tokenizer tooling.
- **Upstream matrix.** The vLLM XPU supported-models table lists multimodal LMs
  including `Qwen/Qwen3.5-35B-A3B` (`Qwen3_5MoeForConditionalGeneration`, the MoE
  sibling of our dense `Qwen3_5`), but only at BF16 / Online FP8. Multimodal +
  GPTQ-INT4 on XPU is not an upstream-validated combination.
- **Validation that would be required.** Boot with seqs 1 and MTP off, confirm
  modality limits are non-zero and which attention backend was selected, send one
  `image_url` request, measure visual-token count, TTFT and peak VRAM, then
  re-enable MTP and the mixed-batch harness. For a quality claim, upstream
  reports ERQA 66.30% (Swift) vs 67.45% (base BF16) — an independent run would be
  needed before the model card says anything stronger than "structurally
  supported, not validated".

---

## 12. Execution order (checklist)

- [x] Phase 0 (GPU-free): vendored patch, extracted `_xpu_ops.py`, applier validated
      (hash pin + anchor + AST + py_compile, idempotent).
- [x] Phase 1: `patches/patch_gdn_split.py` + `docker/Dockerfile.gdn-split`; image
      `vllm-xpu-gdn-split:0.1.12.3-p1` built (patched file sha `1b69d2b9…`).
- [x] Launcher with wait-for-ready + artifact capture (now consolidated into
      `launch.sh`), both on port 8080; 8192 and 16384 were measured, with 16384
      selected for the target long-context workload (14.2).
- [x] Adopt decision: use chunk budget 16384 for 128k-context work; see 14.2 for
      the 8192 comparison.
- [ ] Phase 2 (optional): kernels-upgrade image + ABI gate + matrix at seqs 8/16.
- [x] Committed everything (patches, image recipe, launchers, harness, docs);
      artifacts stay gitignored.

---

## 13. Provenance

- Guard string inspected in `vllm_xpu_kernels/_xpu_C.abi3.so` (kernels 0.1.12.3).
- Call site `vllm/_xpu_ops.py:171`; op registration `vllm/_xpu_ops.py:1260`
  (`direct_register_custom_op`, `eager_break_during_capture`, `mutates_args=["core_attn_out","z"]`).
- Mixed-batch metadata support: `vllm/v1/attention/backends/gdn_attn.py`
  (`spec_token_indx` / `non_spec_token_indx` via `argsort`).
- Crash evidence from our own run (2026-09-17 06:52:30, `swift-b70-mtp4`,
  seqs=64, MTP4): scheduler dump shows one 5-token spec decode + one 6656-token
  prefill chunk in a single invocation.

---

## 14. Results (executed 2026-09-17)

Image under test: `vllm-xpu-gdn-split:0.1.12.3-p1`
(base `f01e24f6c7ff…`, kernels 0.1.12.3, vLLM `0.27.2rc1.dev77+gac7509e2b.xpu`,
patched `_xpu_ops.py` sha `1b69d2b96bb99a3d4b153148af4bb9937abca8cf4a2fb1deac27864018894808`,
marker count 1 — verified inside the running container).
All A0–A3 runs: MTP 3, `max-model-len 131072`, `max-num-batched-tokens 8192`,
`gpu-memory-utilization 0.92`, prefix caching on. (The chunk size was raised to
16384 afterwards — see 14.2.)

| cell | image | seqs | scenarios | verdict |
|---|---|---|---|---|
| A0 `base` | pinned `f01e24f6…` (unpatched) | 8 | s2 ×2 | **FAIL as expected** — engine dead on run 1 |
| A1 `split` | patched | 1 | s1, s2 | PASS |
| A2 `split` | patched | 8 | s1, s2 ×3, s3 | PASS |
| A3 `split` | patched | 16 | s1, s2 ×3, s3 | PASS |

A0 failure signature (harness sanity check, identical to the original incident):
```
num_scheduled_tokens={...-9d78a25e: 4, ...-a7a9e7d9: 8000}, total=8004
scheduled_spec_decode_tokens={...-9d78a25e: [-1, -1, -1]}
RuntimeError: causal_conv1d does not support spec-decode and non-spec
(prefill + decode) tokens in the same invocation ...
```
`mutually exclusive` count in server logs: A0 = 1, A2 = 0, A3 = 0.

Performance:

| metric | A1 (seqs 1) | A2 (seqs 8) | A3 (seqs 16) |
|---|---|---|---|
| s1 solo TTFT | 1.01 s | 0.97 s | 0.96 s |
| s1 solo decode | 59.6 tok/s | 58.2 tok/s | 59.6 tok/s |
| s2 prefill TTFT | 34.4 s (serialized behind decode) | 26.9–27.5 s | 27.0–27.5 s |
| s2 decode rate during prefill | 56.5 tok/s | ~14 tok/s (interleaved) | ~14 tok/s |
| s3 storm | not run | 12/12 HTTP 200, wall 142 s | 12/12 HTTP 200, wall 145 s |
| s3 TTFT p50 / max | — | 97.4 s / 134.0 s | 103.2 s / 136.8 s |
| s3 preemptions | — | 4 | 7 |
| MTP acceptance (s1/s2/s3) | — | 65% / 62% / 63% | 67% / 60% / 62% |

Readings:

- The split backport removes the crash at concurrency 2, 8 and 16, including the
  ≥4 regime where the naive split hit Inductor index-out-of-bounds. XPU graph
  capture still succeeds (PIECEWISE 17 sizes + FULL 9 sizes at seqs 16).
- MTP stays active across mixed steps (acceptance ~62%, unchanged from solo),
  i.e. the spec path is genuinely exercised while prefills run.
- Solo performance is unchanged (within noise) vs the seqs 1 baseline.
- Storm TTFT (p50 ~100 s at 8192 chunks) is queueing/chunk-budget dominated:
  four ~10k-token prefills serialise into 8192-token chunks. That is a separate
  tuning lever (raised to 16384 in 14.2) and not related to this fix.
- seqs 16 produced more preemptions (7 vs 4) in the storm; seqs 8 is the safer
  default for this GPU/KV budget.

Artifacts (gitignored):
```
artifacts/2026-09-17_080651-swift-b70-mtp-split-seqs8/   # A0 (crash)
artifacts/2026-09-17_080908-swift-b70-mtp-split-seqs8/   # A2
artifacts/2026-09-17_081538-swift-b70-mtp-split-seqs16/  # A3
artifacts/2026-09-17_082209-swift-b70-mtp-split-seqs1/   # A1
```
Each contains `bench.json`, `server.log`, `non-default-args.txt`, `versions.txt`,
`docker-inspect.json`, `cell.txt`.

Reproduce a cell:
```bash
./run_ab_matrix.sh A2-split-seqs8 vllm-xpu-gdn-split:0.1.12.3-p1 8 s1,s2,s3 3
KEEP=1 ./run_ab_matrix.sh A2-split-seqs8 vllm-xpu-gdn-split:0.1.12.3-p1 8 s1,s2,s3 3  # inspect container after
```

Not attempted: Phase 2 (kernels 0.1.14.1/0.1.15.1 + vLLM #48109). The Phase 1
backport is validated and sufficient on our pinned base.

### 14.2 Chunk-size change: `8192` -> `16384` (executed 2026-09-17)

C1/C2/C2b experiments used `--max-num-batched-tokens 16384` (round ~16k,
replacing the unaligned `16234` of the qw38speed reference); current `launch.sh`
uses the selected `16384` default and listens on **port 8080**.

Cells (patched image unless noted, seqs 8 unless noted):

| cell | config | result |
|---|---|---|
| C1 | pinned base, seqs 1, 16384 | PASS — solo 1.00 s / 59.6 tok/s |
| C2 | patched, seqs 8, 16384, s1+s2+s3 | PASS — p50 140.7 s, wall 160.1 s |
| C2b | patched, seqs 8, 16384, s3 only | PASS — p50 144.7 s, wall 156.6 s |

Comparison against the 8192 runs (A1/A2):

| metric | 8192 | 16384 |
|---|---|---|
| s1 solo TTFT / decode | 0.97 s / 58.2 tok/s | 0.93 s / 59.6 tok/s |
| s2 prefill TTFT (10k prompt, decode overlap) | 26.9–27.5 s | 26.9–27.4 s |
| solo 10k prefill TTFT (no concurrency, one-off runs) | 26.60 s | 26.70 s |
| s3 storm wall | 142.4 s | 160.1 s / 156.6 s |
| s3 TTFT p50 | 97.4 s | 140.7 s / 144.7 s |
| s3 TTFT max | 134.0 s | 151.8 s / 147.9 s |
| s3 preemptions | 4 | 10 |
| s3 MTP acceptance | 62.5% | 62.5% |

Reading: the round 16384 chunk is **not** a win on this workload.

- Solo and overlapped long-prefill TTFT are unchanged (prefill compute dominates
  over per-step overhead; a 10k prompt is 2 chunks at 8192 and 1 at 16384).
- Burst latency regresses ~45% (p50 97 s -> 141–145 s) and preemptions rise
  4 -> 10, because one step can now carry a full 10k-token prefill and
  two of them, monopolising steps and starving decodes for everyone queued
  behind; wall time rises ~10%.
- The fix itself is unaffected: no crashes, MTP acceptance identical.

Recommendation from this narrow 10k-prompt storm: `8192` improves burst
latency, but current target workload uses `16384` because it was not faster at
128k context. Switch per workload if concurrency latency becomes priority.

---

## 15. Execution log — what we did and how

All of this happened on 2026-09-17, on the single Arc Pro B70. The pre-existing
serving container (`swift-b70-mtp`, pinned base image) was stopped by the user
beforehand and was never removed, renamed, re-tagged or executed against by any
of this work; its launcher at the time (`launch_swift_mtp.sh`) kept the pinned
image and `--max-num-seqs 1`. Both launchers were later consolidated into
`launch.sh`.

### 15.1 Phase 0 — patch vendoring and offline validation (no GPU)

How:

1. `curl` the upstream fix comment (id `5301844645` on
   `vllm-xpu-kernels#510`) into `patches/gdn_split_dispatch.upstream.md`, and
   extract its fenced diff into `patches/gdn_split_dispatch.diff` (179 lines).
2. Extract the target module from the pinned image without touching the GPU:
   `docker run --rm --entrypoint cat <pinned-digest> .../vllm/_xpu_ops.py > _work/_xpu_ops.py`
   and check it against the recorded pin `47080ce1…`.
3. `patches/patch_gdn_split.py` then applies the diff **without** `patch(1)`:

   - parses the unified diff and rebuilds both sides, keeping context lines on
     both sides (the vendored hunk shares its closing `)` as a context line — a
     naive `+`/`-` split silently drops it and produces a syntax error);
   - refuses to run if the file hash is not the pin or if the old block does not
     appear exactly once;
   - AST-checks that `_gdn_attention_core_xpu_impl` exists and that after the
     replacement it contains the nested `_invoke` helper;
   - writes the file, runs `py_compile`, and restores the original bytes if
     compilation fails;
   - is idempotent via marker `B70_GDN_SPLIT_DISPATCH` and prints before/after
     hashes.

Result: `47080ce1…` -> `1b69d2b9…`, idempotent re-run prints `already patched`.

### 15.2 Phase 1 — derived image

How:

```bash
docker build -f docker/Dockerfile.gdn-split -t vllm-xpu-gdn-split:0.1.12.3-p1 .
```

- `.dockerignore` keeps the 19 GB checkpoint, venvs and artifacts out of the
  build context (build finished in ~3 s because only two small layers are added).
- `docker/Dockerfile.gdn-split` starts from the pinned digest, copies the diff +
  applier into `/opt/b70/patches`, then `RUN` applies the patch, `py_compile`s the
  module, `grep`s for the marker and prints the sha256 — the build fails if any
  of that is off.
- Resulting image: `sha256:1de2e2f6…`, containing `_xpu_ops.py` sha `1b69d2b9…`.
- Nothing else is baked: the MTP patches (`patch_mtp_nightly.py`,
  `patch_mtp_boundary.py`) are still bind-mounted and applied at container start,
  exactly as before, so MTP behavior is byte-identical to `swift-b70-mtp`.

### 15.3 Harness and launcher (how repeatability is achieved)

- `launch.sh` (then `launch_swift_mtp_max_num_seq.sh`) — the container recipe:
  model mount, `/dev/dri` + render group, Triton cache, env, runtime MTP patches,
  patched image, container name `swift-b70-mtp-split`, `MAX_NUM_SEQS`. It removes
  only its own container name, polls `GET /v1/models` until ready (die/timeout ->
  dump logs, exit 1), then records `artifacts/<date>-<name>-seqs<N>/`:
  `launcher.out`, `v1_models.json`, `non-default-args.txt` (the engine's own
  record of the flags it really ran with), `versions.txt` (kernels/vLLM version,
  patched-file sha, marker count), `docker-inspect.json`.
- `bench_mixed.py` — stdlib-only concurrency harness: `s1` solo, `s2` staggered
  prefill-while-decoding with unique long prompts (prefix caching cannot mask the
  prefill), `s3` 12-request storm. TTFT is first non-empty stream chunk; decode
  rate is measured after the first token. It refuses to trust a run if the
  container died, any request errored, or the server log matches
  `mutually exclusive`, `EngineDeadError`, `EngineCore encountered a fatal
  error`.
- `run_ab_matrix.sh` — one cell: launch -> bench -> capture full `server.log` +
  final `docker inspect` -> write `cell.txt` -> remove its own container unless
  `KEEP=1`.

### 15.4 Matrix execution

Commands (one cell each):

```bash
./run_ab_matrix.sh A0-base-seqs8      '<pinned digest>'                   8 s2 2
./run_ab_matrix.sh A1-split-seqs1     vllm-xpu-gdn-split:0.1.12.3-p1     1 s1,s2 1
./run_ab_matrix.sh A2-split-seqs8     vllm-xpu-gdn-split:0.1.12.3-p1     8 s1,s2,s3 3
./run_ab_matrix.sh A3-split-seqs16    vllm-xpu-gdn-split:0.1.12.3-p1    16 s1,s2,s3 3
```

Sequence and outcome:

1. **A0 first** as the harness sanity check: the unpatched pinned image crashed on
   `s2` run 1 with the exact production signature (one 4-token spec decode + one
   8000-token prefill chunk in a single invocation), engine dead. This is what
   proves the harness really reproduces the bug.
2. **A2** (the primary case) passed: 12/12 storm requests HTTP 200, engine alive,
   `mutually exclusive` count 0 in the log, MTP acceptance ~62%.
3. **A3** passed at seqs 16 — including the >=4 concurrency regime where the
   naive split hit Inductor index-out-of-bounds; XPU graph capture succeeded
   (PIECEWISE 17 sizes + FULL 9).
4. **A1** as the seqs-1 baseline for comparison.

Server ready time was ~120–126 s per cell, each cell ~4–6 min wall. Evidence
directories are listed in section 14 and are gitignored.

Cross-checks used to make sure the result was not accidental:

- `versions.txt` from inside the running container confirmed the patched module
  (`1b69d2b9…`, marker count 1) and kernels 0.1.12.3.
- MTP acceptance counters (`/metrics`) stayed ~62% during the mixed workloads, so
  the speculative path was genuinely active while prefills ran — the split path
  was exercised, not bypassed.
- In-s2 the decode request kept streaming while a 10k-token prefill ran
  (decode rate dropping to ~14 tok/s), i.e. the two token classes really were
  co-scheduled; in A1 (seqs 1) the same workload serialised instead.

### 15.5 Post-run changes

- Renamed via `git mv` (history preserved):
  `launch_swift_mtp4.sh` -> `launch_swift_mtp.sh` (known-good) and
  `launch_swift_mtp_split.sh` -> `launch_swift_mtp_max_num_seq.sh`.
- Later the same day: both launchers were deleted and replaced by a single
  `launch.sh` (defaults `MAX_NUM_SEQS=4`, `GPU_UTIL=0.94`); the pinned base is
  reached with `IMAGE=<digest> MAX_NUM_SEQS=1 ./launch.sh`.
- Ports unified to **8080** in both launchers (run one at a time);
  `run_ab_matrix.sh` and `bench_mixed.py` defaults follow.
- `--max-num-batched-tokens` raised `8192` -> `16384` in both launchers,
  replacing the unaligned `16234`; re-validated (C1/C2/C2b) and measured
  (section 14.2).
- README serve-flag block refreshed (it was stale: 0.88 util, `--max-num-seqs
  64`, MTP 4) and the obsolete `FT=0` prefix removed.

### 15.6 Reproduce from scratch

```bash
# offline (no GPU): patch validation
mkdir -p _work && docker run --rm --entrypoint cat \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py > _work/_xpu_ops.py
python3 patches/patch_gdn_split.py --target _work/_xpu_ops.py   # 47080ce1… -> 1b69d2b9…

# image
docker build -f docker/Dockerfile.gdn-split -t vllm-xpu-gdn-split:0.1.12.3-p1 .

# one A/B cell (launch -> bench -> evidence -> teardown)
./run_ab_matrix.sh A2-split-seqs8 vllm-xpu-gdn-split:0.1.12.3-p1 8 s1,s2,s3 3
KEEP=1 ./run_ab_matrix.sh A2-split-seqs8 vllm-xpu-gdn-split:0.1.12.3-p1 8 s1,s2,s3 3  # leave running

# or interactively
./launch.sh
```

### 15.7 Current file state

| path | role |
|---|---|
| `patches/gdn_split_dispatch.upstream.md` | vendored upstream comment (provenance) |
| `patches/gdn_split_dispatch.diff` | upstream diff, parsed by the applier |
| `patches/patch_gdn_split.py` | hash-pinned, anchor-based, idempotent applier |
| `docker/Dockerfile.gdn-split` | derived image recipe (`vllm-xpu-gdn-split:0.1.12.3-p1`) |
| `.dockerignore` | keeps the 19 GB checkpoint out of build context |
| `launch.sh` | single launcher: patched image, `MAX_NUM_SEQS` (default 4), `GPU_UTIL` 0.94; pinned base via `IMAGE=<digest> MAX_NUM_SEQS=1` |
| `bench_mixed.py` | s1/s2/s3 concurrency harness + abort detection |
| `run_ab_matrix.sh` | launch -> bench -> capture -> teardown cell runner |
| `_work/`, `artifacts/` | scratch + evidence, gitignored |

Decision: chunk budget is `16384` for the target 128k-context workload; see 14.2 for the 8192 comparison.
