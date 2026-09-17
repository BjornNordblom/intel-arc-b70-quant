# Upload plan — publishing the GPTQ INT4 quant to Hugging Face

Status: **PREP — not uploaded**. Companion: [`MODEL_CARD.md`](MODEL_CARD.md)
(becomes the HF repo `README.md`).

Repo under preparation:

```
hf/
├── MODEL_CARD.md          # -> README.md on HF (frontmatter + body)
├── UPLOAD_PLAN.md         # this file
├── LICENSE                # Swift Open License v1.0   (copied from base model)
├── LICENSE-APACHE-2.0     # Qwen3.8-27B licence       (copied from base model)
├── NOTICE                 # UkisAI change notice      (copied from base model)
└── .gitattributes         # LFS rules for the upload repo
```

---

## 1. Pre-flight — must all pass before anything is uploaded

### 1.1 Licence compliance (blocking)

The artifact is a derivative of `ukisai/Swift-Qwen3.8-27b` (Swift Open License
v1.0, an Apache-2.0 derivative) and, through it, of `Qwen/Qwen3.8-27B`
(Apache-2.0). Redistribution is permitted; the following are **required**:

- [ ] `LICENSE` (Swift Open License v1.0) shipped in the uploaded repo — Swift Open License §4(a).
- [ ] `LICENSE-APACHE-2.0` shipped as well — §4(e) for the Base Model portion.
- [ ] `NOTICE` reproduced — §4(d).
- [ ] The protected notice in the model card (verbatim, already in
      `MODEL_CARD.md` → "License and attribution").
- [ ] Our own modification notice included (addendum; must not read as modifying
      the licence).
- [ ] Commercial-use threshold (US$1M gross annual revenue incl. affiliates)
      disclosed, since §5 continues to apply to derivatives that contain the
      Swift Contribution.
- [ ] Trademark care: "UkisAI" used only for attribution, never in the repo name
      or in a way implying endorsement.
- [ ] No relicensing: repo licence metadata must be `other` /
      `swift-open-license-1.0` pointing at the base LICENSE. Do **not** tag it
      `apache-2.0`.
- [ ] Re-check the base model's current licence text and gating at upload time
      (they can change; the HF API currently reports the repo as not gated, while
      its README frontmatter says `gated: true`).

Decision to make: **public ungated** (recommended, matches the licence) vs gated.
Gating adds friction without changing the licence obligations.

### 1.2 Artifact completeness (blocking)

`gptqmodel` wrote a minimal file set; the upload must restore the non-weight
files from the source checkpoint so the repo is usable by `transformers`
(slow-tokenizer assets) and for multimodal tooling:

```bash
SRC=${SRC_MODEL:-/path/to/Swift-Qwen3.8-27b}   # source BF16 checkpoint
D=Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
for f in vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json; do
  cp "$SRC/$f" "$D/$f"
done
```

- [ ] `vocab.json`, `merges.txt`, `preprocessor_config.json`,
      `video_preprocessor_config.json` added (all unmodified Apache-2.0 files).
- [ ] Decided whether to keep `quant_log.csv` (useful provenance; contains only
      per-module timings) — recommended: keep and mention in the card.
- [ ] `quantize_config.json` reviewed for local paths
      (`meta.offload_to_disk_path=/tmp/gptqmodel_…`); harmless, but disclose or
      sanitise.
- [ ] `tokenizer_config.json` still contains `"is_local": true` from the local
      export; harmless, optional cleanup.

### 1.3 Quality gates (blocking)

```bash
cd "$(git rev-parse --show-toplevel)"    # repo root
.venv/bin/python verify_quant.py                 # expect: VERIFY PASS
DEVICE=cpu .venv/bin/python smoke_test.py        # expect: SMOKE TEST PASS
# serve the local dir and run the endpoint tests
./launch.sh && .venv/bin/python test_serve.py
```

- [ ] `VERIFY PASS` on the exact directory that will be uploaded (re-run after
      copying the extra files).
- [ ] Local serve + `test_serve.py` pass.
- [ ] SHA-256 manifest generated and stored in the repo being uploaded:

```bash
cd Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
sha256sum *.safetensors model.safetensors.index.json > CHECKSUMS.sha256
sha256sum -c CHECKSUMS.sha256
```

**Optional but recommended — run a real quality check first.** Right now the
card says "no standard benchmarks were run on this artifact". Cheapest defensible
options: (a) perplexity on a held-out slice (e.g. wikitext-103 or a
UltraChat held-out split) for BF16 vs this INT4 checkpoint; (b) a small
GPQA/AIME/IFBench subset with the same prompts as the upstream INT4 table. Even
one number turns "unverified" into "measured, small-sample". If skipped, the
disclosure stays as written.

### 1.4 Naming / metadata decisions

- [ ] HF namespace and repo name. Suggestion:
      `<hf-user>/Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (mirrors the
      source naming, no trademark use).
- [ ] `pipeline_tag`: suggested `image-text-to-text` (vision tensors are
      preserved and the source is multimodal) **with** the explicit
      "vision not validated" note; alternative is `text-generation` to understate.
- [ ] `base_model: ukisai/Swift-Qwen3.8-27b` + `base_model_relation: quantized`
      (already in the card frontmatter).
- [ ] Revision naming: tag the first upload `v1.0-gptq-int4-g128`; any re-quant
      (different calibration, method, exclusion set) gets its own revision and a
      changelog entry in the card.

---

## 2. Disclosure checklist — what must appear publicly

**Method and provenance**
- [ ] GPTQ INT4 W4A16, g128, symmetric, `desc_act=false`; quantizer version;
      `lm_head` excluded.
- [ ] Calibration data (`HuggingFaceH4/ultrachat_200k` `train_sft[:256]`, 2048-token
      cap) — so users can judge domain suitability.
- [ ] Preserved modules: 15 `mtp.*` + 333 `model.visual.*` tensors.
- [ ] Hardware and wall time (Arc Pro B70, 2.40 h) → also the energy estimate.
- [ ] Contract identical to the community reference artifact except a quant-time
      flag, and the clarification that the reference quantizes the *base* model
      while this one quantizes the *Swift* fine-tune.

**Known issues / operational caveats**
- [ ] MTP draft requires an unquantized build (env gate/patches for vLLM XPU).
- [ ] `vllm-xpu-kernels < 0.1.14.1`: mixed spec-decode + prefill batches abort the
      engine → use `--max-num-seqs 1`, kernels ≥ 0.1.14.1, or the Python
      split-dispatch backport (link the recipe repo).
- [ ] fp8 KV cache is a serving choice, not part of the checkpoint.
- [ ] Long-context validated to 131,072 (native 262,144).
- [ ] Vision supported structurally, not validated.

**Limitations and honesty**
- [ ] No standard quality benchmarks run on this artifact (unless 1.3 is done).
- [ ] Quantization is lossy; per-module error not published.
- [ ] No safety alignment/red-teaming by the uploader; base-model risks apply.
- [ ] Intended use / out-of-scope statements.
- [ ] Unaffiliated/unofficial statement (no UkisAI or Alibaba endorsement).

**Licence chain**
- [ ] The verbatim redistributor notice, the two licences + NOTICE in the repo,
      the US$1M commercial threshold, and the "no trademark rights" note.
- [ ] Our own modification addendum.

---

## 3. Upload procedure

```bash
# 0. staging dir (gitignored) — exact content of the HF repo
mkdir -p hf/upload && cd hf/upload        # run from the repo root
cp ../../Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16/* .
cp ../LICENSE ../LICENSE-APACHE-2.0 ../NOTICE ../MODEL_CARD.md .
mv MODEL_CARD.md README.md
cp ../.gitattributes .

# 1. create the repo (public, licence "other")
hf auth whoami
hf repo create <hf-user>/Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --repo-type model --public

# 2. upload: large, resumable (19 GB, 5 shards)
hf upload-large-folder <hf-user>/Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 . \
  --repo-type model
#    (small files first is fine too: hf upload <repo> . --repo-type model)

# 3. tag the revision
hf download <hf-user>/<repo> --repo-type model --revision main --quiet
```

`.gitattributes` (LFS): `*.safetensors filter=lfs diff=lfs merge=lfs -text`,
`*.mp4`, `*.png` if any media are ever added.

Rollback / maintenance: `hf repo delete` or set private; a local copy plus the
`CHECKSUMS.sha256` manifest always remains, so a takedown is never data-loss.

---

## 4. Post-upload verification (do not skip)

- [ ] `hf download <repo> --include 'model-00001*' 'model.safetensors.index.json'`
      and compare `sha256sum` with `CHECKSUMS.sha256`.
- [ ] Serve **from the Hub id** (not the local dir) and run the endpoint test:

```bash
vllm serve <hf-user>/<repo> --quantization gptq --dtype float16 \
  --served-model-name swift38 --max-model-len 131072 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --language-model-only
.venv/bin/python test_serve.py
.venv/bin/python bench_mtp4.py        # compare with the 58.8 tok/s / 3.38 local figure
```

- [ ] Card renders: frontmatter parsed (licence shows `other` +
      `swift-open-license-1.0`, base model linkage visible), tables intact.
- [ ] `LICENSE`, `LICENSE-APACHE-2.0`, `NOTICE`, `CHECKSUMS.sha256`,
      `quantize_config.json`, `quant_log.csv` present in the file list.
- [ ] Sanity check the model page tags and that no local absolute paths leaked
      into text files (e.g. `grep -r "$HOME" .`).

---

## 5. Risks

| risk | mitigation |
|---|---|
| Licence mis-tagging (`apache-2.0`) would be wrong | metadata is `other` + `swift-open-license-1.0`; licence files shipped |
| Base licence/gating changes after upload | re-check at upload time; card links the licence instead of copying terms into claims |
| Users miss the mixed-batch kernel limitation and hit engine aborts | prominent "How to use" note + link to the recipe repo |
| Users assume benchmark accuracy | explicit "no benchmarks run on this artifact" section |
| Trademark/repo-name complaint | name is descriptive only; no UkisAI mark in the repo name |
| Long-term maintenance load (issues about serving patches) | card routes issues to the GitHub repo, not the model page |

## 6. Open decisions (need a yes/no before upload)

1. Run a small quality evaluation first (perplexity and/or GPQA/AIME subset), or
   upload with the "no benchmarks" disclosure as-is?
2. `pipeline_tag`: `image-text-to-text` (accurate structurally) or
   `text-generation` (understates, but matches what was validated)?
3. Keep `quant_log.csv` in the public repo?
4. Sanitise `meta.offload_to_disk_path` in `quantize_config.json` or leave it?
5. HF namespace/repo name confirmation, and whether to mention the reference
   artifact `SergiioB/…` in the card (recommended: yes, as comparison context).
