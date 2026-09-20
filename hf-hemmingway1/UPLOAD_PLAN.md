# Upload plan — publishing the Hemmingway-1 GPTQ INT4 quant to Hugging Face

Status: **UPLOADED — public** at
https://huggingface.co/bjonor/Hemmingway-1-GPTQ-Int4-sym-G128-MTP-BF16
(commit `449ada00524cf1de6aaba8fdc46adcde547b090d`, 2026-09-20, uploaded with
the `hfdeploy` write token). Payload staged in [`upload/`](upload/), weights
hardlinked from the quant output dir, gitignored.

Post-upload verification: all 18 files present; all 5 shard LFS SHA-256 match
`CHECKSUMS.sha256`; `tokenizer.json` LFS SHA-256 matches; `README.md`,
`config.json`, `quantize_config.json`, `CHECKSUMS.sha256` byte-identical to
local; repo public and ungated; card metadata parsed (`license: apache-2.0`,
`base_model: Altworld/Hemmingway-1`).

Verified before upload (2026-09-20): `verify_quant.py` PASS (400 qweight, 15
BF16 `mtp.*`, 0 visual, `lm_head` untouched); tiny text-model smoke PASS; vLLM
XPU serve + `test_serve.py` PASS **without MTP** (33.4 tok/s decode, seqs=1).
MTP speculative decoding aborts engine startup in the pinned runtime
(`profile_run` → `qwen3_5_mtp` dummy-run rotary seq_len mismatch with N=1 and
N=3) — disclosed in the card. `meta.offload_to_disk_path` sanitized to `null` in
both `quantize_config.json` and `config.json`.

Companion: [`MODEL_CARD.md`](MODEL_CARD.md) (becomes the HF repo `README.md`).

Repo under preparation:

```
hf-hemmingway1/
├── MODEL_CARD.md          # -> README.md on HF (frontmatter + body)
├── UPLOAD_PLAN.md         # this file
├── LICENSE                # Apache-2.0              (copied from base model)
├── NOTICE                 # uploader modification notice + attribution
├── .gitattributes         # LFS rules for the upload repo
└── upload/                # exact HF repo payload (weights + card + licence)
```

Source checkpoint: `/opt/models/Hemmingway-1` (`Altworld/Hemmingway-1`,
revision `4d711aac0f0043075ae334d2a3de3db3e10135c9`).
Quant output: `Hemmingway-1-GPTQ-Int4-sym-G128-MTP-BF16/`.

---

## 1. Pre-flight — must all pass before anything is uploaded

### 1.1 Licence compliance (blocking)

The artifact is a derivative of `Altworld/Hemmingway-1`, itself built on
`Qwen/Qwen3.8-27B`. **Both are plain Apache-2.0** — simpler than the Swift
quant (no custom licence, no revenue threshold). Obligations:

- [ ] `LICENSE` (Apache-2.0, copied verbatim from the base model) shipped.
- [ ] `NOTICE` shipped: states the modification (quantization) and retains
      attribution to Altworld and Alibaba Cloud. The base model ships **no**
      NOTICE file, so this one is ours; it must not imply endorsement.
- [ ] Modified files carry a prominent change notice — the card's "License and
      attribution" section plus `NOTICE` cover this (§4(b)).
- [ ] HF metadata `license: apache-2.0` (no `other`/custom licence here).
- [ ] Trademark care: "Hemmingway", "Altworld" and "Qwen" used only for
      attribution; no endorsement claimed anywhere in the card or repo name.
- [ ] Re-check the base model's licence and gating at upload time (can change).

Decision to make: **public ungated** (matches Apache-2.0 and the base repo).

### 1.2 Artifact completeness (blocking)

The source is text-only and uses a fast tokenizer; it ships **no**
`vocab.json`, `merges.txt`, `preprocessor_config.json` or
`video_preprocessor_config.json` — do not copy those. Verify instead:

- [ ] `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
      `config.json`, `generation_config.json` present in the upload dir.
- [ ] `quantize_config.json` present and contract-complete (`quant_log.csv`
      recommended, keep and mention in the card).
- [ ] `config.json` carries the `quantization_config` block and
      `architectures: ["Qwen3_5ForCausalLM"]` (not the VLM class).
- [ ] `model.safetensors.index.json` exists; the 15 `mtp.*` keys are present
      and map to unquantized BF16 tensors (the source's separate
      `model-mtp.safetensors` shard is merged into the output — confirm after
      the run).
- [ ] Sanitise `quantize_config.json` if `meta.offload_to_disk_path` contains a
      local temp path (harmless, but disclose or strip).
- [ ] `tokenizer_config.json` `"is_local": true` from the local export —
      harmless, optional cleanup.
- [ ] `README.md` (this card), `LICENSE`, `NOTICE`, `.gitattributes`,
      `CHECKSUMS.sha256` copied into the payload.

### 1.3 Quality gates (blocking)

```bash
cd "$(git rev-parse --show-toplevel)"    # repo root
OUT=Hemmingway-1-GPTQ-Int4-sym-G128-MTP-BF16 EXPECT_VISUAL=0 \
  .venv/bin/python verify_quant.py       # expect: VERIFY PASS
DEVICE=cpu .venv/bin/python smoke_test.py   # Swift/VLM path — see note below
./launch.sh && .venv/bin/python test_serve.py   # optional: serve the local dir
```

- [ ] `VERIFY PASS` on the exact directory that will be uploaded: expect
      `mtp tensors preserved: 15`, `visual tensors: 0`,
      `quantized modules: 400`, `lm_head quantized: False`.
- [ ] Text-model smoke test: the tiny text-model flow already passed on XPU
      during the run preflight (13 qweight modules for 2 layers, MTP copied
      verbatim). `smoke_test.py` currently builds a VLM and does not cover the
      text path — either parameterize it or re-run the tiny text check.
- [ ] SHA-256 manifest generated in the payload:

```bash
cd Hemmingway-1-GPTQ-Int4-sym-G128-MTP-BF16
sha256sum *.safetensors model.safetensors.index.json > CHECKSUMS.sha256
sha256sum -c CHECKSUMS.sha256
```

**Optional but recommended — a real quality check.** The card currently says no
standard benchmarks were run. Cheapest defensible options: (a) perplexity on a
held-out slice (wikitext-103 or an UltraChat held-out split), BF16 vs this INT4
checkpoint; (b) a small GPQA/AIME subset. Even one number turns "unverified"
into "measured, small-sample".

### 1.4 Naming / metadata decisions

- [ ] HF namespace and repo name. Suggestion:
      `<hf-user>/Hemmingway-1-GPTQ-Int4-sym-G128-MTP-BF16` (descriptive,
      mirrors the Swift quant naming; no trademark in the name).
- [ ] `pipeline_tag: text-generation`; `library_name: transformers`;
      `base_model: Altworld/Hemmingway-1` + `base_model_relation: quantized`.
- [ ] Revision naming: tag the first upload `v1.0-gptq-int4-g128`; any re-quant
      (different calibration, method or exclusion set) gets its own revision and
      a changelog entry in the card.

---

## 2. Disclosure checklist — what must appear publicly

**Method and provenance**
- [ ] GPTQ INT4 W4A16, g128, symmetric, `desc_act=false`; quantizer version;
      `lm_head` excluded.
- [ ] Calibration data (`HuggingFaceH4/ultrachat_200k` `train_sft[:256]`,
      2048-token cap) and the source/calibration revisions.
- [ ] Preserved modules: 15 `mtp.*` BF16 tensors; the model is text-only (no
      vision tower).
- [ ] Hardware and wall time (Arc Pro B70, actual time) → energy estimate.
- [ ] Contract relationship to `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
      (same contract, different base fine-tune).

**Known issues / operational caveats**
- [ ] MTP draft requires an unquantized build (env gate/patches for vLLM XPU).
- [ ] Gated-DeltaNet mixed spec-decode + prefill limitation inherited from the
      recipe (`--max-num-seqs 1` or kernels ≥ 0.1.14.1 + vLLM PR, or the Python
      split-dispatch backport) — noted as **not re-validated on this artifact**.
- [ ] fp8 KV cache is a serving choice, not part of the checkpoint.
- [ ] No serving benchmark numbers for this artifact.
- [ ] Long context validated only to whatever is actually served (card says not
      validated; update if a serve test is run).

**Limitations and honesty**
- [ ] No standard quality benchmarks run on this artifact (unless 1.3 done).
- [ ] Quantization is lossy; per-module error not published.
- [ ] No safety alignment/red-teaming by the uploader; base-model risks apply.
- [ ] Intended use / out-of-scope statements (including the base model's
      medical/legal/financial warning).
- [ ] Unaffiliated/unofficial statement (no Altworld or Alibaba endorsement).

**Licence chain**
- [ ] Apache-2.0 `LICENSE`, our `NOTICE` with the modification notice, and the
      attribution lines in the card.

---

## 3. Upload procedure

```bash
# 0. stage the exact HF repo payload (gitignored)
mkdir -p hf-hemmingway1/upload && cd hf-hemmingway1/upload
#    weights: hardlink (same filesystem) or copy
cp -l ../../Hemmingway-1-GPTQ-Int4-sym-G128-MTP-BF16/* .
cp ../MODEL_CARD.md README.md
cp ../LICENSE ../NOTICE ../.gitattributes .
#    (re-copy the non-weight files from the source if the quant output is
#     missing any tokenizer/chat-template file)
(cd ../../Hemmingway-1-GPTQ-Int4-sym-G128-MTP-BF16 && \
  sha256sum *.safetensors model.safetensors.index.json > ../CHECKSUMS.sha256)
cp ../CHECKSUMS.sha256 .

# 1. create the repo (public, Apache-2.0)
hf auth whoami
hf repo create <hf-user>/Hemmingway-1-GPTQ-Int4-sym-G128-MTP-BF16 \
  --repo-type model --public

# 2. upload: large, resumable
hf upload-large-folder <hf-user>/Hemmingway-1-GPTQ-Int4-sym-G128-MTP-BF16 . \
  --repo-type model
```

`.gitattributes` (LFS): `*.safetensors`, and the standard binary list already
copied from the repo root `.gitattributes`.

Rollback / maintenance: `hf repo delete` or set private; local weights plus
`CHECKSUMS.sha256` always remain, so a takedown is never data-loss.

---

## 4. Post-upload verification (do not skip)

- [ ] `hf download <repo> --include 'model-*' 'model.safetensors.index.json'`
      and compare `sha256sum` with `CHECKSUMS.sha256`.
- [ ] Serve **from the Hub id** (not the local dir) and re-run the endpoint test
      if serving was validated locally first:
      `vllm serve <hf-user>/<repo> --quantization gptq --dtype float16
      --max-model-len 131072 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
- [ ] Card renders: frontmatter parsed (Apache-2.0, base model linkage
      visible), tables intact.
- [ ] `LICENSE`, `NOTICE`, `CHECKSUMS.sha256`, `quantize_config.json`,
      `quant_log.csv` present in the file list.
- [ ] Sanity check that no local absolute paths leaked into text files
      (`grep -r "$HOME\|/opt/models\|/tmp/" .` on the payload).

---

## 5. Risks

| risk | mitigation |
|---|---|
| Attribution missed (base ships no NOTICE) | our own `NOTICE` with modification + attribution; card links both base models |
| Users assume benchmark accuracy or speed | explicit "not run / not measured on this artifact" sections |
| Users miss the GDN mixed-batch kernel limitation | prominent "How to use" note + link to the recipe repo; marked not re-validated here |
| MTP weights accidentally quantized | `verify_quant.py` fails loudly if any `mtp.*` carries `.qweight`/`.qzeros`/`.scales` |
| Long-term maintenance load (serving issues) | card routes issues to the GitHub repo, not the model page |

---

## 6. Open decisions (need a yes/no before upload)

1. Run a small quality evaluation first (perplexity and/or GPQA/AIME subset), or
   upload with the "no benchmarks" disclosure as-is?
2. Keep `quant_log.csv` in the public repo? (recommended: yes, provenance)
3. Sanitise `meta.offload_to_disk_path` in `quantize_config.json` or leave it?
4. HF namespace/repo name confirmation.
5. Validate serving (launch.sh + test_serve.py + bench) before upload, or upload
   first and benchmark after?
