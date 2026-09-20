---
license: apache-2.0
base_model: Altworld/Hemmingway-1
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
language:
- en
datasets:
- HuggingFaceH4/ultrachat_200k
tags:
- gptq
- int4
- w4a16
- 4-bit
- quantized
- vllm
- intel-xpu
- arc-pro-b70
- mtp
- speculative-decoding
- gated-deltanet
- qwen3_5
- text-generation
- creative-writing
---

# Hemmingway-1 — GPTQ INT4 (W4A16, group-128, symmetric) + BF16 MTP head

Unofficial 4-bit GPTQ quantization of
[`Altworld/Hemmingway-1`](https://huggingface.co/Altworld/Hemmingway-1) for
low-VRAM inference, with the MTP (multi-token-prediction) draft head
deliberately **kept unquantized**. Produced with `gptqmodel 7.3.2` on an Intel
Arc Pro B70 (Xeon/e-core host, 30.3 GiB VRAM).

This model is **not affiliated with, endorsed by, or supported by Altworld,
hemmingway.io, or Alibaba Cloud**. It is a community quantization; the Apache-2.0
terms of the base models below continue to apply (see
[License and attribution](#license-and-attribution)).

## Model details

| | |
|---|---|
| Base (fine-tune) | `Altworld/Hemmingway-1` (Apache-2.0) |
| Base (original) | `Qwen/Qwen3.8-27B` (Apache-2.0), Copyright 2026 Alibaba Cloud |
| Architecture | `Qwen3_5ForCausalLM` (text-only), 64 layers (hybrid Gated-DeltaNet linear attention + full attention every 4th layer), hidden 5120, 24 Q / 4 KV heads, head_dim 256, 248,320 vocab, ~27B params |
| Quantization | GPTQ, 4-bit weights, 16-bit activations (W4A16), `group_size=128`, `sym=true`, `desc_act=false`, `lm_head` not quantized |
| Preserved unquantized | 15 `mtp.*` tensors (BF16 draft head; shipped by the source as a separate `model-mtp.safetensors` shard), plus embeddings, norms and Gated-DeltaNet parameters |
| Quantized modules | 400 (all quantizable linear projections of the 64 layers) |
| Checkpoint size | 18.6 GB (17.4 GiB), 5 safetensors shards (2,066 tensors in the index) |
| Native context | 262,144 tokens |
| Quantizer | `gptqmodel 7.3.2` (torch 2.9.1+xpu) |
| Source revision | `4d711aac0f0043075ae334d2a3de3db3e10135c9` |
| Calibration | `HuggingFaceH4/ultrachat_200k` `train_sft[:256]`, rendered with the model chat template, truncated to 2048 tokens |
| Calibration revision | `8049631c405ae6576f93f445c6b8166f76f5505a` |
| Code | quantization, verification and Intel-XPU serving recipe: [BjornNordblom/intel-arc-b70-quant](https://github.com/BjornNordblom/intel-arc-b70-quant); serving patches from [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) |

The `quantize_config.json` follows the same contract as the community reference
artifact
[`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
(W4A16, g128, sym, `desc_act=false`, MTP excluded), applied here to the
**Hemmingway-1** fine-tune instead of the base Qwen3.8-27B.

## What was changed vs the source checkpoint

- All quantizable linear projections of the language model → GPTQ INT4
  (W4A16, g128, sym, `desc_act=false`).
- `mtp.*` draft tensors (15) kept in their original BF16 dtype; the `dynamic`
  exclusion `{"-:.*mtp.*": {}}` in `quantize_config.json` records this. The
  source's separate `model-mtp.safetensors` shard is merged into the output
  checkpoint (`model-00005-of-00005.safetensors`; confirmed via
  `model.safetensors.index.json`).
- Added: `quantize_config.json`, `quant_log.csv` (per-module quantization
  timings), `README.md`, `LICENSE`, `NOTICE`, `.gitattributes`,
  `CHECKSUMS.sha256`.
- Unchanged content: `config.json` (plus a `quantization_config` block),
  `generation_config.json`, `chat_template.jinja`, `tokenizer.json`,
  `tokenizer_config.json`.

## Quantization details

| | |
|---|---|
| Method | GPTQ (gptqmodel 7.3.2), true-sequential, static groups off |
| Bits / group | 4-bit / 128, symmetric, `desc_act=false` |
| Damping | `damp_percent=0.05`, `damp_auto_increment=0.01`, Hessian staging FP32 |
| Fallback | RTN for modules exceeding a 0.5% error threshold |
| Calibration | 256 UltraChat-SFT samples, ≤2048 tokens each |
| Pack format | `int32`, `checkpoint_format=gptq`, `lm_head=false` |
| Wall time | 2.39 h on one Arc Pro B70 (host i9-13900K, 64 GB RAM) |
| Verification | contract check (`verify_quant.py`) PASS; vLLM XPU serve + endpoint/streaming test PASS without MTP; MTP draft fails to start in the tested runtime (see below) |

## How to use

### vLLM (Intel XPU)

The serving recipe is the same family as the Swift quant in the code repo.
Requires a vLLM XPU build with Gated-DeltaNet support (tested there with
`vllm/vllm-openai-xpu` @ `vllm 0.27.2rc1.dev77+gac7509e2b.xpu`,
`vllm-xpu-kernels 0.1.12.3`).

```bash
vllm serve <this-repo> \
  --quantization gptq --dtype float16 --max-model-len 131072 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
  --max-num-seqs 1 --max-num-batched-tokens 16384 --enable-prefix-caching
```

This is the configuration that was verified on this artifact (Intel Arc Pro
B70, `vllm-xpu-gdn-split:0.1.12.3-p1`, ready in 96 s; endpoint + streaming test
PASS, 33.4 tok/s post-first-token decode on a short prompt).

Notes for this model on XPU:

- **MTP speculative decoding did not start on this artifact in the tested
  runtime.** With `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`
  (tried N=1 and N=3) the engine aborts during `profile_run` in the MTP draft
  dummy run: `RuntimeError: query, key and positions must have the same
  batch_size and seq_len` (traceback through `qwen3_5_mtp.py` →
  `qwen3_next.py::_project_qkv_gate` → `rotary_embedding`). This is a runtime
  limitation, not a checkpoint defect: the 15 `mtp.*` tensors are present and
  unquantized (`verify_quant.py` confirms). Disable speculative decoding until
  the runtime path is fixed, or test a newer vLLM XPU build.
- Because of the above, the MTP BF16-draft runtime gate used for the Swift quant
  in the recipe repo (`B70_MTP_BF16_DRAFT=1`) was not required for this
  verification run.
- The Gated-DeltaNet mixed-batch limitation documented for the Swift quant
  (`vllm-xpu-kernels < 0.1.14.1`: spec-decode + prefill in one batch aborts the
  engine) is not reachable here while spec decoding is off.
- `--kv-cache-dtype fp8` is a serving choice, not part of the checkpoint.

### Transformers

GPTQ checkpoints need a GPTQ-capable loader (`gptqmodel` or `auto-gptq`) and a
`transformers` version that knows `qwen3_5_text`. The `mtp.*` tensors are not
used by `transformers` inference and can be ignored.

## Evaluation

What was actually measured on this artifact:

| Check | Result |
|---|---|
| Quantization contract (`verify_quant.py`) | **PASS** — bits 4, group 128, sym, `desc_act=false`, 15 MTP tensors preserved (none quantized), 400 modules quantized, `lm_head` untouched, no unindexed shards |
| Tiny-model smoke test (2-layer text `Qwen3_5ForCausalLM`, XPU) | PASS — MTP tensors copied verbatim, none quantized |
| vLLM XPU serve + endpoint/streaming test (no MTP, `--max-num-seqs 1`) | **PASS** — ready in 96 s, 33.4 tok/s post-first-token decode, TTFT 0.16 s (short prompt) |
| MTP speculative decoding | **FAILED to start** in the tested runtime (see "How to use"); checkpoint MTP tensors are intact |
| Standard quality benchmarks (MMLU, GPQA, AIME, IFBench, perplexity) | **Not run** on this artifact |
| Concurrency / sustained load | **Not measured** |

The base model's own published evaluations are indicative but were not measured
on this quantization. Treat accuracy and speed figures as unverified for this
checkpoint.

## Intended use and out-of-scope

- Intended: local inference and evaluation of the Hemmingway-1 fine-tune on
  Intel XPU / low-VRAM setups, with optional MTP speculative decoding.
- Out of scope: any safety-critical, medical, legal, or production decision
  making; anything requiring verified accuracy on this specific checkpoint.
  The base model authors' own warning applies: it can be wrong and still sound
  certain, so do not use it to decide anything medical, legal or financial.
- No safety alignment or red-teaming was performed by the uploader. Behavioural
  risks of the base model (bias, hallucinations, prompt-injection
  susceptibility) also apply here and may be amplified by quantization.

## Limitations

- Quantization is lossy; per-module error was not published beyond the RTN
  fallback threshold report.
- Serving was verified without MTP only; the MTP draft aborts engine startup
  in the tested XPU runtime (see above).
- Long-context behaviour was not validated on this artifact (native 262,144).
- Exporting to other formats (GGUF/AWQ) from this checkpoint is untested.

## Environmental impact

One quantization run: **2.39 h** on a single Arc Pro B70 (230 W board power
cap) — roughly **0.55 kWh** board energy, plus host overhead (i9-13900K, 64 GB
RAM). Serving costs depend on runtime and context length.

## License and attribution

The base fine-tune [`Altworld/Hemmingway-1`](https://huggingface.co/Altworld/Hemmingway-1)
and the original [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
are both licensed under the **Apache License 2.0**. This quantization is
distributed under the same license; see [`LICENSE`](LICENSE).

Changes made by the uploader (also in [`NOTICE`](NOTICE)):

> GPTQ INT4 W4A16 quantization of Hemmingway-1 by Bjorn Nordblom, 2026:
> language-model linear projections quantized with gptqmodel 7.3.2 on an Intel
> Arc Pro B70; the MTP draft head preserved in BF16. No changes to the model
> architecture or the licence terms.

Copyright in the base models remains with their respective authors (Altworld;
Alibaba Cloud for Qwen3.8-27B). Names are used for attribution only; no
trademark rights are granted and no endorsement is implied.

## Citation

```bibtex
@misc{hemmingway-1-gptq-int4,
  title        = {Hemmingway-1 GPTQ INT4 (W4A16, group-128, symmetric) with preserved BF16 MTP head},
  author       = {Nordblom, Bjorn},
  year         = {2026},
  howpublished = {Hugging Face},
  note         = {Quantization of Altworld/Hemmingway-1},
}

@misc{hemmingway-1,
  title  = {Hemmingway-1},
  author = {Altworld},
  year   = {2026},
  url    = {https://huggingface.co/Altworld/Hemmingway-1}
}

@misc{qwen3.8-27b,
  title  = {Qwen3.8-27B},
  author = {Alibaba Cloud},
  year   = {2026},
  url    = {https://huggingface.co/Qwen/Qwen3.8-27B}
}
```

## Acknowledgements

- Altworld for the Hemmingway-1 fine-tune.
- Alibaba Cloud / Qwen for Qwen3.8-27B (Apache-2.0).
- The `gptqmodel` project for the quantizer.
- SergiioB's [intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
  for the Intel Arc Pro B70 XPU serving patches used by the recipe repo (MIT,
  Copyright (c) 2026 SergiioB).

## Model card contact

Issues and questions: https://github.com/BjornNordblom/intel-arc-b70-quant/issues
