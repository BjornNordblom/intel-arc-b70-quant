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

Requires a vLLM XPU build with Gated-DeltaNet support. Verified configuration
on an Intel Arc Pro B70 (`vllm-xpu-gdn-split:0.1.12.3-p1`,
`vllm 0.27.2rc1.dev77+gac7509e2b.xpu`):

```bash
vllm serve <this-repo> \
  --quantization gptq --dtype float16 --max-model-len 131072 \
  --gpu-memory-utilization 0.94 --kv-cache-dtype fp8 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Ready in ~100 s; ~32 tok/s post-first-token decode on a short prompt, with
clean answers and valid tool calls. MTP speculative decoding can be added on
the newer nightly build (verified, see below).

#### Always pass the two reasoning flags

The fine-tune defaults to **thinking mode** (`reasoning_effort=xhigh`) whenever
`enable_thinking` is not set: its chat template injects a reasoning system
message and opens a think block on every request. On a server without a
reasoning parser the whole planning monologue is returned inside
`message.content`; clients store and replay that text, the model then reasons
about its own previous reasoning, and the output looks like the model is
**repeating itself and never answering**. Measured on this artifact:

| request | without the flags | with the flags |
|---|---|---|
| `Reply with exactly: hello` | planning monologue, no answer | `hello`, 2 tokens, `stop` |
| short writing request | hundreds of planning tokens, often hits `max_tokens` | clean answer, ~50 tokens |

- `--default-chat-template-kwargs '{"enable_thinking": false}'` answers directly.
- `--reasoning-parser qwen3` keeps reasoning out of `content` (separate
  `reasoning` / `reasoning_content` field) for requests that opt back in with
  `"chat_template_kwargs": {"enable_thinking": true}`.
- Sampling: the checkpoint's `generation_config.json` sets
  `temperature 1.0 / top_k 20 / top_p 0.95`. Around `temperature 0.7` the
  monologue is much less likely to degrade; the shipped 1.0 is where apparent
  repetition is most visible.

This is **not** a quantization artifact: the unquantized
`Altworld/Hemmingway-1` weights produce the same planning monologue under the
same template and defaults (measured on CPU with `transformers`). The fine-tune
appears to be trained for an internal harness system prompt — the model refers
to a "system-reminder" with nine planning steps that appears in neither the
chat template nor the config.

#### MTP speculative decoding

The 15 `mtp.*` draft tensors are present and unquantized (BF16). On the pinned
runtime above MTP did **not** start: with
`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`
(tried N=1 and N=3) the engine aborts during `profile_run` in the draft
`dummy_run` with `RuntimeError: query, key and positions must have the same
batch_size and seq_len` (traceback `qwen3_5_mtp.py` →
`qwen3_next.py::_project_qkv_gate` → `rotary_embedding`). The abort happens in
shape-check code before any weights are used, and it is specific to this
checkpoint's **text-only** config layout (`Qwen3_5ForCausalLM` /
`qwen3_5_text`; the community reference checkpoint ships as a multimodal
`Qwen3_5ForConditionalGeneration` config).

Upstream changed exactly that code path afterwards: vLLM `0.29.1` added XPU +
MRoPE support to the fused QK-norm/RoPE step. **Verified on the unpatched
nightly image** (`vllm/vllm-openai-xpu:nightly` from 2026-09-20,
`0.29.1rc1.dev422`): the engine starts with

```bash
vllm serve <this-repo> ... \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Measured on this artifact (Arc Pro B70, `--max-num-seqs 1`): ready in 131 s,
mean acceptance length **2.38**, per-position draft acceptance
**0.71 / 0.43 / 0.24** (average 46%), and **45.3 tok/s** on a 200-token writing
request at `temperature 0.7` (the pinned image without MTP was ~32 tok/s, but
that is a different vLLM build — indicative, not a controlled A/B). The nightly
image does not carry the Gated-DeltaNet split-dispatch backport from the pinned
image, so keep `--max-num-seqs 1` there.

Other notes:

- `--kv-cache-dtype fp8` is a serving choice, not part of the checkpoint.
- The Gated-DeltaNet mixed-batch limitation documented for the sibling Swift
  quant (`vllm-xpu-kernels < 0.1.14.1`: spec-decode + prefill in one batch
  aborts the engine) is not reachable here while speculative decoding is off.
- `--enable-auto-tool-choice --tool-call-parser qwen3_xml` is optional; the
  chat template is the Qwen3-style XML tool format.

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
| vLLM XPU serve + endpoint/streaming test (no MTP, `--max-num-seqs 1`) | **PASS** — ready in ~100 s, ~32 tok/s post-first-token decode |
| Reasoning-mode fix (`--reasoning-parser qwen3` + `enable_thinking=false`) | **PASS** — exact short answers, valid tool calls; without the flags the planning monologue is returned in `content` (see "How to use") |
| MTP speculative decoding | **PASS** on the unpatched nightly `0.29.1rc1.dev422` — starts in 131 s, mean acceptance length 2.38, avg draft acceptance 46%, 45.3 tok/s on a 200-token writing request; **FAILED to start** on the pinned `0.27.2rc1.dev77` runtime (see "How to use") |
| Standard quality benchmarks (MMLU, GPQA, AIME, IFBench, perplexity) | **Not run** on this artifact |
| Concurrency / sustained load | **Not measured** |

The base model's own published evaluations are indicative but were not measured
on this quantization. Treat accuracy and speed figures as unverified for this
checkpoint.

## Intended use and out-of-scope

- Intended: local inference and evaluation of the Hemmingway-1 fine-tune on
  Intel XPU / low-VRAM setups; MTP speculative decoding pending runtime
  verification (see "How to use").
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
- Serving was verified without MTP on the pinned image and with MTP on the
  nightly image; concurrency and long-context behaviour were not measured.
- The fine-tune enables thinking mode by default and expects an internal harness
  system prompt that is not shipped with the weights. Without the serving flags
  documented above, plain chat requests produce long planning monologues, and
  clients that display the reasoning stream as normal assistant text will make
  it look like the model is stuck repeating itself. This is inherited from the
  base fine-tune (reproduced on unquantized weights), not caused by the
  quantization. Even with thinking disabled, about 1 in 5 sampled replies
  (`temperature 0.7`, landlord-prompt test) still opened with planning-style
  prose instead of the answer — retry, lower the temperature, or give the model
  a short system instruction.
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
