---
license: other
license_name: swift-open-license-1.0
license_link: https://huggingface.co/ukisai/Swift-Qwen3.8-27b/blob/main/LICENSE
base_model: ukisai/Swift-Qwen3.8-27b
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
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
- tool-calling
---

# Swift-Qwen3.8-27B — GPTQ INT4 (W4A16, group-128, symmetric) + BF16 MTP head

Unofficial 4-bit GPTQ quantization of
[`ukisai/Swift-Qwen3.8-27b`](https://huggingface.co/ukisai/Swift-Qwen3.8-27b)
for low-VRAM inference, with the MTP (multi-token-prediction) draft head and the
vision tower deliberately **kept unquantized**. Produced with `gptqmodel 7.3.2`
on an Intel Arc Pro B70 (Xeon/e-core host, 30.3 GiB VRAM), ~2.4 h wall time.

This model is **not affiliated with, endorsed by, or supported by UkisAI or
Alibaba Cloud**. It is a community quantization; the licence terms of the base
models below continue to apply (see [License](#license-and-attribution)).

## Model details

| | |
|---|---|
| Base (fine-tune) | `ukisai/Swift-Qwen3.8-27b` (Swift Open License v1.0) |
| Base (original) | `Qwen/Qwen3.8-27B` (Apache-2.0), Copyright 2026 Alibaba Cloud |
| Architecture | `Qwen3_5ForConditionalGeneration`, 64 layers (hybrid Gated-DeltaNet linear attention + full attention every 4th layer), hidden 5120, 24 Q / 4 KV heads, head_dim 256, 248,320 vocab |
| Quantization | GPTQ, 4-bit weights, 16-bit activations (W4A16), `group_size=128`, `sym=true`, `desc_act=false`, `lm_head` not quantized |
| Preserved unquantized | 15 `mtp.*` tensors (BF16 draft head), 333 `model.visual.*` tensors |
| Quantized modules | 400 (all linear projections of the 64 language layers) |
| Checkpoint size | ~19 GB, 5 safetensors shards (2399 tensors) |
| Native context | 262,144 tokens (validated serving up to 131,072) |
| Quantizer | `gptqmodel 7.3.2` (torch 2.9.1+xpu) |
| Source revision | `048328f4059015b63f860a453bf94834af0db683` |
| Calibration | `HuggingFaceH4/ultrachat_200k` `train_sft[:256]`, truncated to 2048 tokens |
| Calibration revision | `8049631c405ae6576f93f445c6b8166f76f5505a` |
| Repository | quantization recipe + verification: https://github.com/BjornNordblom/intel-arc-b70-quant |

The `quantize_config.json` is field-for-field identical to the community
reference artifact `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` except for
the quant-time-only `meta.offload_to_disk` flag. Note that the reference artifact
quantizes the **base** Qwen3.8-27B; this repository quantizes the **Swift**
fine-tune.

## What was changed vs the source checkpoint

- All linear projections of the language model → GPTQ INT4 (W4A16, g128, sym,
  `desc_act=false`).
- `mtp.*` draft tensors (15) and `model.visual.*` vision tensors (333) kept in
  their original dtype; the `dynamic` exclusion `{"-:.*mtp.*": {}}` in
  `quantize_config.json` records this.
- Added: `quantize_config.json`, `quant_log.csv` (per-module quantization timings).
- Unchanged: `config.json`, `generation_config.json`, `chat_template.jinja`,
  `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `processor_config.json`, `video_preprocessor_config.json`.
- Added licence files: `LICENSE` (Swift Open License v1.0),
  `LICENSE-APACHE-2.0`, `NOTICE`.

## Quantization details

| | |
|---|---|
| Method | GPTQ (gptqmodel 7.3.2), true-sequential, static groups off |
| Bits / group | 4-bit / 128, symmetric, `desc_act=false` |
| Damping | `damp_percent=0.05`, `damp_auto_increment=0.01`, Hessian staging FP32 |
| Fallback | RTN for modules exceeding a 0.5% error threshold (none reported) |
| Calibration | 256 UltraChat-SFT samples, ≤2048 tokens each |
| Pack format | `int32`, `checkpoint_format=gptq`, `lm_head=false` |
| Wall time | 2.40 h on one Arc Pro B70; host i9-13900K, 64 GB RAM |
| Verification | contract check (`VERIFY PASS`), tiny-model smoke test, endpoint + streaming test — see the GitHub repo |

## How to use

### vLLM (Intel XPU) — tested configuration

Requires a vLLM XPU build with Gated-DeltaNet support (this card was tested with
`vllm/vllm-openai-xpu` @ `vllm 0.27.2rc1.dev77+gac7509e2b.xpu`,
`vllm-xpu-kernels 0.1.12.3`).

```bash
vllm serve <this-repo> \
  --quantization gptq --dtype float16 --max-model-len 131072 \
  --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 \
  --max-num-seqs 1 --max-num-batched-tokens 16384 --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --language-model-only
```

Notes for this base model on XPU:

- **MTP draft must be built unquantized.** The checkpoint flags this via the
  `dynamic` exclusion, but the XPU build tested here also needs the draft layer
  built without `quant_config` (upstream patch used by the recipe repo:
  `B70_MTP_BF16_DRAFT=1` gate plus a small metadata patch for the
  max-model-length boundary).
- **`vllm-xpu-kernels` < 0.1.14.1 has a mixed-batch limitation**: batching
  speculative-decode tokens together with prefill tokens aborts the engine
  (`causal_conv1d does not support spec-decode and non-spec ... tokens in the
  same invocation`). Use `--max-num-seqs 1`, or kernels ≥ 0.1.14.1 together with
  a vLLM that also carries the companion vLLM PR #48109 (both are required — the
  kernels release alone is not sufficient), or the Python split-dispatch
  backport used by the recipe repo (a single-function change in `vllm/_xpu_ops.py`).
- `--kv-cache-dtype fp8` is a serving choice, not part of the checkpoint.

### Transformers

GPTQ checkpoints need a GPTQ-capable loader (`gptqmodel` or `auto-gptq`) and
`transformers` ≥ 4.5x. The `mtp.*` tensors are not used by `transformers`
inference and can be ignored.

## Evaluation

What was actually measured on this artifact (host: Arc Pro B70, vLLM XPU,
MTP 3 speculative tokens, fp8 KV cache):

| Check | Result |
|---|---|
| Quantization contract (`verify_quant.py`) | PASS — bits 4, group 128, sym, `desc_act=false`, 15 MTP tensors preserved, 333 vision tensors, 400 modules quantized, `lm_head` untouched |
| Tiny-model smoke test + endpoint/streaming test | PASS |
| Decode, p512/g128, median of 5 | 58.8 tok/s (this artifact) vs 60.4 tok/s (`SergiioB/…` reference quant) |
| MTP draft acceptance | 3.38 / 4 tokens accepted (59.5%) vs 3.43 (60.9%) for the reference quant |
| Solo TTFT / decode (short prompt) | ~0.95 s / ~59 tok/s |
| Concurrency | 12-request storm with 4 × ~10k-token prefills: 12/12 HTTP 200, no engine failure, MTP acceptance ~62% (with the mixed-batch fix above) |

**No standard quality benchmarks (MMLU, GPQA, AIME, IFBench, perplexity) were
run on this quantized artifact by the uploader.** UkisAI publishes INT4 W4A16
evaluations for the base model (GPQA-Diamond 88.38%, IFBench 71.25%, AIME 2026
84.00%) which are indicative but were measured on a different quantization
pipeline, and the reference artifact mentioned above does not replace an
independent evaluation of this checkpoint. Treat accuracy figures as unverified
for this artifact.

## Intended use and out-of-scope

- Intended: local inference and evaluation of the Swift fine-tune on
  Intel XPU / low-VRAM setups, including agent-style tool calling via
  `qwen3_xml`.
- Out of scope: any safety-critical, medical, legal, or production decision
  making; anything requiring verified accuracy on this specific checkpoint;
  vision/multimodal use — the vision tower is preserved but **was not validated**
  (serving was tested language-only), and training data provenance is inherited
  from the base models.
- No safety alignment or red-teaming was performed by the uploader. Behavioural
  risks of the base models (bias, hallucinations, prompt-injection
  susceptibility) also apply here and may be amplified by quantization.

## Limitations

- Quantization is lossy; per-module error was not published beyond the RTN
  fallback threshold report.
- The MTP draft head only works in runtimes that can build it unquantized (see
  above); otherwise disable speculative decoding.
- Exporting to other formats (GGUF/AWQ) from this checkpoint is untested.
- Multimodal inputs are structurally supported but untested.
- Long-context behaviour was validated to 131,072 tokens, not to the native
  262,144.

## Environmental impact

One quantization run: ~2.4 h on a single Arc Pro B70 (230 W board power cap) —
roughly **0.55 kWh** board energy, plus host overhead (i9-13900K, 64 GB RAM).
Serving costs depend on runtime and context length.

## License and attribution

The following notice is required by the Swift Open License v1.0 for
redistributors of quantizations/conversions/merges (verbatim from the base
model's LICENSE appendix):

> Copyright 2026 UkisAI. Swift Contribution licensed under the Swift Open
> License v1.0 (https://huggingface.co/ukisai/Swift-Qwen3.8-27b/blob/main/LICENSE).
> Derivative of Qwen3.8-27B, Copyright 2026 Alibaba Cloud, Apache License 2.0.

Included files: [`LICENSE`](LICENSE) (Swift Open License v1.0, governs the Swift
Contribution), [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0) (governs Qwen3.8-27B
and the files unmodified from it), [`NOTICE`](NOTICE) (what UkisAI changed).

Key terms to be aware of:

- **Commercial-use threshold.** Personal, research, educational, evaluation and
  commercial use are free for individuals and organisations with gross annual
  revenue (including affiliates) up to **US$1,000,000**. Above that threshold,
  commercial use of the Swift Contribution requires a separate Swift Enterprise
  License from UkisAI. Nothing in the Swift Open License limits your rights in
  Qwen3.8-27B itself, which remains Apache-2.0.
- **No trademark rights.** The licence grants no rights to the "UkisAI" name or
  marks; this card uses them only for attribution.
- **Termination.** Non-compliance terminates the licence for the Swift
  Contribution; rights in the base model under Apache-2.0 are unaffected.

Addendum for this quantization (the uploader's modification notice):

> GPTQ INT4 quantization of the Swift Contribution by Bjorn Nordblom, 2026,
> distributed under the same Swift Open License v1.0 terms that apply to the
> Swift Contribution. No additional rights are granted.

## Citation

```bibtex
@misc{swift-qwen3.8-27b-gptq-int4,
  title        = {Swift-Qwen3.8-27B GPTQ INT4 (W4A16, group-128, symmetric) with preserved BF16 MTP head},
  author       = {Nordblom, Bjorn},
  year         = {2026},
  howpublished = {Hugging Face},
  note         = {Quantization of ukisai/Swift-Qwen3.8-27b},
}

@misc{swift-qwen3.8-27b,
  title  = {Swift-Qwen3.8-27B},
  author = {UkisAI},
  year   = {2026},
  url    = {https://huggingface.co/ukisai/Swift-Qwen3.8-27b}
}

@misc{qwen3.8-27b,
  title  = {Qwen3.8-27B},
  author = {Alibaba Cloud},
  year   = {2026},
  url    = {https://huggingface.co/Qwen/Qwen3.8-27B}
}
```

## Acknowledgements

- UkisAI for the Swift fine-tune and the licence terms above.
- Alibaba Cloud / Qwen for Qwen3.8-27B (Apache-2.0).
- The `gptqmodel` project for the quantizer.
- Intel's Arc Pro B70 XPU inference cookbook for the serving patches (MTP BF16
  draft, GDN boundary handling, mixed-batch split-dispatch backport).

## Model card contact

Issues and questions: https://github.com/BjornNordblom/intel-arc-b70-quant/issues
