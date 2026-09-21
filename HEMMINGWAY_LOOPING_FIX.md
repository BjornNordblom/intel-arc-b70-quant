# HEMMINGWAY_LOOPING_FIX — "Hemmingway-1 repeats itself / never answers"

Status: **EXECUTED 2026-09-21** — `hem-b70-mtp` relaunched with
`--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'`.
Outcome: plain requests answer directly, thinking requests keep reasoning out of
`content`. Verdict: **not quant damage** — the official bf16 base behaves
identically.

Runtime note (2026-09-21): the repo default moved to the stock
`vllm/vllm-openai-xpu:nightly` (`0.29.1rc1.dev422`) plus `patches/`;
`--reasoning-parser qwen3` works there unchanged (reasoning lands in the
`reasoning` field). The derived image remains the legacy fallback.

Owner: Bjorn
Related: `hf-hemmingway1/upload/chat_template.jinja`, `hf-hemmingway1/UPLOAD_PLAN.md`,
container `hem-b70-mtp`, `patches/patch_mtp_nightly.py`

---

## 1. Symptom

Simple commands sent to the Hemmingway-1 GPTQ quant produced a self-referential
planning monologue that restated the same points, appeared to repeat, and often
never reached an answer. Suspect was the GPTQ quant (linear_attention layers).

## 2. Root cause — three stacked problems

1. **Template defaults to thinking ON.** `chat_template.jinja` only emits an
   empty think block when the caller passes `enable_thinking=false`; otherwise it
   injects a system message ("Reasoning effort is set to xhigh…") and opens
   `thinking`. Every request that does not opt out runs in think mode. This is a
   model-level default, not a serving setting.

2. **No reasoning parser on the server.** `hem-b70-mtp` ran without
   `--reasoning-parser`, so the generated reasoning is returned in
   `message.content` (`</think>` is a special token and is stripped). Clients
   store that text as the assistant message and replay it next turn; the model
   then reasons about its own previous reasoning. That is the repeat spiral.
   Reproduced: by turn 3 the model was re-litigating its turn-2 answer instead of
   answering the new question.

3. **Fine-tune expects an unpublished harness prompt.** The model repeatedly
   invents a "system-reminder" to work through *nine planning steps*
   ("constraints, two shapes, premise check, grounding, language traps, facts
   owned, reread, premise check, grounding"). That text appears in neither the
   chat template nor the config — it is a training-time harness prompt from the
   authors' app. Without it the model can spend the whole token budget planning
   (`finish_reason=length`) and never produce the answer. `generation_config.json`
   overriding sampling to `temperature 1.0 / top_k 20` makes the rumination
   worse.

## 3. Evidence

| test | result |
|---|---|
| Official `Altworld/Hemmingway-1` bf16, CPU transformers, same template/prompt, thinking default | same nine-step monologue as the GPTQ model |
| GPTQ quant, `Reply with exactly: hello`, thinking off | `hello`, 2 tokens, `stop`, 5/5 seeds |
| GPTQ quant, tool call (`ls`) | valid `tool_calls`, thinking on or off |
| GPTQ quant, 1500-token article | coherent, no repeated n-grams |
| quant_log.csv linear_attn losses | 1e-7…3e-5 — nothing pathological |
| `speculative_config` in the run | `None` — MTP not active, not a loop source |
| `qwen3` reasoning parser on this vLLM build | splits reasoning/content; `is_reasoning_end(</think>)` = True |

Multi-turn feedback simulation (thinking on, client stores content verbatim):

- T1: planning monologue + answer (683 tokens)
- T2: re-plans from scratch, "let me work through the nine steps properly"
- T3: "This is a repeat of the second request… I answered Fig, plum, persimmon"
  — model works on its own prior output, not the user request

## 4. Fix

Server flags (applied to `hem-b70-mtp`):

```bash
--reasoning-parser qwen3 \
--default-chat-template-kwargs '{"enable_thinking": false}'
```

- `enable_thinking=false` by default → direct answers, no planning monologue.
- `--reasoning-parser qwen3` → if a client opts into thinking, reasoning lands in
  the `reasoning` field (`reasoning_content` for OpenAI-compatible clients that
  read it) and `content` stays clean, breaking the feedback spiral.

Per-request, without a restart:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

`temperature` around 0.7 is recommended for the writing use case; the
`generation_config.json` default of 1.0 is where the monologue degrades fastest.

## 5. What was NOT the cause

- GPTQ on `linear_attention.{in_proj_qkv,in_proj_z,out_proj}` — base model shows
  the same behaviour, quant outputs are otherwise clean and exact.
- MTP / speculative decoding — `speculative_config=None` in this run.
- fp8 KV cache, prefix caching, GDN split-dispatch patches — no involvement in
  the reproductions (deterministic with cache hits at 0%).

## 6. Execution log

```
docker stop hem-b70-mtp && docker rm hem-b70-mtp
docker run -d --name hem-b70-mtp --network host --ipc host \
  --device /dev/dri --group-add 991 \
  -e TRITON_CACHE_DIR=/workspace/triton_cache \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /opt/models/triton_cache:/workspace/triton_cache \
  -v .../hf-hemmingway1/upload:/model \
  -v .../patches/patch_mtp_nightly.py:/patch_mtp.py \
  -v .../patches/patch_mtp_boundary.py:/patch_boundary.py \
  --entrypoint bash vllm-xpu-gdn-split:0.1.12.3-p1 -lc \
  "set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model \
    --quantization gptq --dtype float16 --max-model-len 131072 \
    --gpu-memory-utilization 0.94 --kv-cache-dtype fp8 --port 8080 \
    --max-num-seqs 1 --max-num-batched-tokens 8192 --enable-prefix-caching \
    --served-model-name hemmingway1 --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{\"enable_thinking\": false}'"
```

Verification after restart is recorded in section 7 below.

## 7. Verification (2026-09-21, run `2026-09-20_212029-hem-b70-mtp-seqs1`)

| request | before | after |
|---|---|---|
| `Reply with exactly: hello` | nine-step monologue, no answer | `hello`, 2 tokens, `stop` |
| landlord text, no kwargs | monologue, often `finish_reason=length` | clean message, 50 tokens, `stop` |
| `enable_thinking: true` | reasoning in `content` | monologue in `reasoning`, `content` empty until think ends |
| tool call (developer role + shell tool) | worked | still works |
