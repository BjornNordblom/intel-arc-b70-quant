"""GPTQ Int4 sym G128 quantization of Swift-Qwen3.8-27b for Intel Arc XPU (W4A16).

Contract: bits=4, group_size=128, sym=True, desc_act=False, lm_head not quantized,
and `dynamic={"-:.*mtp.*": {}}` so the MTP draft head stays unquantized. Vision
and MTP handling otherwise comes from gptqmodel internals (module_tree /
out_of_model_tensors).

Calibration: HuggingFaceH4/ultrachat_200k train_sft[:256], rendered with the
Swift ChatML template, truncated to 2048 tokens per sample. Output dir:
./Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16

Run (XPU, gptqmodel 7.3.2 - the .venv-xpu venv; the CPU .venv has 7.5.0 and is
only for smoke_test.py/verify_quant.py):
    DEVICE=xpu .venv-xpu/bin/python -u quant_swift.py

Env:
  MODEL_ID         source checkpoint dir (default /opt/models/Swift-Qwen3.8-27b)
  SOURCE_REVISION  source Hub commit (default 048328f4059015b63f860a453bf94834af0db683)
  CALIB_REVISION   calibration Hub commit (default 8049631c405ae6576f93f445c6b8166f76f5505a)
  OUTPUT_DIR       output dir (default ./Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
  DEVICE           torch device for quantization (default cpu, "xpu" on the Arc Pro B70).
"""
import json
import os
import time

from datasets import load_dataset
from transformers import AutoProcessor
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import FORMAT

MODEL_ID = os.environ.get("MODEL_ID", "/opt/models/Swift-Qwen3.8-27b")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16")
SOURCE_REVISION = os.environ.get("SOURCE_REVISION", "048328f4059015b63f860a453bf94834af0db683")
CALIB_DATASET = "HuggingFaceH4/ultrachat_200k"
CALIB_REVISION = os.environ.get("CALIB_REVISION", "8049631c405ae6576f93f445c6b8166f76f5505a")
CALIB_SAMPLES = 256
MAX_SEQ_LEN = 2048  # config max is 262144 -> unbounded, must cap for runtime
DEVICE = os.environ.get("DEVICE", "cpu")  # "xpu" on the Arc Pro B70

if os.path.realpath(MODEL_ID) == os.path.realpath(OUTPUT_DIR):
    raise SystemExit("MODEL_ID and OUTPUT_DIR must refer to different paths")
if os.path.exists(OUTPUT_DIR):
    raise SystemExit(f"OUTPUT_DIR already exists; choose an empty path: {OUTPUT_DIR}")

t0 = time.time()
import torch
print(f"device={DEVICE} torch={torch.__version__}", flush=True)
print(f"source_revision={SOURCE_REVISION} calibration_revision={CALIB_REVISION}", flush=True)

# 1. Processor / tokenizer
processor = AutoProcessor.from_pretrained(
    MODEL_ID, revision=SOURCE_REVISION, trust_remote_code=True
)

# 2. Calibration: multi-turn chat formatted with Swift's ChatML template
ds = load_dataset(
    CALIB_DATASET,
    revision=CALIB_REVISION,
    split=f"train_sft[:{CALIB_SAMPLES}]",
)
tok = processor.tokenizer
calibration_dataset = []
for ex in ds:
    text = processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
    ids = tok(text, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    calibration_dataset.append(tok.decode(ids, skip_special_tokens=False))
print(f"calibration samples: {len(calibration_dataset)} "
      f"(max chars {max(len(s) for s in calibration_dataset)})", flush=True)

# 3. Contract: W4A16, group 128, symmetric, desc_act off (Intel XPU kernel path)
quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,
    format=FORMAT.GPTQ,
    device=DEVICE,                  # XPU (Arc Pro B70) or cpu
    dynamic={
        # keep MTP draft head out of quantization (gptqmodel also copies mtp.* verbatim)
        "-:.*mtp.*": {},
    },
)

# 4. Load on CPU (bf16, offload_to_disk=True default keeps peak RAM bounded)
print("Loading Swift-Qwen3.8-27b...", flush=True)
model = GPTQModel.from_pretrained(
    MODEL_ID,
    quantize_config=quantize_config,
    revision=SOURCE_REVISION,
    trust_remote_code=True,
)

# 5. Quantize language backbone (vision tower is not in module_tree -> bf16 passthrough)
print("Quantizing backbone...", flush=True)
model.quantize(calibration_dataset, batch_size=1)

# 6. Save
print(f"Saving to {OUTPUT_DIR}...", flush=True)
model.save(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json")) as fh:
    keys = list(json.load(fh)["weight_map"])
print("mtp keys preserved:", len([k for k in keys if k.startswith("mtp")]))
print("visual keys:", len([k for k in keys if ".visual." in k]))
print("quantized (.qweight):", len([k for k in keys if k.endswith(".qweight")]))
print(f"done in {(time.time() - t0) / 3600:.2f} h", flush=True)
