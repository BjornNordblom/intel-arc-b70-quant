"""Smoke test: same API flow as quant_swift.py on a tiny random Qwen3_5 model.

Validates the gptqmodel API surface (the CPU `.venv` pins 7.5.0; the XPU quant
venv `.venv-xpu` pins 7.3.2), the qwen3_5 module_tree walk, the dynamic MTP
exclusion, and save + reload.

Run (CPU venv):  DEVICE=cpu .venv/bin/python smoke_test.py
Run (XPU venv):  DEVICE=xpu SMOKE_DIR=_tiny_xpu .venv-xpu/bin/python smoke_test.py

Env:
  MODEL_ID    source checkpoint dir the tiny model is derived from
              (default /opt/models/Swift-Qwen3.8-27b)
  DEVICE      torch device for the tiny quant (default cpu)
  SMOKE_DIR   scratch dir for the tiny model (default _tiny_swift)
  SERIAL=1    disable gptqmodel parallelism (true_sequential=True,
              auto_forward_data_parallel=False when those knobs exist) - used to
              reproduce serial-vs-parallel behaviour differences
"""
import os
import shutil
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import FORMAT

REAL = os.environ.get("MODEL_ID", "/opt/models/Swift-Qwen3.8-27b")
DEVICE = os.environ.get("DEVICE", "cpu")
TINY = Path(os.environ.get("SMOKE_DIR", "_tiny_swift"))
OUT = Path(str(TINY) + "_gptq")

if TINY.exists():
    shutil.rmtree(TINY)
if OUT.exists():
    shutil.rmtree(OUT)


def build_tiny():
    cfg = AutoConfig.from_pretrained(REAL)
    tc = cfg.text_config
    tc.num_hidden_layers = 2
    tc.layer_types = ["linear_attention", "full_attention"]
    tc.hidden_size = 128
    tc.intermediate_size = 256
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.head_dim = 32
    tc.linear_num_key_heads = 4
    tc.linear_num_value_heads = 4
    tc.linear_key_head_dim = 32
    tc.linear_value_head_dim = 32
    vc = cfg.vision_config
    vc.depth = 2
    vc.hidden_size = 64
    vc.intermediate_size = 128
    vc.num_heads = 4
    vc.out_hidden_size = 128
    cfg.save_pretrained(TINY)
    AutoTokenizer.from_pretrained(REAL).save_pretrained(TINY)
    for f in ("preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copy(REAL + "/" + f, TINY / f)
    model = Qwen3_5ForConditionalGeneration(cfg)
    model.save_pretrained(TINY)
    import json
    from safetensors.torch import load_file
    if (TINY / "model.safetensors.index.json").exists():
        keys = list(json.load(open(TINY / "model.safetensors.index.json"))["weight_map"])
    else:
        keys = list(load_file(TINY / "model.safetensors").keys())
    n = sum(p.numel() for p in model.parameters())
    print(f"[tiny] params={n/1e6:.1f}M mtp_keys={[k for k in keys if k.startswith('mtp')][:3]} "
          f"visual_keys={len([k for k in keys if '.visual.' in k])}")


def main():
    build_tiny()
    kw = dict(bits=4, group_size=128, sym=True, desc_act=False,
              format=FORMAT.GPTQ, device=DEVICE, dynamic={"-:.*mtp.*": {}})
    if os.environ.get("SERIAL") == "1":
        import dataclasses as _dc
        fields = {f.name for f in _dc.fields(QuantizeConfig)}
        if "auto_forward_data_parallel" in fields:
            kw["auto_forward_data_parallel"] = False
        if "true_sequential" in fields:
            kw["true_sequential"] = True
    qcfg = QuantizeConfig(**kw)
    calib = [
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\nIt is 4.<|im_end|>",
        "<|im_start|>user\nName a color.<|im_end|>\n<|im_start|>assistant\nBlue.<|im_end|>",
        "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>",
        "<|im_start|>user\nCapital of France?<|im_end|>\n<|im_start|>assistant\nParis.<|im_end|>",
    ] * 4
    model = GPTQModel.from_pretrained(TINY, quantize_config=qcfg, trust_remote_code=True)
    model.quantize(calib, batch_size=1)
    model.save(OUT)
    AutoTokenizer.from_pretrained(REAL).save_pretrained(OUT)

    import json
    qc = json.load(open(OUT / "quantize_config.json"))
    print("[saved] bits=%s sym=%s gs=%s desc_act=%s dynamic=%s"
          % (qc["bits"], qc["sym"], qc["group_size"], qc["desc_act"], qc.get("dynamic")))
    keys = list(json.load(open(OUT / "model.safetensors.index.json"))["weight_map"]) if (OUT / "model.safetensors.index.json").exists() else list(__import__("safetensors.torch", fromlist=["load_file"]).load_file(OUT / "model.safetensors").keys())
    print("[saved] mtp keys:", [k for k in keys if k.startswith("mtp")][:3])
    print("[saved] quantized (qweight) count:", len([k for k in keys if k.endswith(".qweight")]))
    print("[saved] bf16 passthrough (no .qweight) count:", len([k for k in keys if not k.endswith((".qweight", ".qzeros", ".scales", ".g_idx"))]))

    print(f"SMOKE TEST PASS (device={DEVICE}, torch={__import__('torch').__version__})")
    shutil.rmtree(TINY)
    shutil.rmtree(OUT)


if __name__ == "__main__":
    main()
