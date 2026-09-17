"""Post-quant verification for Swift GPTQ output.

Checks quantize_config.json contract + that MTP draft tensors survived unquantized.
Run: .venv/bin/python verify_quant.py
"""
import json
import sys
from pathlib import Path

OUT = "Swift-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
EXPECT = {
    "bits": 4,
    "group_size": 128,
    "sym": True,
    "desc_act": False,
    "format": "gptq",
    "lm_head": False,
}
EXPECTED_COUNTS = {"mtp": 15, "visual": 333, "qweight": 400}

qc = json.load(open(f"{OUT}/quantize_config.json"))
fails = []

for k, want in EXPECT.items():
    got = qc.get(k)
    print(f"{k:12} = {got!r:8} (want {want!r})", "OK" if got == want else "FAIL")
    if got != want:
        fails.append(f"{k}: got {got!r}, want {want!r}")

dyn = qc.get("dynamic") or {}
mtp_patterns = [p for p in dyn if "mtp" in p and p.startswith("-:")]
print(f"\ndynamic      = {json.dumps(dyn)}")
print(f"mtp exclusion patterns: {mtp_patterns or 'NONE'}")
if not mtp_patterns:
    fails.append("no negative (exclusion) dynamic pattern covering mtp")

idx_path = f"{OUT}/model.safetensors.index.json"
try:
    weight_map = json.load(open(idx_path))["weight_map"]
    keys = list(weight_map)
    indexed_files = set(weight_map.values())
    out_path = Path(OUT)
    missing_files = sorted(f for f in indexed_files if not (out_path / f).is_file())
    unindexed_files = sorted(
        p.name for p in out_path.glob("*.safetensors") if p.name not in indexed_files
    )
    if missing_files:
        fails.append(f"index references missing shards: {missing_files}")
    if unindexed_files:
        fails.append(f"unindexed safetensor shards present: {unindexed_files}")
except FileNotFoundError:
    import glob
    from safetensors.torch import load_file
    files = glob.glob(f"{OUT}/*.safetensors")
    keys = [k for f in files for k in load_file(f)]

mtp = [k for k in keys if k.startswith("mtp")]
mtp_quantized = [k for k in mtp if k.endswith((".qweight", ".qzeros", ".scales", ".g_idx"))]
visual = [k for k in keys if ".visual." in k]
lm_head_q = [k for k in keys if k.startswith("lm_head") and k.endswith(".qweight")]
qw = [k for k in keys if k.endswith(".qweight")]

print(f"\nmtp tensors preserved : {len(mtp)} (quantized: {len(mtp_quantized)})")
print(f"visual tensors        : {len(visual)}")
print(f"lm_head quantized     : {bool(lm_head_q)}")
print(f"quantized modules     : {len(qw)}")

if len(mtp) != EXPECTED_COUNTS["mtp"]:
    fails.append(f"mtp tensor count: got {len(mtp)}, want {EXPECTED_COUNTS['mtp']}")
if mtp_quantized:
    fails.append(f"mtp tensors quantized: {mtp_quantized[:3]}")
if len(visual) != EXPECTED_COUNTS["visual"]:
    fails.append(f"visual tensor count: got {len(visual)}, want {EXPECTED_COUNTS['visual']}")
if lm_head_q:
    fails.append("lm_head was quantized (lm_head=false expected)")
if len(qw) != EXPECTED_COUNTS["qweight"]:
    fails.append(f"quantized module count: got {len(qw)}, want {EXPECTED_COUNTS['qweight']}")

print()
if fails:
    print("VERIFY FAIL")
    [print("  -", f) for f in fails]
    sys.exit(1)
print("VERIFY PASS")
