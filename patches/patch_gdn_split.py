#!/usr/bin/env python3
"""Apply the GDN split-dispatch patch to vllm/_xpu_ops.py.

Backport of the fix for the mixed spec-decode + non-spec batch crash on XPU:

    RuntimeError: causal_conv1d does not support spec-decode and non-spec
    (prefill + decode) tokens in the same invocation

Upstream: https://github.com/vllm-project/vllm-xpu-kernels/issues/510#issuecomment-5301844645
Fixed upstream in vllm-xpu-kernels v0.1.14 (PR vllm-xpu-kernels#537) for builds
that also carry vLLM PR #48109. Our pinned image (kernels 0.1.12.3,
vLLM 0.27.2rc1.dev77+gac7509e2b) predates both, hence this Python-only backport.

The patch code is taken verbatim from patches/gdn_split_dispatch.diff: every
"+" line of the vendored diff becomes the replacement block, and every "-" line
must match the target file exactly. Any mismatch aborts without writing.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import pathlib
import py_compile
import sys

PINNED_SHA256 = "47080ce161348db461ac6777f818742a067ef6db5cb64a67165864fbad21d3e1"
TARGET_DEFAULT = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
MARKER = "B70_GDN_SPLIT_DISPATCH"
DIFF_PATH = pathlib.Path(__file__).with_name("gdn_split_dispatch.diff")
FN_NAME = "_gdn_attention_core_xpu_impl"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_blocks() -> tuple[str, str]:
    """Return (old_block, new_block) reconstructed from the vendored unified diff.

    Context lines belong to both sides; '-' lines only to the old block and '+'
    lines only to the new block. (The closing ')' of the original call is a
    context line in the vendored diff, so a naive +/- split would drop it.)
    """
    old: list[str] = []
    new: list[str] = []
    in_hunk = False
    for ln in DIFF_PATH.read_text().splitlines():
        if ln.startswith("@@"):
            in_hunk = True
            continue
        if ln.startswith("diff --git") or ln.startswith("index ") or ln.startswith("---") or ln.startswith("+++"):
            in_hunk = False
            continue
        if not in_hunk:
            continue
        if ln.startswith("-"):
            old.append(ln[1:])
        elif ln.startswith("+"):
            new.append(ln[1:])
        elif ln.startswith(" ") or ln == "":
            body = ln[1:] if ln.startswith(" ") else ""
            old.append(body)
            new.append(body)
        else:
            sys.exit(f"patch: unexpected diff line: {ln!r}")
    if not old or not new:
        sys.exit(f"patch: no hunk content found in {DIFF_PATH}")
    return "\n".join(old), "\n".join(new)


def verify_structure(source: str, expect_split: bool) -> None:
    """AST check: the target function exists (and contains the nested split helper)."""
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FN_NAME]
    if len(funcs) != 1:
        sys.exit(f"patch: expected exactly one top-level {FN_NAME}(), found {len(funcs)}")
    nested = {n.name for n in funcs[0].body if isinstance(n, ast.FunctionDef)}
    if expect_split and "_invoke" not in nested:
        sys.exit("patch: patched function has no nested _invoke() helper")
    if not expect_split and "_invoke" in nested:
        sys.exit("patch: unpatched function unexpectedly defines _invoke()")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_DEFAULT, help="path to vllm/_xpu_ops.py")
    ap.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="skip the pinned-hash check (only for reviewing the replacement on a copy)",
    )
    args = ap.parse_args()

    path = pathlib.Path(args.target)
    if not path.exists():
        sys.exit(f"patch: target not found: {path}")

    raw = path.read_bytes()
    before = sha256(raw)
    text = raw.decode()

    if MARKER in text:
        print(f"already patched ({MARKER}) — no-op")
        return

    verify_structure(text, expect_split=False)

    if before != PINNED_SHA256 and not args.allow_hash_mismatch:
        sys.exit(
            "patch: refusing to patch unexpected file\n"
            f"  expected sha256 {PINNED_SHA256}\n"
            f"  actual   sha256 {before}"
        )

    old_block, new_block = load_blocks()
    if text.count(old_block) != 1:
        sys.exit(
            f"patch: anchor block not found exactly once (count={text.count(old_block)}); "
            "the vendored diff and the target file disagree"
        )

    # insert the marker as a real comment line before the nested helper
    # (the vendored hunk ends with trailing context, so do not append to the block)
    patched = text.replace(old_block, new_block, 1)
    helper_anchor = "\n    def _invoke("
    if patched.count(helper_anchor) != 1:
        sys.exit(f"patch: expected exactly one nested _invoke definition, found {patched.count(helper_anchor)}")
    patched = patched.replace(
        helper_anchor, f"\n    # {MARKER}\n    def _invoke(", 1
    )

    verify_structure(patched, expect_split=True)
    if patched.count(MARKER) != 1:
        sys.exit("patch: marker count != 1 after replacement")

    path.write_text(patched)
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        path.write_bytes(raw)
        sys.exit(f"patch: py_compile failed, file restored — {exc}")

    after = sha256(path.read_bytes())
    print(f"patched {path}")
    print(f"  sha256 {before} -> {after}")
    print(f"  marker {MARKER}")


if __name__ == "__main__":
    main()
