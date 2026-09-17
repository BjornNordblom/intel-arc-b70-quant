# patches/

MTP runtime patches applied inside the container at launch time. Both are
vendored **verbatim** (unmodified) from a third-party MIT-licensed repository so
work from a plain clone with no external checkout.

| file | sha256 | purpose |
|---|---|---|
| `patch_mtp_nightly.py` | `4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14` | Builds the MTP draft layer unquantized when `B70_MTP_BF16_DRAFT=1` (the checkpoint's `dynamic` exclusion only triggers for checkpoints that set it; ours has `dynamic=null`). Marker: `B70_MTP_NIGHTLY_DRAFT`. |
| `patch_mtp_boundary.py` | `41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50` | Handles the truncated final speculative group at `max-model-len` by reclassifying it as a stateful non-spec prefill step, which the XPU GDN kernel accepts. Marker: `B70_MTP_PARTIAL_FINAL_GROUP`. |

Provenance:

```
repo:    https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
commit:  3beb704b5b86baed2a874a8cc96821116c97e080 (2026-08-16)
path:    patches/
license: MIT, Copyright (c) 2026 SergiioB
```

Full MIT notice: [`LICENSE.SergiioB`](LICENSE.SergiioB).

`launch.sh` mounts these files read-only and executes them at container start
(`python /patch_mtp.py; python /patch_boundary.py`), before `vllm serve`. Override
their location with `PATCH_DIR=...` if you keep your own copies.

Also here (first-party, MIT):

- `patch_gdn_split.py` + `gdn_split_dispatch.diff` + `gdn_split_dispatch.upstream.md`
  — the GDN split-dispatch backport applier and its vendored upstream diff;
  see `../MAX_NUM_SEQ_FIX.md`.
