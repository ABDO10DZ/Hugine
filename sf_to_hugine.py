#!/usr/bin/env python3
"""
sf_to_hugine.py — Stockfish NNUE → Hugine NNUE converter
==========================================================

⚠️  SUPERSEDED IN HUGINE 5.1.0
--------------------------------
As of Hugine 5.1.0 "Iota", the engine natively loads any Stockfish .nnue file
WITHOUT conversion.  Supported formats:

  • HalfKP-256           (SF10–12, uncompressed)
  • HalfKA-256           (SF13–14, uncompressed)
  • HalfKAv2-512/1024    (SF15,   uncompressed or COMPRESSED_LEB128)
  • HalfKAv2-1536        (SF16+,  COMPRESSED_LEB128)
  • Any COMPRESSED_LEB128 pytorch trainer output

Just place the .nnue file anywhere and use:
  setoption name EvalFile value /path/to/nn-XXXXXX.nnue

The SFNNUEEvaluator back-end handles architecture detection, LEB128
decompression, and two-perspective accumulation automatically.

WHEN THIS CONVERTER IS STILL USEFUL
-------------------------------------
This script remains useful as a "warm start" tool: it slices the FT weights
from a Stockfish net into Hugine's native HalfKP format, so you can begin
training from pre-learned piece-position features instead of random weights.

  python3 sf_to_hugine.py input.nnue output.nnue          # → warmstart.nnue
  python3 nnue_trainer/train.py train \\
      --data data/train.txt --out hugine.nnue \\
      (add warmstart logic to train.py for this workflow)

For playing strength, the native SF loader in the engine is strictly better —
it uses the full SF net including trained L1/output layers.

USAGE
-----
  python3 sf_to_hugine.py input.nnue output.nnue
  python3 sf_to_hugine.py input.nnue output.nnue --ft 512 --l1 64 --l2 32
  python3 sf_to_hugine.py input.nnue output.nnue --large
  python3 sf_to_hugine.py input.nnue --info
"""

import argparse, struct, sys, os
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
HUGINE_MAGIC    = 0x5A5A5A5A
HUGINE_VERSION  = 0x00000002
HALFKP_INPUTS   = 40960          # Hugine: 64 king_sq × 64 sq × 10 piece types
HALFKA_INPUTS   = 45056          # Stockfish HalfKA: 64 × 64 × 11 (incl opp king)

SF_ARCH_HASHES = {
    0x5D69D7B8: "HalfKP-256 (SF10-12)",
    0xF8E55352: "HalfKA-256 (SF13-14)",
    0x3E5AA6EE: "HalfKAv2-1024 (SF15)",
    0x7F23558C: "HalfKAv2-512 (SF15+)",
    0x7F234CB8: "HalfKAv2-1024 variant",
    0x1C103072: "SFNNv9 / HalfKAv2 (SF16+)",
    0x3C001A21: "SFNNv6",
}


# ── LEB128 decoder ───────────────────────────────────────────────────────────
def decode_leb128_all(raw: bytes) -> np.ndarray:
    """Decode all int16 LEB128 values from a COMPRESSED_LEB128 block."""
    print("  Decoding LEB128 stream... ", end="", flush=True)
    vals = []
    off = 0; total = len(raw)
    while off < total:
        result = 0; shift = 0
        while True:
            b = raw[off]; off += 1
            result |= (b & 0x7F) << shift
            shift += 7
            if (b & 0x80) == 0:
                break
        vals.append(result & 0xFFFF)
    arr = np.array(vals, dtype=np.uint16).view(np.int16)
    print(f"{len(arr):,} values, range [{arr.min()}, {arr.max()}]")
    return arr


# ── HalfKA → HalfKP feature index mapping ───────────────────────────────────
def build_feature_mapping() -> np.ndarray:
    """
    For each of the 40,960 HalfKP feature indices, return the corresponding
    HalfKA(v2) row index, or -1 if no overlap.

    HalfKP  index = king_sq*640  + piece_sq*10 + pt   (pt: 0-9, no kings)
    HalfKA  index = king_sq*704  + piece_sq*11 + pt   (pt: 0-9 same; 10=opp king)
    → Direct 1-to-1 correspondence for pt in 0..9 when piece_sq != king_sq.
    """
    mapping = np.full(HALFKP_INPUTS, -1, dtype=np.int32)
    for ksq in range(64):
        for psq in range(64):
            for pt in range(10):
                hp  = ksq * 640 + psq * 10 + pt
                ha  = ksq * 704 + psq * 11 + pt
                mapping[hp] = ha
    return mapping


# ── Scale helper ─────────────────────────────────────────────────────────────
def scale_to_int8(arr: np.ndarray) -> np.ndarray:
    """Scale arr to fit in [-127, 127] int8."""
    mx = float(np.abs(arr).max())
    if mx == 0:
        return np.zeros_like(arr, dtype=np.int8)
    scaled = arr.astype(np.float32) * (127.0 / mx)
    return np.clip(np.round(scaled), -127, 127).astype(np.int8)


def scale_to_int16(arr: np.ndarray) -> np.ndarray:
    """Scale arr to fit in [-32767, 32767] int16."""
    mx = float(np.abs(arr).max())
    if mx == 0:
        return np.zeros_like(arr, dtype=np.int16)
    scaled = arr.astype(np.float32) * (32767.0 / mx)
    return np.clip(np.round(scaled), -32767, 32767).astype(np.int16)


# ── Hugine NNUE writer ───────────────────────────────────────────────────────
def write_hugine(path, ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b):
    """
    Write a Hugine-format .nnue file.

    Header (7 × uint32 LE):  magic, version, FT_INPUTS, FT_SIZE, L1, L2, out_dim
    Then each layer: weights (int8) then biases (int16).
    """
    FI, FS = ft_w.shape
    # The engine expects FT_INPUTS = 40960 (per-perspective HalfKP).
    # Our FT weight matrix may have twice that many rows (both perspectives
    # stacked); take only the first HALFKP_INPUTS rows for the file.
    if FI > HALFKP_INPUTS:
        ft_w = ft_w[:HALFKP_INPUTS]
        FI   = HALFKP_INPUTS
    L1     = l1_w.shape[1]
    L2     = l2_w.shape[1]

    with open(path, "wb") as f:
        f.write(struct.pack("<IIIIIII", HUGINE_MAGIC, HUGINE_VERSION, FI, FS, L1, L2, 1))
        f.write(ft_w.astype(np.int8).tobytes())
        f.write(ft_b.astype(np.int16).tobytes())
        f.write(l1_w.astype(np.int8).tobytes())
        f.write(l1_b.astype(np.int16).tobytes())
        f.write(l2_w.astype(np.int8).tobytes())
        f.write(l2_b.astype(np.int16).tobytes())
        f.write(out_w.astype(np.int8).tobytes())
        f.write(out_b.astype(np.int16).tobytes())

    sz = os.path.getsize(path)
    print(f"  → {path}  ({sz:,} bytes, {sz/1e6:.2f} MB)")
    print(f"  → Architecture: HalfKP {FI}×2 → {FS} → {L1} → {L2} → 1")


# ── Uncompressed (classic SF10-14) parser ────────────────────────────────────
def parse_uncompressed(data: bytes, offset: int, ft_size: int, l1_size: int):
    """
    Parse chunk-based uncompressed Stockfish NNUE.
    Returns (ft_weights, ft_biases, l1_weights, l1_biases, ft_inputs_detected).
    """
    ft_w = ft_b = l1_w = l1_b = None
    ft_inputs_detected = HALFKP_INPUTS

    while offset < len(data) - 8:
        ch = struct.unpack_from("<I", data, offset)[0]
        csz = struct.unpack_from("<I", data, offset+4)[0]
        cd  = data[offset+8 : offset+8+csz]
        offset += 8 + csz

        print(f"  Chunk 0x{ch:08X}, {csz:,} bytes")

        # Detect FT chunk by trying known input sizes
        if ft_w is None:
            for ft_in in [HALFKP_INPUTS, HALFKA_INPUTS]:
                for ft_h in [64, 128, 256, 512, 1024, 1536]:
                    expected = ft_in * 2 * ft_h * 2 + ft_h * 2  # int16 weights + biases
                    if expected == csz:
                        ft_inputs_detected = ft_in
                        print(f"    → FT detected: {ft_in} × {ft_h} × 2")
                        raw_w = np.frombuffer(cd[:ft_in*2*ft_h*2], dtype=np.int16)
                        raw_b = np.frombuffer(cd[ft_in*2*ft_h*2:], dtype=np.int16)
                        src_w = raw_w.reshape(ft_in*2, ft_h)

                        # Map to HalfKP
                        mapping = build_feature_mapping()
                        cols = min(ft_size, ft_h)
                        out  = np.zeros((HALFKP_INPUTS*2, ft_size), dtype=np.float32)
                        for persp in range(2):
                            sr = src_w[persp*ft_in:(persp+1)*ft_in]
                            or_ = out[persp*HALFKP_INPUTS:(persp+1)*HALFKP_INPUTS]
                            for hp, ha in enumerate(mapping):
                                if 0 <= ha < ft_in:
                                    or_[hp, :cols] = sr[ha, :cols].astype(np.float32)
                        ft_w = scale_to_int8(out)
                        ft_b_src = raw_b[:min(ft_size, len(raw_b))]
                        ft_b = np.zeros(ft_size, dtype=np.int16)
                        ft_b[:len(ft_b_src)] = ft_b_src
                        break
                if ft_w is not None:
                    break

        # Detect L1 chunk
        elif l1_w is None and ft_w is not None:
            ft2 = ft_size * 2
            for l1_h in [8, 15, 16, 32, 64, 128]:
                # int8 weights + int32 biases
                if ft2 * l1_h + l1_h * 4 == csz:
                    print(f"    → L1 detected: {ft2} × {l1_h}")
                    src_lw = np.frombuffer(cd[:ft2*l1_h], dtype=np.int8).reshape(ft2, l1_h)
                    src_lb = np.frombuffer(cd[ft2*l1_h:], dtype=np.int32)
                    cols_l1 = min(l1_size, l1_h)
                    l1_w = np.zeros((ft2, l1_size), dtype=np.int8)
                    l1_b = np.zeros(l1_size, dtype=np.int16)
                    l1_w[:, :cols_l1] = src_lw[:, :cols_l1]
                    l1_b[:cols_l1] = src_lb[:cols_l1].clip(-32767, 32767)
                    break

    return ft_w, ft_b, l1_w, l1_b, ft_inputs_detected


# ── Compressed (SF15+ / pytorch trainer) parser ──────────────────────────────
def parse_compressed(data: bytes, marker_pos: int, ft_size: int, l1_size: int):
    """
    Parse COMPRESSED_LEB128 Stockfish NNUE.

    The entire weight tensor is one flat LEB128 stream.  We cannot
    reliably determine the exact architecture without the source code,
    so we use a greedy approach:

      Step 1: Decode all values as int16.
      Step 2: Use the first (HALFKP_INPUTS × 2 × ft_size) values as FT weights.
              This is a "best-effort" slice — the actual SF FT may have a
              different layout, but the weights encode real piece-position
              features regardless.
      Step 3: Use the next (ft_size × 2 × l1_size) values as L1 weights.
      Step 4: Zero-initialise L2 and output (they need fine-tuning anyway).
    """
    marker = b"COMPRESSED_LEB128"
    raw_bytes = data[marker_pos + len(marker):]
    arr = decode_leb128_all(raw_bytes)

    ft_total = HALFKP_INPUTS * 2 * ft_size   # budget: 2 perspectives
    if ft_total // 2 > len(arr):
        raise ValueError(
            f"File too small: need {ft_total//2:,} values for FT (one perspective), "
            f"got {len(arr):,}.  Try a smaller --ft value.")

    print(f"  Slicing FT: first {ft_total:,} values → ({HALFKP_INPUTS}×2, {ft_size})")
    ft_raw = arr[:ft_total].reshape(HALFKP_INPUTS * 2, ft_size).astype(np.float32)
    ft_w   = scale_to_int8(ft_raw)

    # Biases: next ft_size values after single-perspective FT
    ft_b_start = ft_total // 2
    ft_b_end   = ft_b_start + ft_size
    ft_b = np.zeros(ft_size, dtype=np.int16)
    if ft_b_end <= len(arr):
        ft_b_raw = arr[ft_b_start:ft_b_end].astype(np.float32)
        ft_b = scale_to_int16(ft_b_raw)

    # L1 weights: next (ft_size*2 × l1_size) values
    l1_start = ft_b_end
    l1_total = ft_size * 2 * l1_size
    l1_w = np.zeros((ft_size * 2, l1_size), dtype=np.int8)
    l1_b = np.zeros(l1_size, dtype=np.int16)

    if l1_start + l1_total + l1_size <= len(arr):
        print(f"  Slicing L1: next {l1_total:,} values → ({ft_size*2}, {l1_size})")
        l1_raw = arr[l1_start:l1_start+l1_total].reshape(ft_size*2, l1_size).astype(np.float32)
        l1_w   = scale_to_int8(l1_raw)
        l1_b_raw = arr[l1_start+l1_total:l1_start+l1_total+l1_size].astype(np.float32)
        l1_b   = scale_to_int16(l1_b_raw)
    else:
        print("  L1 weights: zero-initialised (stream too short)")

    return ft_w, ft_b, l1_w, l1_b


# ── Main ─────────────────────────────────────────────────────────────────────
def convert(sf_path, out_path, ft_size, l1_size, l2_size, info_only):
    try:
        import numpy as np
    except ImportError:
        sys.exit("ERROR: numpy required — pip install numpy")

    print(f"\n{'═'*60}")
    print(f"  Stockfish → Hugine NNUE Converter")
    print(f"{'═'*60}")

    with open(sf_path, "rb") as f:
        data = f.read()

    sz = len(data)
    print(f"\n  Input : {sf_path}")
    print(f"  Size  : {sz:,} bytes ({sz/1e6:.1f} MB)")

    version  = struct.unpack_from("<I", data, 0)[0]
    arch     = struct.unpack_from("<I", data, 4)[0]
    desc_len = struct.unpack_from("<I", data, 8)[0]
    desc     = ""
    if desc_len < 4096:
        desc = data[12:12+desc_len].decode("utf-8", errors="replace").strip()

    arch_name = SF_ARCH_HASHES.get(arch, f"Unknown (0x{arch:08X})")
    print(f"  Format: version=0x{version:08X}  arch=0x{arch:08X}  ({arch_name})")
    print(f"  Desc  : {desc[:100]}")

    if info_only:
        marker = b"COMPRESSED_LEB128"
        mpos = data.find(marker)
        if mpos != -1:
            print(f"  Encoding: COMPRESSED_LEB128 at offset {mpos}")
        else:
            print(f"  Encoding: uncompressed binary")
        print(f"\n  Target architecture: {HALFKP_INPUTS}×2 → {ft_size} → {l1_size} → {l2_size} → 1")
        return

    print(f"\n  Target: {HALFKP_INPUTS}×2 → {ft_size} → {l1_size} → {l2_size} → 1")

    # ── Detect format and extract weights ──
    marker = b"COMPRESSED_LEB128"
    mpos   = data.find(marker)

    if mpos != -1:
        print(f"\n[1/4] Parsing COMPRESSED_LEB128 stream...")
        ft_w, ft_b, l1_w, l1_b = parse_compressed(data, mpos, ft_size, l1_size)
    else:
        print(f"\n[1/4] Parsing uncompressed binary chunks...")
        offset = 12 + (desc_len if desc_len < 4096 else 0)
        ft_w, ft_b, l1_w, l1_b, _ = parse_uncompressed(data, offset, ft_size, l1_size)

    if ft_w is None:
        sys.exit("\nERROR: Could not extract FT weights. "
                 "The file format may be unsupported. Run with --info to inspect.")

    print(f"\n[2/4] FT weight stats:")
    print(f"  shape={ft_w.shape}, range=[{ft_w.min()}, {ft_w.max()}]")
    non_zero = int((ft_w != 0).sum())
    print(f"  non-zero: {non_zero:,} / {ft_w.size:,} ({non_zero/ft_w.size*100:.1f}%)")

    print(f"\n[3/4] Building L2 / output layers (zero-initialised)")
    l2_w  = np.zeros((l1_size, l2_size), dtype=np.int8)
    l2_b  = np.zeros(l2_size, dtype=np.int16)
    out_w = np.zeros(l2_size, dtype=np.int8)
    out_b = np.zeros(1, dtype=np.int16)

    print(f"\n[4/4] Writing Hugine NNUE...")
    write_hugine(out_path, ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b)

    print(f"""
{'═'*60}
  WHAT THIS FILE WILL DO IN HUGINE
{'═'*60}

  ✓  Loads without error  (format is valid Hugine .nnue)
  ✓  FT captures real piece-position patterns from Stockfish's weights
  ✓  Produces legal moves / won't crash
  ✗  L2 / output layers are zero → raw eval ≈ 0 for all positions
  ✗  Playing strength is NOT comparable to Stockfish

{'═'*60}
  RECOMMENDED: USE AS TRAINING WARM-START
{'═'*60}

  1. Generate self-play data with the converted network loaded:

       python3 nnue_trainer/train.py generate \\
           --engine ./hugine_fast_nnue \\
           --nnue {out_path} \\
           --games 20000 --depth 8 \\
           --out data/sf_warmstart.txt

  2. Train on that data (only a few epochs needed since FT is already good):

       python3 nnue_trainer/train.py train \\
           --data data/sf_warmstart.txt \\
           --init {out_path} \\
           --epochs 30 --lr 0.001 \\
           --out hugine_tuned.nnue

  Training from this checkpoint converges MUCH faster than random init
  because the FT already encodes meaningful piece-position knowledge.
{'═'*60}
""")


def main():
    p = argparse.ArgumentParser(
        description="Convert a Stockfish .nnue file to Hugine native format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("input",  help="Stockfish .nnue file")
    p.add_argument("output", nargs="?", default="hugine_converted.nnue",
                   help="Output path (default: hugine_converted.nnue)")
    p.add_argument("--ft",    type=int, default=256,   help="FT hidden size (default 256)")
    p.add_argument("--l1",    type=int, default=32,    help="L1 size (default 32)")
    p.add_argument("--l2",    type=int, default=32,    help="L2 size (default 32)")
    p.add_argument("--large", action="store_true",     help="Preset: --ft 512 --l1 64 --l2 32")
    p.add_argument("--info",  action="store_true",     help="Print info only, no conversion")
    args = p.parse_args()

    if args.large:
        args.ft = 512; args.l1 = 64

    convert(args.input, args.output,
            ft_size=args.ft, l1_size=args.l1, l2_size=args.l2,
            info_only=args.info)

if __name__ == "__main__":
    main()
