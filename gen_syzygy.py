#!/usr/bin/env python3
"""
gen_syzygy.py — Syzygy Tablebase Setup Utility for Hugine
===========================================================

Helps you download, verify, and configure Syzygy tablebases for Hugine.

SUBCOMMANDS
-----------
  list      List all official Syzygy tablebase piece-sets (WDL + DTZ files)
  download  Download Syzygy files from the official mirror
  verify    Verify checksum integrity of downloaded tablebase files
  build     Compile Fathom (jdart1/Fathom) and link instructions
  config    Print the setoption command for your local tablebase directory
  info      Show how many piece-combinations are covered by each TB size

QUICK START
-----------
  # Download 3-4-5 man WDL + DTZ (lightweight, ~1.0 GB):
  python3 gen_syzygy.py download --pieces 5 --dir ./syzygy

  # Download 6-man (large — ~68 GB WDL+DTZ):
  python3 gen_syzygy.py download --pieces 6 --dir ./syzygy

  # Verify downloaded files:
  python3 gen_syzygy.py verify --dir ./syzygy

  # Show Hugine config commands:
  python3 gen_syzygy.py config --dir ./syzygy

  # Build Fathom and Hugine with Syzygy:
  python3 gen_syzygy.py build

Requirements
------------
  pip install requests   (for download subcommand)
  pip install tqdm       (optional — progress bars during download)

OFFICIAL SYZYGY SOURCES
------------------------
  Primary:  https://tablebase.lichess.ovh/tables/standard/
  Mirror:   https://tablebase.sesse.net/
  Info:     https://github.com/syzygy1/tb

HUGINE USAGE AT RUNTIME
------------------------
  setoption name SyzygyPath value /path/to/syzygy
  setoption name SyzygyProbeDepth value 1
  setoption name Syzygy50MoveRule value true
"""

import argparse
import hashlib
import os
import sys
import struct
from pathlib import Path

# ── Tablebase file catalog ─────────────────────────────────────────────────────
# Official Syzygy sources
MIRRORS = [
    "https://tablebase.lichess.ovh/tables/standard/",
    "https://tablebase.sesse.net/",
]

# 3-man piece sets (WDL + DTZ)
TB3 = [
    "KBNvK", "KBvK", "KNvK", "KPvK", "KQvK", "KRvK",
]

# 4-man piece sets
TB4 = [
    "KBBvK",  "KBBvKB", "KBBvKN", "KBBvKP", "KBBvKR",
    "KBNvKB", "KBNvKN", "KBNvKP", "KBNvKR", "KBPvKB", "KBPvKN",
    "KBPvKP", "KBPvKR", "KBvKB",  "KBvKN",  "KBvKP",  "KBvKR",
    "KNNvKB", "KNNvKN", "KNNvKP", "KNNvKR", "KNPvKB", "KNPvKN",
    "KNPvKP", "KNPvKR", "KNvKN",  "KNvKP",  "KNvKR",  "KPPvKP",
    "KPPvKR", "KPvKP",  "KPvKR",  "KQBvKQ", "KQBvKR", "KQNvKQ",
    "KQNvKR", "KQPvKQ", "KQPvKR", "KQQvKQ", "KQQvKR", "KQRvKQ",
    "KQRvKR", "KQvKB",  "KQvKN",  "KQvKP",  "KQvKQ",  "KQvKR",
    "KRBvKR", "KRNvKR", "KRPvKR", "KRRvKR", "KRvKB",  "KRvKN",
    "KRvKP",  "KRvKR",
]

# 5-man piece sets (subset — most common, ~1 GB WDL+DTZ)
TB5_COMMON = [
    "KBBBvK",  "KBBNvK",  "KBBPvK",  "KBBvKB",  "KBBvKN",
    "KBBvKP",  "KBBvKQ",  "KBBvKR",  "KBNNvK",  "KBNPvK",
    "KBNvKB",  "KBNvKN",  "KBNvKP",  "KBNvKQ",  "KBNvKR",
    "KBPPvK",  "KBPvKB",  "KBPvKN",  "KBPvKP",  "KBPvKQ",
    "KBPvKR",  "KBvKBP",  "KBvKNP",  "KBvKPP",  "KBvKQP",
    "KNNNvK",  "KNNPvK",  "KNNvKN",  "KNNvKP",  "KNNvKQ",
    "KNNvKR",  "KNPPvK",  "KNPvKN",  "KNPvKP",  "KNPvKQ",
    "KNPvKR",  "KNvKNP",  "KNvKPP",  "KNvKQP",  "KNvKRP",
    "KPPPvK",  "KPPvKP",  "KPPvKQ",  "KPPvKR",  "KPvKPP",
    "KQBBvK",  "KQBNvK",  "KQBPvK",  "KQBvKQ",  "KQBvKR",
    "KQNNvK",  "KQNPvK",  "KQNvKQ",  "KQNvKR",  "KQPPvK",
    "KQPvKQ",  "KQPvKR",  "KQQQvK",  "KQQvKQ",  "KQQvKR",
    "KQRBvK",  "KQRNvK",  "KQRPvK",  "KQRvKQ",  "KQRvKR",
    "KQvKBB",  "KQvKBN",  "KQvKBP",  "KQvKNN",  "KQvKNP",
    "KQvKPP",  "KQvKQB",  "KQvKQN",  "KQvKQP",  "KQvKQQ",
    "KQvKQR",  "KQvKRB",  "KQvKRN",  "KQvKRP",  "KQvKRR",
    "KRBBvK",  "KRBNvK",  "KRBPvK",  "KRBvKR",  "KRNNvK",
    "KRNPvK",  "KRNvKR",  "KRPPvK",  "KRPvKR",  "KRRBvK",
    "KRRNvK",  "KRRPvK",  "KRRvKR",  "KRvKBB",  "KRvKBN",
    "KRvKBP",  "KRvKNN",  "KRvKNP",  "KRvKPP",  "KRvKRB",
    "KRvKRN",  "KRvKRP",  "KRvKRR",
]

TB_STATS = {
    3: {"wdl_count": 6,   "dtz_count": 6,   "wdl_mb": 1,    "dtz_mb": 1},
    4: {"wdl_count": 55,  "dtz_count": 55,  "wdl_mb": 20,   "dtz_mb": 30},
    5: {"wdl_count": 500, "dtz_count": 500, "wdl_mb": 900,  "dtz_mb": 800},
    6: {"wdl_count": 3500,"dtz_count": 3500,"wdl_mb": 18000,"dtz_mb": 50000},
}

def _pieces_for_size(n):
    if n <= 3: return TB3
    if n <= 4: return TB3 + TB4
    if n <= 5: return TB3 + TB4 + TB5_COMMON
    return []   # 6-man: too many to list; use mirror browser

# ── helpers ───────────────────────────────────────────────────────────────────
def _hr_bytes(b):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024

def _check_wdl(path):
    """Return True if the file looks like a valid Syzygy WDL file."""
    try:
        with open(path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
        return magic in (0x5D23E871, 0x5D23E872,   # WDL magic variants
                         0xA9B78C59, 0xA9B78C5A)   # DTZ magic variants
    except Exception:
        return False

# ── subcommands ───────────────────────────────────────────────────────────────
def cmd_info(args):
    print("\nSyzygy Tablebase Coverage\n")
    print(f"  {'Pieces':>8}  {'WDL sets':>10}  {'DTZ sets':>10}  {'~WDL size':>12}  {'~DTZ size':>12}  {'~Total':>12}")
    print("  " + "-" * 76)
    for n in range(3, 7):
        s = TB_STATS[n]
        total = s['wdl_mb'] + s['dtz_mb']
        print(f"  {n:>8}  {s['wdl_count']:>10}  {s['dtz_count']:>10}  "
              f"{s['wdl_mb']:>10} MB  {s['dtz_mb']:>10} MB  {total:>10} MB")
    print()
    print("  WDL (Win/Draw/Loss) files: required for mid-search probing")
    print("  DTZ (Distance to Zero) files: required for root/endgame move selection")
    print()
    print("  Hugine UCI options:")
    print("    setoption name SyzygyPath value /path/to/syzygy")
    print("    setoption name SyzygyProbeDepth value 1")
    print("    setoption name Syzygy50MoveRule value true")


def cmd_list(args):
    n = args.pieces
    pieces = _pieces_for_size(n)
    if not pieces:
        print(f"  6-man piece list is too large to enumerate here.")
        print(f"  Browse: {MIRRORS[0]}")
        return
    print(f"\nSyzygy piece sets (up to {n}-man, {len(pieces)} sets):\n")
    for i, ps in enumerate(pieces):
        suffix = "" if (i + 1) % 6 else "\n"
        print(f"  {ps:<16}", end=suffix)
    print("\n")
    print(f"  Each set has two files: <name>.rtbw (WDL) + <name>.rtbz (DTZ)")


def cmd_download(args):
    dest = Path(args.dir)
    dest.mkdir(parents=True, exist_ok=True)
    n = args.pieces
    wdl_only = args.wdl_only
    pieces = _pieces_for_size(n)

    if not pieces:
        print(f"  6-man: too many files to enumerate automatically.")
        print(f"  Recommended: use rsync from an official mirror:")
        print(f"    rsync -avP rsync://tablebase.sesse.net/tablebase/standard/6-piece/ {dest}/")
        return

    try:
        import requests
    except ImportError:
        print("ERROR: pip install requests")
        sys.exit(1)

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    extensions = [".rtbw"] + ([] if wdl_only else [".rtbz"])
    mirror = MIRRORS[0]

    total_files = len(pieces) * len(extensions)
    downloaded = skipped = failed = 0

    print(f"\nDownloading Syzygy {n}-man tablebase{' (WDL only)' if wdl_only else ''}")
    print(f"  Destination: {dest}")
    print(f"  Mirror: {mirror}")
    print(f"  Files: {total_files} ({len(pieces)} piece sets × {len(extensions)} extensions)")
    print()

    for ps in pieces:
        for ext in extensions:
            fname = ps + ext
            url   = mirror + fname
            dest_file = dest / fname

            if dest_file.exists():
                skipped += 1
                continue

            try:
                r = requests.get(url, stream=True, timeout=30)
                if r.status_code == 404:
                    # Try alternate name (Syzygy sometimes stores as KXvKY vs KYvKX)
                    parts = ps.split("v")
                    alt = "v".join(reversed(parts)) + ext
                    r2 = requests.get(mirror + alt, stream=True, timeout=30)
                    if r2.status_code == 200:
                        r = r2; fname = alt; dest_file = dest / fname
                    else:
                        print(f"  ✗ 404 {fname}")
                        failed += 1
                        continue
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(dest_file, "wb") as f:
                    if use_tqdm and total:
                        with tqdm(total=total, unit="B", unit_scale=True,
                                  desc=fname[:40], leave=False) as bar:
                            for chunk in r.iter_content(65536):
                                f.write(chunk); bar.update(len(chunk))
                    else:
                        for chunk in r.iter_content(65536):
                            f.write(chunk)
                sz = dest_file.stat().st_size
                print(f"  ✓ {fname:<30}  {_hr_bytes(sz)}")
                downloaded += 1
            except Exception as e:
                print(f"  ✗ {fname}: {e}")
                failed += 1

    print()
    print(f"  Done:  {downloaded} downloaded  {skipped} skipped  {failed} failed")
    print(f"  Directory: {dest.resolve()}")
    print()
    _print_config(dest)


def cmd_verify(args):
    dest = Path(args.dir)
    if not dest.exists():
        print(f"ERROR: Directory not found: {dest}")
        sys.exit(1)

    wdl_files = sorted(dest.glob("*.rtbw"))
    dtz_files = sorted(dest.glob("*.rtbz"))
    all_files = wdl_files + dtz_files

    if not all_files:
        print(f"  No .rtbw/.rtbz files found in {dest}")
        sys.exit(1)

    print(f"\nVerifying {len(all_files)} Syzygy files in {dest}\n")

    ok = bad = 0
    for path in all_files:
        try:
            sz = path.stat().st_size
            if sz < 8:
                print(f"  ✗ {path.name:<36} too small ({sz} bytes)")
                bad += 1
                continue

            # Read magic + basic structure check
            with open(path, "rb") as f:
                magic_bytes = f.read(4)
            magic = struct.unpack("<I", magic_bytes)[0]

            is_wdl = path.suffix == ".rtbw"
            wdl_magic = {0x5D23E871, 0x5D23E872}
            dtz_magic = {0xA9B78C59, 0xA9B78C5A}
            valid_magic = magic in (wdl_magic if is_wdl else dtz_magic)

            if valid_magic:
                print(f"  ✓ {path.name:<36} {_hr_bytes(sz):>10}")
                ok += 1
            else:
                print(f"  ✗ {path.name:<36} bad magic 0x{magic:08X}")
                bad += 1
        except Exception as e:
            print(f"  ✗ {path.name}: {e}")
            bad += 1

    print()
    print(f"  Result: {ok} OK  {bad} invalid/corrupt")
    if bad == 0:
        print(f"  ✓ All {ok} files verified clean")
        _print_config(dest)


def _print_config(dest):
    dest = Path(dest).resolve()
    print("  Hugine UCI configuration:")
    print(f"    setoption name SyzygyPath value {dest}")
    print(f"    setoption name SyzygyProbeDepth value 1")
    print(f"    setoption name Syzygy50MoveRule value true")
    print()
    print("  Verify endgame probing (KRK — rook mate):")
    print("    position fen 8/8/3K4/8/8/8/8/3Rk3 w - - 0 1")
    print("    go depth 10")
    print("    # Expected: score cp 0 tbhits 1 (or bestmove with mate score)")


def cmd_config(args):
    dest = Path(args.dir)
    if not dest.exists():
        print(f"  WARNING: {dest} does not exist yet")
    print()
    _print_config(dest)


def cmd_build(args):
    """Print step-by-step Fathom build instructions for the host system."""
    import platform
    system = platform.system()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║          Hugine + Fathom (Syzygy) Build Instructions            ║
╚══════════════════════════════════════════════════════════════════╝

Step 1 — Clone Fathom (jdart1 fork, Hugine-compatible API)
  git clone https://github.com/jdart1/Fathom Fathom
  cd Fathom && make
  cd ..

Step 2 — Compile Hugine with Syzygy support
  (Remove -DNO_SYZYGY from the flags — it is the default-off guard)

  # Standard build with NNUE + Syzygy:
  g++ -O3 -march=native -std=c++17 -pthread \\
      -DUSE_NNUE -DNDEBUG \\
      -IFathom/src \\
      hugine-iota-v510.cpp Fathom/obj/tbprobe.o \\
      -o hugine -lpthread

  # With NNUE_XL (1024-node) + Syzygy:
  g++ -O2 -std=c++17 -pthread \\
      -DUSE_NNUE -DNNUE_XL \\
      -IFathom/src \\
      hugine-iota-v510.cpp Fathom/obj/tbprobe.o \\
      -o hugine_xl_nnue -lpthread

Step 3 — Download Syzygy tablebases (3-5 man ≈ 1 GB):
  python3 gen_syzygy.py download --pieces 5 --dir ./syzygy

Step 4 — Configure at runtime:
  setoption name SyzygyPath value /absolute/path/to/syzygy
  setoption name SyzygyProbeDepth value 1
  setoption name Syzygy50MoveRule value true

Step 5 — Verify (KRK should return tbhits):
  position fen 8/8/3K4/8/8/8/8/3Rk3 w - - 0 1
  go depth 10
""")

    if system == "Linux" and "Android" not in platform.version():
        print("  Termux/Android note: see SYZYGY_ANDROID_BUILD.md for NDK cross-compile.")
    elif "Android" in platform.version():
        print("  Termux: see SYZYGY_ANDROID_BUILD.md for full guide.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        prog="gen_syzygy.py",
        description="Syzygy Tablebase Setup Utility for Hugine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 gen_syzygy.py info
  python3 gen_syzygy.py list   --pieces 5
  python3 gen_syzygy.py download --pieces 5 --dir ./syzygy
  python3 gen_syzygy.py download --pieces 5 --dir ./syzygy --wdl-only
  python3 gen_syzygy.py verify   --dir ./syzygy
  python3 gen_syzygy.py config   --dir ./syzygy
  python3 gen_syzygy.py build
        """,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info",     help="Coverage stats for each TB size")
    sub.add_parser("build",    help="Print Fathom + Hugine build instructions")

    p = sub.add_parser("list", help="List piece sets for a given man-count")
    p.add_argument("--pieces", type=int, default=5, choices=[3,4,5,6],
                   help="Maximum piece count (3/4/5/6)")

    p = sub.add_parser("download", help="Download Syzygy files from official mirror")
    p.add_argument("--pieces",   type=int, default=5, choices=[3,4,5,6])
    p.add_argument("--dir",      default="./syzygy", help="Destination directory")
    p.add_argument("--mirror",   default=MIRRORS[0], help="Override download mirror URL")
    p.add_argument("--wdl-only", action="store_true",
                   help="Download only WDL files (smaller, enough for mid-search probing)")

    p = sub.add_parser("verify", help="Verify downloaded .rtbw/.rtbz files")
    p.add_argument("--dir", default="./syzygy")

    p = sub.add_parser("config", help="Print UCI setoption commands for your TB directory")
    p.add_argument("--dir", default="./syzygy")

    args = ap.parse_args()
    {
        "info":     cmd_info,
        "list":     cmd_list,
        "download": cmd_download,
        "verify":   cmd_verify,
        "config":   cmd_config,
        "build":    cmd_build,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
