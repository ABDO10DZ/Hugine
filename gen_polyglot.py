#!/usr/bin/env python3
"""
Generates polyglot_random.h from the canonical polyglot source.

This script extracts the 781 Zobrist random values from:
  https://github.com/ulthiel/polyglot/blob/master/random.h

Usage:
  # Option A - auto download (requires internet):
  python3 gen_polyglot.py --download

  # Option B - from a local copy of random.h:
  python3 gen_polyglot.py random.h

  # Option C - verify an existing polyglot_random.h:
  python3 gen_polyglot.py --verify

The generated polyglot_random.h should be placed next to hugine.cpp.
Then recompile: g++ ... hugine.cpp  (it auto-detects the include).
"""
import sys, re, struct, hashlib
from pathlib import Path

# ── Known anchor values for verification ─────────────────────────────────────
TARGET_HASH_STARTPOS = 0x463b96181691fc9c  # well-known standard Polyglot hash
ENTRY_0_EXPECTED     = 0x9D39247E33776D41  # Random64[0] from polyglot source

STARTPOS_INDICES = (
    *[1*64+sq for sq in range(8,16)],    # wP a2-h2
    *[0*64+sq for sq in range(48,56)],   # bP a7-h7
    3*64+1, 3*64+6,                       # wN b1,g1
    2*64+57, 2*64+62,                     # bN b8,g8
    5*64+2, 5*64+5,                       # wB c1,f1
    4*64+58, 4*64+61,                     # bB c8,f8
    7*64+0, 7*64+7,                       # wR a1,h1
    6*64+56, 6*64+63,                     # bR a8,h8
    9*64+3,                               # wQ d1
    8*64+59,                              # bQ d8
    11*64+4,                              # wK e1
    10*64+60,                             # bK e8
    768, 769, 770, 771,                   # castling (all 4 rights)
)

def parse_table(text):
    """Extract all 64-bit hex values from random.h content."""
    vals = [int(x, 16) for x in re.findall(r'0x([0-9A-Fa-f]{16,16})', text)]
    if len(vals) < 781:
        # Try without leading zeros
        vals = [int(x, 0) for x in re.findall(r'0x[0-9A-Fa-f]+', text)]
        vals = [v for v in vals if v <= 0xFFFFFFFFFFFFFFFF]
    return vals[:781]

def verify_table(table):
    if len(table) < 781:
        return False, f"Only {len(table)} entries (need 781)"
    if table[0] != ENTRY_0_EXPECTED:
        return False, f"Entry[0]={table[0]:#018x} expected={ENTRY_0_EXPECTED:#018x}"
    h = 0
    for idx in STARTPOS_INDICES:
        h ^= table[idx]
    if h != TARGET_HASH_STARTPOS:
        return False, f"Startpos hash={h:#018x} expected={TARGET_HASH_STARTPOS:#018x}"
    return True, "OK"

def write_header(table, path="polyglot_random.h"):
    lines = [
        "// polyglot_random.h — 781 standard Polyglot Zobrist random values",
        "// Source: https://github.com/ulthiel/polyglot/blob/master/random.h",
        "// Public domain. Place next to hugine.cpp and recompile.",
        "// Verified: startpos hash = 0x463b96181691fc9c",
        "#pragma once",
        "#include <cstdint>",
        "namespace PolyglotRandomH {",
        "static const uint64_t table[781] = {",
    ]
    for i in range(0, 781, 4):
        chunk = table[i:min(i+4, 781)]
        comment = f"  // [{i}] piece_sq"
        if i == 768: comment = "  // [768] castling"
        if i == 772: comment = "  // [772] ep_file"
        if i == 780: comment = "  // [780] stm"
        lines.append("    " + ", ".join(f"0x{v:016X}ULL" for v in chunk) + "," + comment)
    lines += ["};", "}  // namespace PolyglotRandomH"]
    Path(path).write_text('\n'.join(lines) + '\n')
    print(f"✅ Written {path}  ({len(table)} entries)")
    ok, msg = verify_table(table)
    print(f"   Verification: {msg}")

def download_and_extract():
    import urllib.request
    url = "https://raw.githubusercontent.com/ulthiel/polyglot/master/random.h"
    print(f"Downloading {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            text = r.read().decode()
        table = parse_table(text)
        ok, msg = verify_table(table)
        if not ok:
            print(f"Verification failed: {msg}")
            sys.exit(1)
        write_header(table)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try: python3 gen_polyglot.py your_local_random.h")
        sys.exit(1)

def from_file(path):
    text = Path(path).read_text()
    table = parse_table(text)
    ok, msg = verify_table(table)
    if not ok:
        print(f"Verification failed: {msg}")
        sys.exit(1)
    write_header(table)

def verify_existing():
    p = Path("polyglot_random.h")
    if not p.exists():
        print("polyglot_random.h not found")
        sys.exit(1)
    text = p.read_text()
    table = parse_table(text)
    ok, msg = verify_table(table)
    print(f"polyglot_random.h: {msg}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify_existing()
    elif "--download" in sys.argv:
        download_and_extract()
    elif len(sys.argv) >= 2:
        from_file(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(0)
