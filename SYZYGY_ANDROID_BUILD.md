# Syzygy / Fathom on Android (Termux) — Complete Build Guide
## Hugine 3.0 "Gama"

Fathom is the standard Syzygy probe library and **does work on ARM64/Android**.
The engine previously blocked it with `ARCH_X86 && !OS_ANDROID` — that guard is
now removed in v4. Follow the steps below and it will compile cleanly.

---

## Step 1 — Install Termux build tools

```bash
pkg update && pkg upgrade -y
pkg install -y clang git
```

---

## Step 2 — Clone fathom

The official maintained fork is from the Stockfish team:

```bash
git clone https://github.com/official-stockfish/Fathom fathom
```

Your directory layout should look like this:

```
hugine-gama-v4.cpp
fathom/
  src/
    tbprobe.h
    tbprobe.c
    tbconfig.h
```

---

## Step 3 — Compile fathom as C (critical — do NOT compile as C++)

```bash
clang -O2 -std=c11 -c fathom/src/tbprobe.c \
      -o fathom/src/tbprobe.o \
      -DNDEBUG
```

That's it. One command, no extra flags needed on modern ARM64.

---

## Step 4 — Compile Hugine with Syzygy enabled

```bash
clang++ -O3 -std=c++17 -pthread \
        hugine-gama-v4.cpp \
        fathom/src/tbprobe.o \
        -o hugine
```

> **Note:** Do NOT pass `-march=native` on Termux — it sometimes generates
> NEON/SVE instructions that crash on older Android kernels. Leave it out and
> clang will auto-detect a safe baseline for your device.

---

## Step 5 — Download tablebase files

You need `.rtbw` and `.rtbz` files for the pieces you want probed.
3-4-5 piece tablebases fit in ~1 GB:

```bash
# Create a TB directory
mkdir -p ~/tablebase

# Example: download KPK (King+Pawn vs King) — tiny, good test
# Full TB downloads from: https://syzygy-tables.info
# Recommended mirror for Termux (no browser needed):
cd ~/tablebase
wget https://tablebase.sesse.net/syzygy/3-4-5/KPK.rtbw
wget https://tablebase.sesse.net/syzygy/3-4-5/KPK.rtbz
```

---

## Step 6 — Set the path in UCI

In your GUI or directly via UCI:

```
setoption name SyzygyPath value /data/data/com.termux/files/home/tablebase
```

Or the shorter Termux path:

```
setoption name SyzygyPath value ~/tablebase
```

---

## Common Build Errors and Fixes

### Error: `unknown type name 'bool'`
**Cause:** Compiling tbprobe.c as C89 or without a C standard.  
**Fix:** Add `-std=c11` to the clang command (already in Step 3).

---

### Error: `implicit declaration of function 'mmap'`
**Cause:** Old Termux clang without `_DEFAULT_SOURCE` defined.  
**Fix:**
```bash
clang -O2 -std=c11 -D_DEFAULT_SOURCE -c fathom/src/tbprobe.c -o fathom/src/tbprobe.o
```

---

### Error: `conflicting types for 'popcount'`
**Cause:** Some fathom versions define their own `popcount` that clashes with `<strings.h>`.  
**Fix:** Add `-DUSE_POPCNT` to tell fathom to use the built-in:
```bash
clang -O2 -std=c11 -DUSE_POPCNT -c fathom/src/tbprobe.c -o fathom/src/tbprobe.o
```

---

### Error: `undefined reference to tb_init` at link time
**Cause:** Forgot to link `tbprobe.o`.  
**Fix:** Make sure `fathom/src/tbprobe.o` is on the clang++ command line (Step 4).

---

### Error: `undefined reference to __stack_chk_fail`
**Cause:** Termux NDK/clang version mismatch.  
**Fix:**
```bash
clang++ -O3 -std=c++17 -pthread \
        -fno-stack-protector \
        hugine-gama-v4.cpp fathom/src/tbprobe.o -o hugine
```

---

### Engine compiles but ignores TBs (SyzygyPath set but no TB hits)

Check these in order:
1. Run `setoption name SyzygyPath value /full/absolute/path` (no `~` expansion in UCI).
2. The `.rtbw`/`.rtbz` files must be readable: `ls -la ~/tablebase/*.rtbw`
3. Verify the engine prints the correct piece count:
   ```
   setoption name SyzygyPath value /path/to/tablebase
   position fen 8/8/8/8/8/8/4K3/4k1P1 w - - 0 1
   eval
   ```
   If you see `tbhits 1` in any subsequent `go` output, TBs are working.

---

## Without Syzygy (lightweight build — always works)

If you just want a working engine without tablebases:

```bash
clang++ -O3 -std=c++17 -pthread \
        -DNO_SYZYGY \
        hugine-gama-v4.cpp -o hugine
```

---

## Summary of all three scenarios

| Scenario | Command |
|---|---|
| No Syzygy (safe, always works) | `clang++ -O3 -std=c++17 -pthread -DNO_SYZYGY hugine-gama-v4.cpp -o hugine` |
| With Syzygy (normal build) | compile tbprobe.c first, then link (Steps 3+4) |
| Force Syzygy even if header check fails | add `-DUSE_SYZYGY` + link tbprobe.o |
