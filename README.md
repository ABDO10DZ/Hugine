# Hugine Chess Engine

Hugine is a UCI‑compliant chess engine written in C++17. It uses magic bitboards for move generation and includes common search techniques like PVS, LMR, null‑move pruning, futility pruning, and SEE. The engine can optionally use Syzygy tablebases and NNUE evaluation (both compile‑time options). The code is commented and meant to be readable, making it a useful reference for chess engine development.

## Features

- Magic bitboard move generation (rook, bishop, queen)
- Transposition table with aging and replacement
- Principal Variation Search with aspiration windows
- Late Move Reduction (LMR)
- Null‑move pruning (adaptive)
- Futility pruning, razoring, probcut
- Static Exchange Evaluation (SEE)
- Piece‑square tables with tapered evaluation (mg/eg)
- Mobility, pawn structure, king safety, passed pawns evaluation
- Optional NNUE evaluation (compile with `-DUSE_NNUE`)
- Optional Syzygy tablebase support (WDL and root moves) – x86 only
- Opening book (Polyglot format)
- Multi‑threading support (up to 64 threads)
- Time management with configurable overhead
- Debug output (optional `-DDEBUG`)

## Build Instructions

### Prerequisites
- C++17 compiler (GCC or Clang)
- pthread library (on Unix‑like systems)
- For Syzygy: [Fathom](https://github.com/jdart1/Fathom) in `fathom/src/tbprobe.h` (x86 only)
- For NNUE: a network file (custom format, not included)

### Build Options

| Option          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `-DDEBUG`       | Enable verbose debug output (stderr).                                      |
| `-DUSE_NNUE`    | Enable NNUE evaluation (requires network file).                            |
| `-DUSE_SYZYGY`  | Force Syzygy support even if auto‑detection fails.                         |
| `-DNO_SYZYGY`   | Explicitly disable Syzygy (required on ARM/Android).                       |

### Build Commands by Platform

| Platform                | Command                                                                                           |
|-------------------------|---------------------------------------------------------------------------------------------------|
| **Linux x86** (with Syzygy) | `g++ -O2 -std=c++17 -pthread -DUSE_SYZYGY hugine.cpp -o hugine`                             |
| **Linux x86** (no Syzygy)    | `g++ -O2 -std=c++17 -pthread hugine.cpp -o hugine`                                          |
| **Windows x86** (MinGW)      | `g++ -O2 -std=c++17 -pthread hugine.cpp -o hugine.exe`                                     |
| **Termux Android** (ARM)     | `g++ -O2 -std=c++17 -pthread -DNO_SYZYGY hugine.cpp -o hugine`                             |
| **ARM Linux** (no Syzygy)    | `g++ -O2 -std=c++17 -pthread -DNO_SYZYGY hugine.cpp -o hugine`                             |
| **Any with debug**           | Add `-DDEBUG` to any of the above commands.                                                      |

**Notes**:
- On x86, Syzygy is auto‑enabled if Fathom headers are found. Use `-DNO_SYZYGY` to disable.
- On ARM/Android you must add `-DNO_SYZYGY` (Fathom uses x86 intrinsics).
- NNUE must be explicitly enabled with `-DUSE_NNUE`.

## Quick Usage Example

After building, run the engine and enter UCI commands directly, or pipe them:

```bash
$ ./hugine
uci
id name Hugine
id author Hugine
...
uciok
isready
readyok
position startpos
go depth 5
info depth 1 ... pv e2e4
...
bestmove e2e4
quit
```

Or from a script:

```bash
echo -e "uci\nisready\nposition startpos\ngo depth 5\nquit" | ./hugine
```

To use with a GUI (like Arena or Cute Chess), point the GUI to the executable.

## Estimated Strength

The ratings below are rough estimates based on engine architecture. Actual performance may vary.

| Time Control        | Estimated CCRL ELO |
|---------------------|--------------------|
| Bullet (1+0)        | 2600 ± 100         |
| Blitz (3+0)         | 2700 ± 100         |
| Rapid (15+10)       | 2800 ± 100         |
| Classical (40/120)  | 2850 ± 100         |
| Undefined           | 2750 (average)     |

### Comparison with Top Engines (Blitz)

| Engine          | Approx. CCRL Blitz ELO |
|-----------------|------------------------|
| Stockfish 16    | 3600+                  |
| Komodo 14       | 3500+                  |
| Houdini 6       | 3400+                  |
| **Hugine**      | **~2700**              |

## TODO / Missing Features

- Pondering (search during opponent’s time)
- More sophisticated time management (smooth allocation, move importance)
- Syzygy DTZ probing
- Multi‑PV output improvements (sorting, PV display)
- Razoring and futility margin tuning
- NNUE incremental updates (currently full refresh on every eval)
- Better move ordering (MVV‑LVA for captures, improved history/killer)
- UCI options `UCI_LimitStrength` and `UCI_Elo`
- Full Chess960 support
- Contempt factor for drawish positions

## Known Bugs / Issues

- Syzygy only works on x86 (Fathom uses x86 intrinsics).
- NNUE loader expects a specific binary format; no default network provided.
- Repetition detection may miss threefold after null moves.
- SEE may be inaccurate for en‑passant or promotions.
- Multi‑threading is simple Lazy SMP; no work sharing between threads.
- Stop signal is checked every 1024 nodes, so response may be slightly delayed.
- Null‑move pruning can be unsafe in low‑material positions.
- Some UCI options (e.g., `Clear Hash`) are implemented but not thoroughly tested.
- Compilation warnings may appear with strict flags (none critical).

## When to Use Hugine

- **Learning**: The code is well‑commented and demonstrates a complete modern engine.
- **Experimentation**: Easy to modify and add your own features.
- **Lightweight play**: Runs on modest hardware, including Android.
- **As a base**: Can be extended for research or as a starting point for a stronger engine.

## When Not to Use Hugine

- **High‑level competition**: Strength is far below top engines.
- **ARM + Syzygy**: Tablebases not available.
- **If you need pondering**: Not yet implemented.
- **For critical tournament play**: Some UCI options may be incomplete; use with caution.

---

**License**: The source code does not include a license. Contact the author for permissions. Provided as‑is for educational and personal use. - never use for commercial
