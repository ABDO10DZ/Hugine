# Hugine 3.0 "Gama"

A UCI chess engine by **0xbytecode**.  
Supports Linux · Windows · macOS · Android/Termux · ARM64.

---

## Features

| Feature | Details |
|---|---|
| Search | Negamax + PVS, YBWC multi-threading |
| Pruning | NMP, LMR (log table), Futility, LMP, ProbCut, SEE bad-capture |
| Extensions | Check extension, recapture, passed-pawn |
| Move ordering | TT move, killer × 2, counter-move, SEE captures, history, continuation |
| Evaluation | Classical (PST + mobility + pawn structure + king safety) or NNUE |
| Hash table | Lockless TT with age-gating and mate-protection |
| Tablebases | Syzygy DTZ via fathom (x86 + ARM64 + Android) |
| UCI options | Threads, Hash, MultiPV, Contempt, Skill Level, UCI_Elo, Ponder, UCI_ShowWDL |
| Opening book | Polyglot .bin |
| Learning | Persistent reinforcement-learning table |
| Chess960 | Full FRC support |
| Platforms | Linux, Windows (MSVC/MinGW), macOS, Android/Termux |

---

## Quick Start

```bash
git clone https://github.com/ABDO10DZ/hugine
cd hugine
./build.sh          # auto-detects everything
./hugine            # opens UCI loop
```

---

## Building

### Automatic (recommended)

```bash
./build.sh
```

The script detects your OS, CPU architecture, best available compiler, whether
fathom (Syzygy) is present, and whether a `.nnue` / `nn-*.bin` network file
exists. It picks the best settings and compiles in one step.

### Build script options

| Flag | Effect |
|---|---|
| `--syzygy` | Force Syzygy on (fathom must be present) |
| `--no-syzygy` | Force Syzygy off |
| `--nnue` | Force NNUE on (network file must be present) |
| `--no-nnue` | Force NNUE off — use classical eval |
| `--debug` | Debug build: `-O0 -g`, assertions enabled |
| `--portable` | No `-march=native` — safe for cross-compilation |
| `--compiler=clang++` | Use a specific compiler |
| `--exe=myname` | Custom output binary name |
| `--help` | Print usage |

Examples:

```bash
./build.sh --syzygy --nnue           # force both on
./build.sh --no-syzygy --portable    # portable build, no tablebases
./build.sh --compiler=clang++ --exe=hugine-avx2
./build.sh --debug                   # debugging
```

### Makefile (alternative)

```bash
make                       # same auto-detection as build.sh
make SYZYGY=1 NNUE=1       # force both on
make SYZYGY=0              # no tablebases
make DEBUG=1               # debug build
make NATIVE=0              # no -march=native
make CXX=clang++           # choose compiler
make EXE=myengine          # custom name
make clean
make help
```

### Manual compile

```bash
# Minimal (no tablebases)
g++ -O3 -std=c++17 -pthread -DNO_SYZYGY hugine-gama-v5.cpp -o hugine

# With Syzygy (fathom must be cloned first)
cc  -O2 -std=c11  -c fathom/src/tbprobe.c -o fathom/src/tbprobe.o
g++ -O3 -std=c++17 -pthread hugine-gama-v5.cpp fathom/src/tbprobe.o -o hugine

# Windows MSVC
cl /O2 /std:c++17 /EHsc /DNO_SYZYGY hugine-gama-v5.cpp

# Android Termux (ARM64)
clang++ -O3 -std=c++17 -pthread -DNO_SYZYGY hugine-gama-v5.cpp -o hugine
```

### Syzygy tablebases (fathom)

```bash
git clone https://github.com/official-stockfish/Fathom fathom
./build.sh --syzygy
```

Then set the tablebase path in your GUI or via UCI:

```
setoption name SyzygyPath value /path/to/tablebases
```

---

## UCI Options

| Option | Type | Default | Description |
|---|---|---|---|
| `Hash` | spin | 256 | TT size in MB |
| `Threads` | spin | 1 | Search threads (YBWC) |
| `MultiPV` | spin | 1 | Number of PV lines to report |
| `Ponder` | check | false | Enable pondering |
| `Contempt` | spin | 0 | Draw contempt in centipawns |
| `Move Overhead` | spin | 10 | Network/GUI lag compensation (ms) |
| `Skill Level` | spin | 20 | 0 = weakest, 20 = full strength |
| `UCI_LimitStrength` | check | false | Enable Elo limit |
| `UCI_Elo` | spin | 1500 | Target Elo when UCI_LimitStrength is on |
| `UCI_ShowWDL` | check | false | Show win/draw/loss probabilities |
| `SyzygyPath` | string | — | Path to Syzygy .rtbw/.rtbz files |
| `SyzygyProbeDepth` | spin | 1 | Min depth for Syzygy probe |
| `BookFile` | string | — | Polyglot .bin opening book |
| `NNUEFile` | string | — | NNUE network file |
| `NNUEWeight` | spin | 100 | NNUE blend weight (0 = classical only) |
| `UCI_Chess960` | check | false | Fischer Random Chess mode |

---

## Custom UCI Commands

```
learn result win|draw|loss   — update learning table from last game PV
learn clear                  — wipe learning table
learn save                   — persist learning table to disk
```

---

## Architecture

```
hugine-gama-v5.cpp
├── Platform detection       (OS, arch, SIMD, Windows/MSVC portability)
├── Bitboards + magic tables (Fancy Magic for rook/bishop)
├── Position                 (make/undo, Zobrist hash, SEE, gives_check)
├── Move generator           (pseudo-legal + legality filter)
├── Evaluation               (Classical PST/mobility/pawn/king-safety or NNUE)
├── Transposition table      (lockless, age-gated, mate-protected)
├── Opening book             (Polyglot)
├── Syzygy probe             (fathom, WDL + DTZ)
├── Learning table           (persistent reinforcement)
├── Search
│   ├── negamax / PVS
│   ├── quiescence
│   ├── YBWC (Young Brothers Wait Concept) threads
│   └── Time management
└── UCI loop
```

---

## Performance

Measured on Linux x86_64, single thread, starting position:

| Version | Depth 14 nodes | NPS |
|---|---|---|
| v4 | 1 759 946 | ~650K |
| v5 | 238 709 | ~575K |

v5 uses **~7× fewer nodes** at the same depth thanks to log-scale LMR and
SEE bad-capture pruning, with equal or better move quality.

---

## License

MIT — see source file header.
