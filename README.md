<div align="center">

# ♟ Hugine

### A high-performance UCI chess engine written in C++17

[![Version](https://img.shields.io/badge/version-5.0_Iota-blue?style=flat-square)](https://github.com/ABDO10DZ/hugine/releases)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-orange?style=flat-square)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-Free%20Non--Commercial-green?style=flat-square)](#-license)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS%20%7C%20Android-lightgrey?style=flat-square)](#-building)
[![Arch](https://img.shields.io/badge/arch-x86__64%20%7C%20ARM64-purple?style=flat-square)](#-building)

**Hugine** is a from-scratch UCI-compliant chess engine featuring NNUE evaluation, opening book support, Syzygy tablebases, Lazy SMP multi-threading, and a full cross-platform build system — targeting Linux, Windows, macOS, and Android/Termux across both x86-64 and ARM64 architectures.

[**Features**](#-features) · [**Building**](#-building) · [**UCI Usage**](#-uci-usage) · [**vs Stockfish**](#️-vs-stockfish) · [**Roadmap**](#-roadmap) · [**License**](#-license)

---

</div>

## 📖 Description

Hugine is a standalone, single-source UCI chess engine built with a focus on correctness, performance, and portability. Version **5.0 Iota** is the most mature release to date, having passed an exhaustive **74-test verification suite** covering:

- UCI protocol compliance
- Perft correctness across all edge cases (en passant, castling, promotions, Chess960)
- Tactical puzzle solving
- NNUE evaluation and SIMD path verification
- Opening book probing and weighted move selection
- Time management under various clock modes
- Robustness against adversarial UCI input
- Multi-thread Lazy SMP stress testing

The engine compiles with a single `g++` invocation and has **no external dependencies** in its base configuration. NNUE and Syzygy support are optional compile-time flags, making it practical to build on anything from a desktop workstation to a phone running Termux.

```
Engine   : Hugine 5.0 "Iota"
Author   : 0xbytecode (ABDO10DZ)
Protocol : UCI
Language : C++17 · Single source file (~5,000 lines)
NNUE     : Optional · AVX2 / SSE4.1 / NEON auto-detected
Syzygy   : Optional · Fathom / jdart1 API
Threading: Lazy SMP
Perft    : ~11–14M nodes/sec  (O3, x86-64)
Search   : ~400–700k nps HCE  |  ~300–550k nps NNUE  (single thread)
```

---

## ✨ Features

| Category | Feature | Details |
|---|---|---|
| **Protocol** | UCI compliant | `uci`, `isready`, `position`, `go`, `stop`, `quit`, `ucinewgame` |
| **Protocol** | UCI options | Hash, Threads, Ponder, MultiPV, MoveOverhead, UCI_Chess960, Skill Level, UCI_ShowWDL |
| **Search** | Iterative deepening | Aspiration windows, soft/hard time limits |
| **Search** | Alpha-beta | Fail-soft negamax with PVS (Principal Variation Search) |
| **Search** | Move ordering | TT move · MVV-LVA · SEE bad-capture pruning · killers · history |
| **Search** | Pruning | Null-move · futility · late-move · ProbCut |
| **Search** | Reductions | LMR with Stockfish-style log tables |
| **Search** | Extensions | Check extensions · singular extensions |
| **Search** | Quiescence | Full quiescence search with delta pruning |
| **Search** | Multi-PV | Up to 500 PV lines |
| **Evaluation** | Hybrid eval | HCE (hand-crafted) blended with NNUE via game phase |
| **Evaluation** | NNUE | 40960→256→32→32→1 architecture |
| **Evaluation** | SIMD | AVX2 · SSE4.1 · ARM NEON — auto-detected at compile time |
| **Evaluation** | NNUE loading | Smart path: `EvalFile`, `./nnue/`, same-dir, `NNUEPath` |
| **Evaluation** | HCE terms | Material · PST · mobility · outposts · passed pawns · king safety |
| **Evaluation** | Draw detection | Repetition · 50-move rule · insufficient material |
| **Threading** | Lazy SMP | Independent iterative deepening per thread, shared TT |
| **Threading** | Safety | Per-thread `Position` copies · TLS NNUE accumulators · thread-0 TM |
| **Book** | Opening book | Polyglot-layout `.bin` format · weighted random move selection |
| **Book** | Options | `BookFile` path · `OwnBook` toggle |
| **Tablebases** | Syzygy | Fathom/jdart1 API · `SyzygyPath` · `SyzygyProbeDepth` · 50-move rule |
| **TT** | Lock-free | Atomic XOR store/probe · per-search age gating |
| **Learning** | Persistent | Hash-keyed position→score table, probed during evaluation |
| **Time mgmt** | Adaptive | Soft/hard limits · score-drop scaling · best-move stability bonus |
| **Portability** | Platforms | Linux · Windows (MSVC/MinGW) · macOS · Android/Termux |
| **Portability** | Architectures | x86-64 · ARM64 |
| **Rules** | Full chess | En passant · castling · promotions · Chess960 · 50-move · threefold |

---

## 🏗 Building

### Requirements

| Tool | Minimum version | Notes |
|---|---|---|
| GCC | 9+ | Recommended |
| Clang | 10+ | Fully supported |
| MSVC | 2019+ | Windows only |
| CMake | 3.16+ | Optional |
| NDK | r25+ | Android cross-compile only |

---

### 1 — Quick One-Liner

No dependencies. Just compile and run:

```bash
# Minimal build (no NNUE, no Syzygy)
g++ -O2 -std=c++17 -pthread hugine.cpp -o hugine

# With NNUE evaluation support
g++ -O2 -std=c++17 -pthread -DUSE_NNUE hugine.cpp -o hugine

# Full optimized release with AVX2 NNUE
g++ -O3 -march=native -std=c++17 -pthread -DUSE_NNUE -DUSE_AVX2 -DNDEBUG hugine.cpp -o hugine
```

---

### 2 — build.sh

The included script auto-detects your platform, architecture, and SIMD capabilities:

```bash
chmod +x build.sh
./build.sh
```

Common options:

```bash
./build.sh --nnue              # Enable NNUE evaluation
./build.sh --nnue --avx2       # NNUE + AVX2 SIMD
./build.sh --debug             # Debug build with ASAN + UBSAN
./build.sh --help              # Show all options
```

---

### 3 — GNU Make

```bash
make                  # Default optimized release
make NNUE=1           # NNUE-enabled release
make debug            # Debug build (ASAN + UBSAN)
make clean            # Remove build artifacts
```

Available targets: `release` · `debug` · `nnue` · `nnue-avx2` · `profile` · `clean` · `install`

---

### 4 — CMake

```bash
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Optional features
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_NNUE=ON -DUSE_AVX2=ON

# Build
cmake --build . --parallel

# Install (optional)
cmake --install .
```

---

## 🔬 Advanced Builds

### Build Variants

| Variant | Extra flags | Purpose |
|---|---|---|
| `base` | *(none)* | No optional features, fast compile |
| `nnue` | `-DUSE_NNUE` | NNUE eval, requires `.nnue` file |
| `nnue-avx2` | `-DUSE_NNUE -DUSE_AVX2 -mavx2` | NNUE + AVX2 SIMD |
| `nnue-sse` | `-DUSE_NNUE -DUSE_SSE41 -msse4.1` | NNUE + SSE4.1 fallback |
| `nnue-neon` | `-DUSE_NNUE -DUSE_NEON` | NNUE on ARM (Android, Apple Silicon) |
| `syzygy` | `-DUSE_SYZYGY` + Fathom sources | Endgame tablebase probing |
| `full` | all of the above | Maximum-strength build |
| `fast` | `-O3 -march=native -DNDEBUG` | Peak speed for native CPU |
| `debug` | `-O0 -g -fsanitize=address,undefined` | Bug hunting |
| `profile` | `-O2 -pg -DNDEBUG` | gprof profiling |

---

### Cross-Platform Script (`cross_platform.sh`)

Builds release binaries for multiple targets in one pass:

```bash
chmod +x cross_platform.sh
./cross_platform.sh
```

Output:

```
dist/
├── hugine-linux-x86_64
├── hugine-linux-aarch64
├── hugine-windows-x86_64.exe
├── hugine-android-aarch64
└── hugine-macos-x86_64          # requires osxcross
```

Required cross-compilers (install as needed):

```bash
# Windows cross-compile (MinGW)
sudo apt install gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64

# ARM64 Linux cross-compile
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Android NDK — set your NDK path
export ANDROID_NDK=/path/to/android-ndk
```

---

### macOS

#### Building natively on macOS

```bash
# Using Homebrew LLVM (recommended)
brew install llvm

/opt/homebrew/opt/llvm/bin/clang++ \
    -O3 -std=c++17 -pthread -DUSE_NNUE -DUSE_NEON -DNDEBUG \
    hugine.cpp -o hugine
```

#### Cross-compiling for macOS from Linux (osxcross)

```bash
# 1. Clone and build osxcross
git clone https://github.com/tpoechtrager/osxcross
cd osxcross
# Place your MacOSX SDK tarball in tarballs/, then:
./build.sh
export PATH="$PWD/target/bin:$PATH"

# 2. Build for macOS x86-64
x86_64-apple-darwin21-clang++ \
    -O3 -std=c++17 -pthread -DUSE_NNUE -DNDEBUG \
    hugine.cpp -o hugine-macos-x86_64

# 3. Build for Apple Silicon (ARM64)
aarch64-apple-darwin21-clang++ \
    -O3 -std=c++17 -pthread -DUSE_NNUE -DUSE_NEON -DNDEBUG \
    hugine.cpp -o hugine-macos-arm64
```

---

### Windows

#### Cross-compile from Linux (MinGW)

```bash
x86_64-w64-mingw32-g++ \
    -O3 -std=c++17 -pthread -static \
    -DUSE_NNUE -DNDEBUG \
    hugine.cpp -o hugine.exe
```

#### Build on Windows with MSVC

```powershell
# PowerShell build script (included)
.\windows_cross.ps1

# Or manually with cl.exe in a Developer Command Prompt
cl /O2 /std:c++17 /EHsc /DUSE_NNUE hugine.cpp /Fe:hugine.exe
```

---

### Android / Termux

#### On-device (inside Termux)

```bash
pkg install clang
clang++ -O3 -std=c++17 -pthread -DUSE_NNUE -DUSE_NEON -DNDEBUG \
    hugine.cpp -o hugine
```

#### Cross-compile with Android NDK

```bash
$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ \
    -O3 -std=c++17 -pthread -static \
    -DUSE_NNUE -DUSE_NEON -DNDEBUG \
    hugine.cpp -o hugine-android-arm64
```

---

### NNUE Network Setup

Place the network file in any of the following locations (resolved in this order):

```
./hugine.nnue
./nnue/hugine.nnue
<same directory as binary>/hugine.nnue
```

Or specify at runtime:

```
setoption name EvalFile value /path/to/network.nnue
```

---

## 🎮 UCI Usage

Connect Hugine to any UCI-compatible GUI (Arena, Cute Chess, Banksia, En Croissant, Fritz) or drive it directly from the terminal:

```
./hugine
uci
isready
position startpos moves e2e4 e7e5
go depth 15
```

### Key UCI Options

| Option | Type | Default | Description |
|---|---|---|---|
| `Hash` | spin | 64 | Transposition table size (MB) |
| `Threads` | spin | 1 | Number of search threads (Lazy SMP) |
| `Ponder` | check | false | Enable pondering on opponent's time |
| `MultiPV` | spin | 1 | Number of principal variations to output |
| `MoveOverhead` | spin | 10 | GUI/network latency compensation (ms) |
| `UCI_Chess960` | check | false | Chess960 / Fischer Random mode |
| `Skill Level` | spin | 20 | Playing strength limiter (0 = weakest, 20 = full) |
| `UCI_ShowWDL` | check | false | Show Win/Draw/Loss % in info lines |
| `OwnBook` | check | true | Enable/disable opening book |
| `BookFile` | string | — | Path to Polyglot `.bin` opening book |
| `EvalFile` | string | hugine.nnue | NNUE network file |
| `NNUEPath` | string | — | Directory to search for NNUE files |
| `SyzygyPath` | string | — | Syzygy tablebase directory (`;`-separated) |
| `SyzygyProbeDepth` | spin | 1 | Minimum depth to probe tablebases |
| `Syzygy50MoveRule` | check | true | Respect 50-move rule in tablebase probing |

---

## ⚔️ vs Stockfish

An honest, architectural comparison. Stockfish is the gold standard of open-source engines built by 100+ contributors over 15+ years — Hugine is a solo project. This table reflects design choices, not an Elo battle.

| Aspect | Hugine 5.0 Iota | Stockfish 17 |
|---|---|---|
| **Codebase** | Single file · ~5k lines | Multi-file · 150k+ lines |
| **Authors** | Solo (0xbytecode) | 100+ contributors |
| **Language** | C++17 | C++17 |
| **UCI** | ✅ Full | ✅ Full |
| **Search** | Alpha-beta + PVS | Alpha-beta + PVS |
| **Null-move pruning** | ✅ | ✅ |
| **LMR** | ✅ Log tables | ✅ Highly tuned |
| **ProbCut** | ✅ | ✅ |
| **SEE pruning** | ✅ | ✅ |
| **Singular extensions** | ✅ Basic | ✅ Multi-cut / multi-extension |
| **Evaluation** | HCE + NNUE (hybrid) | NNUE only (HalfKAv2) |
| **NNUE architecture** | 40960→256→32→32→1 | HalfKAv2 768→2048→1 |
| **NNUE training** | ❌ User-provided weights | ✅ Self-play (billions of games) |
| **SIMD** | AVX2 / SSE4.1 / NEON | AVX-512 / AVX2 / VNNI / NEON |
| **Threading** | Lazy SMP | Lazy SMP (highly tuned) |
| **Tablebases** | ✅ Syzygy (Fathom) | ✅ Syzygy + Gaviota |
| **Opening book** | ✅ Polyglot `.bin` | ❌ Relies on GUI |
| **Perft speed** | ~11–14M nps | ~250M+ nps |
| **Search NPS** | ~400–700k (1 thread) | ~2–4M+ (1 thread) |
| **Estimated Elo** | ~1800–2200¹ | 3600+ |
| **Platforms** | Linux · Win · macOS · Android | Linux · Win · macOS |
| **Build complexity** | Single file, one command | Multi-step CMake |
| **License** | Free, non-commercial | GPL-3.0 |

> ¹ Elo estimate assumes untrained/synthetic NNUE weights. With a properly trained network, evaluation quality — and therefore playing strength — improves substantially.

---

## 🗺 Roadmap

### Active
- [ ] **Polyglot-compatible book hashing** — implement the 781 standard Polyglot Zobrist values (`random.h`) so commercial books work natively
- [ ] **NNUE trainer** — self-play data generation and network training toolchain

### Planned
- [ ] **Syzygy WDL probing** — add WDL table support alongside current DTZ probing
- [ ] **Contempt factor** — tunable draw avoidance vs stronger/weaker opponents
- [ ] **Larger NNUE architecture** — experiment with 512-node hidden layer
- [ ] **Opening book builder** — generate `.bin` books from PGN collections
- [ ] **Automated Elo benchmarking** — match infrastructure vs reference engines
- [ ] **Lightweight GUI** — standalone cross-platform interface (Qt or web-based)

### Completed in 5.0 Iota ✅
- [x] TT mate-score normalization
- [x] Repetition detection fix
- [x] `bestmove 0000` regression on Android/ARM (memory ordering fix)
- [x] LMR log tables (Stockfish-style)
- [x] SEE bad-capture pruning
- [x] Lock-free TT with atomic XOR
- [x] Triangular PV table (stack-allocated, no heap)
- [x] NNUE multi-directory smart loading
- [x] Opening book weighted random selection
- [x] Lazy SMP thread safety (per-thread Position copies, TLS accumulators)
- [x] Multi-thread TT age corruption fix
- [x] Thread-0-only time management (stop race condition fixed)
- [x] UCI `score cp` / `score mate` keyword fix
- [x] Full cross-platform build system
- [x] 74/74 test suite (UCI · perft · puzzles · NNUE · book · TM · robustness · stress)

---

## 📂 Repository Structure

```
hugine/
├── hugine.cpp              # ← Entire engine (single source file)
├── build.sh                # Smart cross-platform build script
├── GNUmakefile             # GNU Make targets
├── CMakeLists.txt          # CMake build
├── cross_platform.sh       # Multi-target cross-compile script
├── windows_cross.ps1       # PowerShell / MSVC build script
├── nnue/
│   └── hugine.nnue         # Place NNUE weights here
├── books/
│   └── *.bin               # Place Polyglot opening books here
├── README.md
├── CHANGELOG.txt
└── LICENSE
```

---

## 📜 License

```
Hugine Chess Engine
Copyright (c) 2024–2025  0xbytecode (ABDO10DZ)

Permission is granted, free of charge, to any person obtaining a copy
of this software, to use, copy, modify, and distribute it for
NON-COMMERCIAL purposes only, subject to the following conditions:

1. This copyright notice must be included in all copies or substantial
   portions of the software.

2. Commercial use — including selling, licensing, embedding in commercial
   products, or using as part of a commercial service — is strictly
   prohibited without prior written permission from the author.

3. Derivative works must retain attribution to the original author
   and must not misrepresent the origin of the software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
THE AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES ARISING FROM ITS USE.

For commercial licensing inquiries:
  https://github.com/ABDO10DZ
```

---

## 👤 Author

<div align="center">

**0xbytecode**

[![GitHub](https://img.shields.io/badge/GitHub-ABDO10DZ-black?style=flat-square&logo=github)](https://github.com/ABDO10DZ)
[![Repository](https://img.shields.io/badge/Repo-hugine-blue?style=flat-square&logo=github)](https://github.com/ABDO10DZ/hugine)

*Built from scratch with a passion for chess, low-level systems programming,*
*and the challenge of making something simultaneously fast, correct, and portable.*

---

**Hugine 5.0 Iota** · ♟ Built in C++17

</div>
