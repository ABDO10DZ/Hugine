# Hugine 2.0

> A UCI chess engine written in C++17 with parallel search, Syzygy endgame tablebases, optional NNUE evaluation, Chess960 support, and an on-disk learning system.

---

## Features

### Move Generation
- Bitboard-based pseudo-legal move generator (magic bitboards for sliding pieces)
- Fully correct castling: standard and Chess960 (king-to-rook UCI protocol)
- En passant, promotions, and all special moves
- SEE (Static Exchange Evaluation) for move ordering and pruning

### Search
- Iterative deepening with aspiration windows (±15 cp, widening by 50)
- Principal Variation Search (PVS / negamax with α-β)
- Quiescence search (depth cap 8, full evasions when in check)
- **Parallel search**: YBWC (Young Brothers Wait Concept) split points, up to 64 threads

| Technique | Details |
|---|---|
| Null move pruning | R = 2 + depth/6 |
| Late Move Reductions (LMR) | base 1, divisor 2 |
| Late Move Pruning (LMP) | base 3, factor 2 |
| Futility pruning | 200 cp × depth |
| Razoring | depth ≤ 6, margins 300/400/600 |
| Static null move | depth > 7, margin 200 |
| Singular extensions | depth ≥ 8, margin 50 |
| ProbCut | depth ≥ 5 |
| IID (Internal Iterative Deepening) | depth ≥ 5, reduction 2 |
| Multi-cut | depth ≥ 6, 2 of top 3 fail-high |
| Check extensions | always |
| Recapture extensions | previous ply captured on same square |
| Passed pawn extensions | advancing past the midpoint |

### Move Ordering
- TT move first
- Killer moves (2 per ply)
- Counter moves
- Follow-up moves
- History heuristic (gravity-scaled, max ±16384)
- Butterfly history
- Capture history (indexed by moving piece × captured piece × target)
- Continuation history (2-ply)
- Correction history

### Evaluation
- Piece-square tables (tapered by game phase)
- Pawn structure: passed pawns, isolated pawns
- Mobility bonuses per piece type
- Outpost detection for knights and bishops
- King safety (shelter pawns, pawn storm, open files)
- Space control
- Material imbalance
- Bishop pair bonus
- Rook on open/semi-open files
- Endgame king activity

**Optional NNUE** (compile with `-DUSE_NNUE`): HalfKP architecture, 256-neuron feature transformer, incremental updates, int8 weights, SIMD-ready.

### Endgame Tablebases
- Syzygy DTZ/WDL via [Fathom](https://github.com/jdart1/Fathom) (auto-detected at compile time)
- Root DTZ probing for optimal endgame play
- Interior WDL probing at depth ≤ 3
- Falls back to evaluation if tablebases not present — no stub required

### Opening Book
- Polyglot `.bin` format
- Configurable variety (0 = best move, 10 = max randomness)

### Learning System
- In-memory hash table (1M entries) updated from game results
- Adjustments applied at every node during search
- Persistent save/load via `LearningFile`

### Time Management
- Soft/hard limits with configurable move overhead
- Adaptive scaling based on score stability and best-move changes
- Full support for `wtime`/`btime`/`winc`/`binc`/`movestogo`/`movetime`/`infinite`/`ponder`

---

## Building

### Requirements
- C++17 compiler (GCC 9+, Clang 10+)
- POSIX threads (`-pthread`)

### Basic Build
```bash
g++ -O2 -std=c++17 -pthread -march=native hugine.cpp -o hugine
```

### With Syzygy Tablebases
Clone Fathom into a `fathom/` subdirectory alongside `hugine.cpp`:
```bash
git clone https://github.com/jdart1/Fathom fathom
g++ -O2 -std=c++17 -pthread -march=native hugine.cpp fathom/src/tbprobe.cpp -o hugine
```
Hugine detects the header automatically at compile time — no extra flags needed.

### With NNUE
```bash
g++ -O2 -std=c++17 -pthread -march=native -DUSE_NNUE hugine.cpp -o hugine
```
Place your network file and point to it with `setoption name EvalFile value <path>`.

### Build Flags Summary

| Flag | Effect |
|---|---|
| `-march=native` | Enables hardware-specific SIMD (recommended) |
| `-DUSE_NNUE` | Enables NNUE evaluation layer |
| `-DNO_SYZYGY` | Force-disables Syzygy even if Fathom is present |
| `-DDEBUG` | Enables extra diagnostics (hash clearing messages, etc.) |

---

## Installation

Hugine is a command-line UCI engine. Use it with any UCI-compatible GUI:

| GUI | Platform | Notes |
|---|---|---|
| [Cute Chess](https://cutechess.com) | Linux / macOS / Windows | Great for engine matches |
| [Arena](http://www.playwitharena.de) | Windows | Feature-rich free GUI |
| [Scid vs. PC](https://scidvspc.sourceforge.net) | Linux / macOS / Windows | Database + engine |
| [En Croissant](https://encroissant.org) | Linux / macOS / Windows | Modern, open source |
| [BanksiaGUI](https://banksiagui.com) | Linux / macOS / Windows | Multi-engine testing |
| [ChessBase](https://www.chessbase.com) | Windows | Commercial |

**Add to your GUI**: point it to the `hugine` binary and select UCI protocol.

---

## UCI Options

| Option | Type | Default | Range | Description |
|---|---|---|---|---|
| `Hash` | spin | 256 | 1–8192 | Transposition table size in MB |
| `Threads` | spin | 1 | 1–64 | Number of search threads (YBWC) |
| `Ponder` | check | false | — | Allow thinking on opponent's time |
| `MultiPV` | spin | 1 | 1–5 | Number of lines to report |
| `Contempt` | spin | 0 | -100–100 | Contempt factor in centipawns |
| `Move Overhead` | spin | 100 | 0–5000 | Safety buffer (ms) subtracted from time |
| `OwnBook` | check | true | — | Use built-in opening book |
| `BookFile` | string | — | — | Path to Polyglot `.bin` opening book |
| `BookVariety` | spin | 0 | 0–10 | Book move randomness |
| `SyzygyPath` | string | — | — | Directory containing Syzygy `.rtbw`/`.rtbz` files |
| `EvalFile` | string | — | — | NNUE network file path (requires `-DUSE_NNUE`) |
| `UCI_Chess960` | check | false | — | Enable Chess960 (FRC) mode |
| `UCI_LimitStrength` | check | false | — | Enable Elo-limited play |
| `UCI_Elo` | spin | 1500 | 800–3000 | Target Elo (when LimitStrength is on) |
| `Learning` | check | false | — | Enable position learning |
| `LearningFile` | string | — | — | File for persistent learning data |
| `LearningRate` | spin | 100 | 1–1000 | Learning adjustment speed |
| `LearningMaxAdjust` | spin | 50 | 0–200 | Maximum centipawn adjustment per position |
| `Clear Learning` | button | — | — | Wipe in-memory learning table |
| `Save Learning` | button | — | — | Write learning table to `LearningFile` |
| `Clear Hash` | button | — | — | Reset transposition table |
| `TuningMode` | check | false | — | Log positions + scores for tuning |
| `TuningFile` | string | — | — | Output file for tuning data |

---

## Custom Commands

Beyond the standard UCI protocol, Hugine supports these extra commands at the prompt:

### `d` — Display position
Prints the board, FEN, side to move, en passant square, 50-move counter, and full castling rights diagnostic (which rook, where the king and rook land).

```
position fen r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1
d
```

### `perft <depth>` — Performance test
Counts leaf nodes at the given depth and reports a move-by-move breakdown (divide), total nodes, time, and NPS.

```
position startpos
perft 5
```

Expected output for the start position:
```
a2a3: 181046
...
Nodes searched: 4865609  depth: 5  time: 430ms  nps: 11,300,000
```

### `eval` — Static evaluation
Prints the evaluation of the current position in centipawns from the side to move's perspective.

### `learn result <win|draw|loss>` — Record result
Updates the learning table with the game outcome for all positions in the last search's PV. Enable with `setoption name Learning value true` first.

```
learn result win
```

---

## Chess960 (Fischer Random Chess)

Enable with `setoption name UCI_Chess960 value true` **before** setting the position.

Move input and output follow the UCI Chess960 convention: castling is sent as **king-to-rook** (e.g. `e1h1` for White kingside) rather than king-to-destination (`e1g1`).

```
setoption name UCI_Chess960 value true
position fen r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4
go depth 10
```

---

## Verification

Run the built-in perft suite to confirm correct move generation:

```bash
printf "position startpos\nperft 5\nquit\n" | ./hugine
# Expected: 4865609

printf "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\nperft 4\nquit\n" | ./hugine
# Expected: 4085603

printf "position fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1\nperft 6\nquit\n" | ./hugine
# Expected: 11030083
```

---

## Project Structure

```
hugine.cpp          — Full engine source (single file, ~4200 lines)
fathom/             — Syzygy tablebase probing library (optional, git submodule)
README.md           — This file
```

The source is divided into four self-contained parts:
1. **Core Foundation** — types, bitboards, Zobrist hashing, FEN parsing, make/undo move, move generation
2. **Evaluation** — HCE evaluation, optional NNUE, Syzygy probing, opening book, learning, time management
3. **Search** — negamax, quiescence, YBWC parallel search, move ordering, all pruning techniques
4. **UCI Interface** — command parser, `d`/`perft`/`eval`/`learn`, `go`/`stop`/`ponderhit`

---

## Author

**0xbytecode**
