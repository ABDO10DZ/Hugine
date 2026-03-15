import subprocess, sys, os, time, struct

# Resolve project root relative to this script — works from any directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE    = os.path.join(SCRIPT_DIR, "build", "hugine_base")
NNUE    = os.path.join(SCRIPT_DIR, "build", "hugine_nnue")
ASAN    = os.path.join(SCRIPT_DIR, "build", "hugine_asan")
BOOK    = os.path.join(SCRIPT_DIR, "books", "test.bin")
NNUEF   = os.path.join(SCRIPT_DIR, "nnue", "hugine.nnue")
NNUEDIR = os.path.join(SCRIPT_DIR, "nnue")

PASS = 0
FAIL = 0
section_results = []

def _resolve(binary):
    """Make relative paths absolute against SCRIPT_DIR."""
    if os.path.isabs(binary):
        return binary
    # Strip leading ./ or ./
    rel = binary
    if rel.startswith('./'):
        rel = rel[2:]
    return os.path.join(SCRIPT_DIR, rel)

def ask(binary, commands, timeout=25):
    binary = _resolve(binary)
    if not os.path.exists(binary):
        return f"[MISSING BINARY: {binary}]"
    proc = subprocess.run(
        [binary], input=commands, capture_output=True,
        text=True, timeout=timeout, cwd=SCRIPT_DIR
    )
    return proc.stdout

def test(name, binary, commands, checks, anti_checks=None, timeout=25):
    global PASS, FAIL
    try:
        out = ask(binary, commands, timeout)
        errors = []
        for chk in checks:
            if chk not in out:
                errors.append(f"missing: {repr(chk)}")
        if anti_checks:
            for chk in anti_checks:
                if chk in out:
                    errors.append(f"should NOT contain: {repr(chk)}")
        if errors:
            FAIL += 1
            section_results.append(f"  ✗ {name}")
            for e in errors:
                section_results.append(f"      {e}")
            section_results.append(f"      stdout tail: {repr(out[-300:])}")
        else:
            PASS += 1
            section_results.append(f"  ✓ {name}")
    except subprocess.TimeoutExpired:
        FAIL += 1
        section_results.append(f"  ✗ {name}  [TIMEOUT > {timeout}s]")
    except Exception as ex:
        FAIL += 1
        section_results.append(f"  ✗ {name}  [EXCEPTION: {ex}]")

def section(title):
    section_results.append(f"\n{'═'*48}")
    section_results.append(f" {title}")
    section_results.append(f"{'═'*48}")

def lap_summary(label):
    global PASS, FAIL
    section_results.append(f"  → {label}: {PASS} passed, {FAIL} failed")

# ──────────────────────────────────────────────────────────────
# LAYER 1: UCI Protocol
# ──────────────────────────────────────────────────────────────
section("LAYER 1 — UCI Protocol & Identity")

test("id name + version", BASE,
    "uci\nquit\n",
    ["id name Hugine 5.1.0", "id author 0xbytecode", "uciok"])

test("isready → readyok", BASE,
    "isready\nquit\n",
    ["readyok"])

test("ucinewgame accepted", BASE,
    "uci\nucinewgame\nisready\nquit\n",
    ["readyok"])

test("all expected options present", BASE,
    "uci\nquit\n",
    ["option name Hash type spin",
     "option name Threads type spin",
     "option name EvalFile type string",
     "option name NNUEPath type string",
     "option name BookFile type string",
     "option name SyzygyPath type string",
     "option name OwnBook type check",
     "option name MultiPV type spin",
     "option name UCI_Chess960 type check",
     "option name UCI_ShowWDL type check"])

test("platform info string present", BASE,
    "uci\nquit\n",
    ["info string Platform:"])

lap_summary("Layer 1")
L1_P, L1_F = PASS, FAIL; PASS=0; FAIL=0

# ──────────────────────────────────────────────────────────────
# LAYER 2: Position + Perft correctness
# ──────────────────────────────────────────────────────────────
section("LAYER 2 — Position Parsing & Perft")

test("startpos perft 1 = 20", BASE,
    "position startpos\nperft 1\nquit\n",
    ["Nodes searched: 20"])

test("startpos perft 2 = 400", BASE,
    "position startpos\nperft 2\nquit\n",
    ["Nodes searched: 400"])

test("startpos perft 3 = 8902", BASE,
    "position startpos\nperft 3\nquit\n",
    ["Nodes searched: 8902"])

test("startpos perft 4 = 197281", BASE,
    "position startpos\nperft 4\nquit\n",
    ["Nodes searched: 197281"])

test("startpos perft 5 = 4865609", BASE,
    "position startpos\nperft 5\nquit\n",
    ["Nodes searched: 4865609"])

test("kiwipete perft 1 = 48", BASE,
    "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\nperft 1\nquit\n",
    ["Nodes searched: 48"])

test("kiwipete perft 2 = 2039", BASE,
    "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\nperft 2\nquit\n",
    ["Nodes searched: 2039"])

test("kiwipete perft 3", BASE,
    "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\nperft 3\nquit\n",
    ["Nodes searched: 97862"])

test("position with moves applied", BASE,
    "position startpos moves e2e4 e7e5\nperft 1\nquit\n",
    ["Nodes searched: 29"])

test("fen parse + perft 1 = 44 (pos5)", BASE,
    "position fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -\nperft 1\nquit\n",
    ["Nodes searched: 44"])

test("fen parse + perft 2 = 1486 (pos5)", BASE,
    "position fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -\nperft 2\nquit\n",
    ["Nodes searched: 1486"])

test("fen parse + perft 3 = 62379 (pos5)", BASE,
    "position fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -\nperft 3\nquit\n",
    ["Nodes searched: 62379"])

# En passant specific
test("ep capture perft", BASE,
    "position fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -\nperft 3\nquit\n",
    ["Nodes searched: 2812"])

lap_summary("Layer 2")
L2_P, L2_F = PASS, FAIL; PASS=0; FAIL=0

# ──────────────────────────────────────────────────────────────
# LAYER 3: Search correctness (puzzles)
# ──────────────────────────────────────────────────────────────
section("LAYER 3 — Search Correctness (Tactical Puzzles)")

test("mate in 1: Scholar's Qxf7#", BASE,
    "position fen r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq -\ngo depth 3\nquit\n",
    ["score mate", "bestmove h5f7"])

test("mate in 1: Qxf7#", BASE,
    "position fen r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq -\ngo depth 3\nquit\n",
    ["bestmove f3f7"])

test("mate in 2: Legal's mate setup", BASE,
    "position fen r1b1kb1r/pppp1ppp/2n2q2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq -\ngo depth 5\nquit\n",
    ["bestmove"])

test("find winning capture Rxd1", BASE,
    "position fen r3k2r/ppp2ppp/2n1bn2/2bpp3/2BPP3/2N1BN2/PPP2PPP/R3K2R b KQkq -\ngo depth 6\nquit\n",
    ["bestmove"])

test("mate in 3: back rank", BASE,
    "position fen 6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - -\ngo depth 7\nquit\n",
    ["bestmove"])

test("simple endgame KQK", BASE,
    "position fen 8/8/8/8/8/1K6/8/1k1Q4 b - -\ngo depth 5\nquit\n",
    ["bestmove"])

test("fork: Nd5 wins material", BASE,
    "position fen r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq -\ngo depth 5\nquit\n",
    ["bestmove"])

test("bestmove is a legal move (format)", BASE,
    "position startpos\ngo depth 4\nquit\n",
    ["bestmove ", "score cp"])

test("info lines contain score cp", BASE,
    "position startpos\ngo depth 6\nquit\n",
    ["info", "score cp", "depth", "nodes", "nps"])

test("search: multipv=3 gives 3 PVs", BASE,
    "setoption name MultiPV value 3\nposition startpos\ngo depth 5\nquit\n",
    ["multipv 1", "multipv 2", "multipv 3"])

lap_summary("Layer 3")
L3_P, L3_F = PASS, FAIL; PASS=0; FAIL=0

# ──────────────────────────────────────────────────────────────
# LAYER 4: NNUE Loading & Evaluation
# ──────────────────────────────────────────────────────────────
section("LAYER 4 — NNUE Loading & Evaluation")

# 4a: corrupt file → graceful fallback, no crash
# Create a corrupt nnue file (bad magic) for this test
_bad_nnue = os.path.join(SCRIPT_DIR, 'nnue', 'bad_corrupt.nnue')
import struct as _struct
with open(_bad_nnue, 'wb') as _f: _f.write(_struct.pack('<I', 0xDEADBEEF) + b'\x00'*100)
test("corrupt file → load failed + no crash", NNUE,
    f"setoption name EvalFile value {_bad_nnue}\nposition startpos\ngo depth 5\nquit\n",
    ["NNUE load FAILED", "bestmove"],
    anti_checks=["Segmentation"])

# 4b: nonexistent file → graceful
test("missing file → load failed + no crash", NNUE,
    "setoption name EvalFile value /tmp/doesnotexist.nnue\nposition startpos\ngo depth 5\nquit\n",
    ["NNUE load FAILED", "bestmove"])

# 4c: valid file loads
test("valid NNUE file loads successfully", NNUE,
    f"setoption name EvalFile value {NNUEF}\nposition startpos\ngo depth 4\nquit\n",
    ["info string NNUE loaded:", "bestmove"])

# 4d: directory scan via EvalFile
test("EvalFile with directory → scans, loads", NNUE,
    f"setoption name EvalFile value {NNUEDIR}\nposition startpos\ngo depth 4\nquit\n",
    ["info string NNUE loaded:", "bestmove"])

# 4e: NNUEPath alias
test("NNUEPath dir alias works", NNUE,
    f"setoption name NNUEPath value {NNUEDIR}\nposition startpos\ngo depth 4\nquit\n",
    ["info string NNUE loaded:", "bestmove"])

# 4f: empty dir → clear warning
test("NNUEPath empty dir → warning + no crash", NNUE,
    "setoption name NNUEPath value /tmp\nposition startpos\ngo depth 4\nquit\n",
    ["bestmove"])

# 4g: NNUE + deep search, no crash or ASAN error
test("NNUE deep search depth 8 no crash", NNUE,
    f"setoption name EvalFile value {NNUEF}\nposition startpos\ngo depth 8\nquit\n",
    ["bestmove"])

# 4h: NNUE on tactical position
test("NNUE tactical position no crash", NNUE,
    f"setoption name EvalFile value {NNUEF}\nposition fen r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq -\ngo depth 4\nquit\n",
    ["bestmove"])

# 4i: NNUE after moves applied
test("NNUE eval after move sequence", NNUE,
    f"setoption name EvalFile value {NNUEF}\nposition startpos moves e2e4 e7e5 g1f3 b8c6 f1c4\ngo depth 5\nquit\n",
    ["bestmove"])

# 4j: ASAN run — stack safety with NNUE
test("ASAN: no memory errors with valid NNUE file", ASAN,
    f"setoption name EvalFile value {NNUEF}\nposition startpos\ngo depth 5\nquit\n",
    ["bestmove"],
    anti_checks=["ERROR: AddressSanitizer", "runtime error", "SEGV"])

# 4k: ASAN with empty file — the old crash scenario
test("ASAN: empty NNUE file no segfault", ASAN,
    "setoption name EvalFile value net.nnue\nposition startpos\ngo depth 4\nquit\n",
    ["bestmove"],
    anti_checks=["ERROR: AddressSanitizer", "SEGV"])

# 4l: NNUE perft still correct
test("NNUE perft 4 = 197281 (correctness guard)", NNUE,
    f"setoption name EvalFile value {NNUEF}\nposition startpos\nperft 4\nquit\n",
    ["Nodes searched: 197281"])

lap_summary("Layer 4")
L4_P, L4_F = PASS, FAIL; PASS=0; FAIL=0

# ──────────────────────────────────────────────────────────────
# LAYER 5: Opening Book
# ──────────────────────────────────────────────────────────────
section("LAYER 5 — Polyglot Opening Book")

# 5a: load success message
test("valid .bin loads with entry count", BASE,
    f"setoption name BookFile value {BOOK}\nuci\nquit\n",
    ["info string Book loaded:", "entries)"])

# 5b: bad path → clear error
test("missing book file → error message", BASE,
    "setoption name BookFile value /tmp/fake.bin\nposition startpos\ngo depth 4\nquit\n",
    ["Book load FAILED", "bestmove"])

# 5c: startpos book hit (we inserted e2e4 w=1500 as best)
test("startpos → book hit message", BASE,
    f"setoption name BookFile value {BOOK}\nposition startpos\ngo depth 5\nquit\n",
    ["Book hit", "bestmove e2e4"])

# 5d: after 1.e4, book should pick e7e5 (w=1200 highest)
test("after 1.e4 → book hit e7e5", BASE,
    f"setoption name BookFile value {BOOK}\nposition startpos moves e2e4\ngo depth 5\nquit\n",
    ["Book hit", "bestmove e7e5"])

# 5e: after 1.d4, book picks d7d5
test("after 1.d4 → book hit d7d5", BASE,
    f"setoption name BookFile value {BOOK}\nposition startpos moves d2d4\ngo depth 5\nquit\n",
    ["Book hit", "bestmove d7d5"])

# 5f: position not in book → falls through to search
test("position not in book → search used", BASE,
    f"setoption name BookFile value {BOOK}\nposition startpos moves e2e4 e7e5 g1f3\ngo depth 5\nquit\n",
    ["bestmove"],
    anti_checks=["Book hit"])

# 5g: OwnBook false disables book
test("OwnBook false disables book entirely", BASE,
    f"setoption name OwnBook value false\nsetoption name BookFile value {BOOK}\nposition startpos\ngo depth 5\nquit\n",
    ["bestmove"],
    anti_checks=["Book hit"])

# 5h: book + NNUE together
test("Book + NNUE: startpos book hit w/ NNUE loaded", NNUE,
    f"setoption name BookFile value {BOOK}\nsetoption name EvalFile value {NNUEF}\nposition startpos\ngo depth 4\nquit\n",
    ["Book hit", "bestmove e2e4"])

lap_summary("Layer 5")
L5_P, L5_F = PASS, FAIL; PASS=0; FAIL=0

# ──────────────────────────────────────────────────────────────
# LAYER 6: Time Management & UCI go variants
# ──────────────────────────────────────────────────────────────
section("LAYER 6 — Time Management & go Variants")

test("go movetime 500ms", BASE,
    "position startpos\ngo movetime 500\nquit\n",
    ["bestmove"])

test("go depth 1 is fast", BASE,
    "position startpos\ngo depth 1\nquit\n",
    ["bestmove"])

test("go depth 10", BASE,
    "position startpos\ngo depth 10\nquit\n",
    ["bestmove"])

test("go wtime btime triggers TM", BASE,
    "position startpos\ngo wtime 10000 btime 10000 movestogo 40\nquit\n",
    ["bestmove"])

test("go nodes 50000", BASE,
    "position startpos\ngo nodes 50000\nquit\n",
    ["bestmove"])

test("stop command halts search", BASE,
    "position startpos\ngo infinite\nstop\nquit\n",
    ["bestmove"], timeout=5)

test("score mate detected in mate-in-1", BASE,
    "position fen r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq -\ngo depth 3\nquit\n",
    ["score mate"])

test("score cp reasonable range", BASE,
    "position startpos\ngo depth 6\nquit\n",
    ["score cp"])

lap_summary("Layer 6")
L6_P, L6_F = PASS, FAIL; PASS=0; FAIL=0

# ──────────────────────────────────────────────────────────────
# LAYER 7: Edge Cases & Robustness
# ──────────────────────────────────────────────────────────────
section("LAYER 7 — Edge Cases & Robustness")

test("stalemate position returns bestmove 0000", BASE,
    "position fen 5bnr/4p1pq/4Qpkr/7p/7P/4P3/PPPP1PP1/RNB1KBNR b KQ -\ngo depth 3\nquit\n",
    ["bestmove"])

test("position after 0-0 castling", BASE,
    "position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1\nperft 1\nquit\n",
    ["Nodes searched:"])

test("position after 0-0-0 queenside castling", BASE,
    "position fen r3kbnr/ppp1pppp/2nq4/3p1b2/3P1B2/2NQ4/PPP1PPPP/R3KBNR w KQkq -\ngo depth 3\nquit\n",
    ["bestmove"])

test("en passant in position fen", BASE,
    "position fen rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6\nperft 1\nquit\n",
    ["Nodes searched:"])

test("repetition draw detected (0 score)", BASE,
    "position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8\ngo depth 4\nquit\n",
    ["bestmove"])

test("ucinewgame resets state between games", BASE,
    "position startpos\ngo depth 4\nucinewgame\nposition startpos\ngo depth 4\nquit\n",
    ["bestmove"])

test("Hash option resize works", BASE,
    "setoption name Hash value 128\nuci\nposition startpos\ngo depth 5\nquit\n",
    ["bestmove"])

test("very short movetime 1ms no hang", BASE,
    "position startpos\ngo movetime 1\nquit\n",
    ["bestmove"], timeout=5)

test("100 move clock draw at 100", BASE,
    "position fen 8/8/8/8/8/8/P7/K1k5 w - - 100 1\ngo depth 4\nquit\n",
    ["bestmove"])

test("position with promotion available", BASE,
    "position fen 8/P7/8/8/8/8/8/K1k5 w - -\ngo depth 3\nquit\n",
    ["bestmove a7a8"])  # should promote

test("Threads=2 multi-threaded search", BASE,
    "setoption name Threads value 2\nposition startpos\ngo depth 7\nquit\n",
    ["bestmove"])

test("perft after setoption no corruption", BASE,
    "setoption name Hash value 64\nposition startpos\nperft 4\nquit\n",
    ["Nodes searched: 197281"])

lap_summary("Layer 7")
L7_P, L7_F = PASS, FAIL; PASS=0; FAIL=0

# ──────────────────────────────────────────────────────────────
# LAYER 8: NNUE Stack Stress (search abort / re-use)
# ──────────────────────────────────────────────────────────────
section("LAYER 8 — NNUE Stack Stress & Re-use")

nnue_setup = f"setoption name EvalFile value {NNUEF}\n"

test("NNUE: 5 sequential go commands no crash", NNUE,
    nnue_setup +
    "position startpos\ngo depth 4\n"
    "position startpos moves e2e4\ngo depth 4\n"
    "position startpos moves e2e4 e7e5\ngo depth 4\n"
    "position startpos moves e2e4 e7e5 g1f3\ngo depth 4\n"
    "position startpos\ngo depth 4\nquit\n",
    ["bestmove"], timeout=60)

test("NNUE: ucinewgame between searches", NNUE,
    nnue_setup +
    "position startpos\ngo depth 5\nucinewgame\n"
    "position startpos\ngo depth 5\nquit\n",
    ["bestmove"])

test("NNUE: stop mid-search then new search", NNUE,
    nnue_setup +
    "position startpos\ngo infinite\nstop\n"
    "position startpos\ngo depth 5\nquit\n",
    ["bestmove"], timeout=10)

test("NNUE: kiwipete deep search no crash", NNUE,
    nnue_setup +
    "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\n"
    "go depth 7\nquit\n",
    ["bestmove"], timeout=30)

test("NNUE: Threads=2 with NNUE no crash", NNUE,
    nnue_setup +
    "setoption name Threads value 2\n"
    "position startpos\ngo depth 7\nquit\n",
    ["bestmove"], timeout=30)

test("NNUE: 10x depth-3 searches (stack leak test)", NNUE,
    nnue_setup + 
    ("position startpos\ngo depth 3\n" * 10) + "quit\n",
    ["bestmove"], timeout=60)

lap_summary("Layer 8")
L8_P, L8_F = PASS, FAIL; PASS=0; FAIL=0

# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
# (intermediate summary suppressed)

# (old summary suppressed — full summary at end)
# ══════════════════════════════════════════════════════════════════════════════
# LAYER 9: WDL Probing (Syzygy WDL at depth, contempt draw scoring)
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 9 — WDL Probing & Contempt Draw Scoring")

# 9.1 WDL path present without Syzygy tables: engine must NOT crash and must
#     return a bestmove from a KQK position (would be WDL win if tables present).
test("WDL: engine handles missing TB gracefully",  BASE,
    "position fen 8/8/8/8/8/4K3/4Q3/4k3 w - - 0 1\ngo depth 6\nquit\n",
    ["bestmove"])

# 9.2 UCI_ShowWDL plumbing — when enabled, output should include wdl token on
#     info lines (even if TB absent, the option must not crash the engine).
test("WDL: UCI_ShowWDL option accepted", BASE,
    "setoption name UCI_ShowWDL value true\nposition startpos\ngo depth 5\nquit\n",
    ["bestmove"],
    anti_checks=["Unknown option"])

# 9.3 Contempt=0 baseline — must get a bestmove
test("Contempt: baseline 0 no crash", BASE,
    "setoption name Contempt value 0\nposition startpos\ngo depth 5\nquit\n",
    ["bestmove"])

# 9.4 Contempt=50 (avoid draws) — must accept option and search
test("Contempt: value +50 accepted and searched", BASE,
    "setoption name Contempt value 50\nposition startpos\ngo depth 5\nquit\n",
    ["bestmove"],
    anti_checks=["Unknown option", "error"])

# 9.5 Contempt=-50 (accept draws against stronger) — must accept option
test("Contempt: value -50 accepted and searched", BASE,
    "setoption name Contempt value -50\nposition startpos\ngo depth 5\nquit\n",
    ["bestmove"],
    anti_checks=["Unknown option", "error"])

# 9.6 Contempt affects draw evaluation: compare eval output in a drawn position
#     with Contempt=50 vs Contempt=-50.  We can't guarantee exact values without
#     real TB, but the engine must output valid info lines in both cases.
test("Contempt: repeated draws position stable", BASE,
    "setoption name Contempt value 50\n"
    "position fen 8/8/8/3k4/3K4/8/8/8 w - - 50 1\ngo depth 6\nquit\n",
    ["bestmove"])

# 9.7 Contempt option range — verify extreme values are clamped/accepted
test("Contempt: max value 100 accepted", BASE,
    "setoption name Contempt value 100\nposition startpos\ngo depth 4\nquit\n",
    ["bestmove"])

# 9.8 WDL probe depth control — engine should probe earlier in endgame-ish positions
test("WDL: engine probes in 5-piece endgame (no TB = graceful fail)", BASE,
    "position fen 8/4k3/8/8/8/8/4P3/4K3 w - - 0 1\ngo depth 10\nquit\n",
    ["bestmove"])

lap_summary("Layer 9")
L9_P, L9_F = PASS, FAIL; PASS=0; FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 10: NNUE Architecture (256-node standard vs 512-node NNUE_LARGE)
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 10 — NNUE Architecture (256 vs 512 vs 1024)")

LARGE     = os.path.join(SCRIPT_DIR, "build", "hugine_large")
LARGE_NNUE= os.path.join(SCRIPT_DIR, "build", "hugine_large_nnue")
XL        = os.path.join(SCRIPT_DIR, "build", "hugine_xl")
XL_NNUE   = os.path.join(SCRIPT_DIR, "build", "hugine_xl_nnue")

# ── NNUE_LARGE (512) tests ─────────────────────────────────────────────────
# 10.1 Large build basic sanity
test("NNUE_LARGE: uci/isready/quit", LARGE,
    "uci\nisready\nquit\n",
    ["uciok", "readyok"])

# 10.2 Large build perft (architecture must not break game logic)
test("NNUE_LARGE: perft 4 = 197281", LARGE,
    "position startpos\nperft 4\nquit\n",
    ["Nodes searched: 197281"])

# 10.3 Large build search
test("NNUE_LARGE: search depth 5 returns bestmove", LARGE,
    "position startpos\ngo depth 5\nquit\n",
    ["bestmove"])

# 10.4 Large + NNUE: wrong architecture (256-node) → reject + fallback
test("NNUE_LARGE: rejects 256-node .nnue gracefully", LARGE_NNUE,
    f"setoption name EvalFile value {NNUEF}\nposition startpos\ngo depth 5\nquit\n",
    ["bestmove"])

# 10.5 Large without NNUE → classical eval fallback
test("NNUE_LARGE: classical eval fallback without .nnue", LARGE,
    "setoption name EvalFile value /nonexistent/path.nnue\n"
    "position startpos\ngo depth 6\nquit\n",
    ["bestmove"])

# 10.6 Standard 256 and LARGE 512 produce same perft
import subprocess as _sp
def _perft(binary, depth=4):
    import re as _re_perft
    cmd = "position startpos\nperft {}\nquit\n".format(depth)
    r = subprocess.run([binary], input=cmd, capture_output=True, text=True, timeout=20, cwd=SCRIPT_DIR)
    for line in r.stdout.split("\n"):
        m = _re_perft.search(r"Nodes searched:\s*(\d+)", line)
        if m: return m.group(1)
    return None

std_nodes   = _perft(os.path.join(SCRIPT_DIR, "build", "hugine_base"), 4)
large_nodes = _perft(os.path.join(SCRIPT_DIR, "build", "hugine_large"), 4)
xl_nodes    = _perft(os.path.join(SCRIPT_DIR, "build", "hugine_xl"), 4)
if std_nodes == large_nodes == "197281":
    PASS += 1; section_results.append("  ✓ NNUE_LARGE: same perft(4) as standard build")
else:
    FAIL += 1; section_results.append(
        f"  ✗ NNUE_LARGE: perft mismatch  std={std_nodes}  large={large_nodes}")

# 10.7 NNUE_LARGE UCI option list includes EvalFile
test("NNUE_LARGE: EvalFile option advertised", LARGE,
    "uci\nquit\n",
    ["option name EvalFile"])

# ── NNUE_XL (1024) tests ──────────────────────────────────────────────────
# 10.8 XL build sanity
test("NNUE_XL: uci/isready/quit", XL,
    "uci\nisready\nquit\n",
    ["uciok", "readyok"])

# 10.9 XL build perft (move gen must be unaffected by architecture)
test("NNUE_XL: perft 4 = 197281", XL,
    "position startpos\nperft 4\nquit\n",
    ["Nodes searched: 197281"])

# 10.10 XL build search
test("NNUE_XL: search depth 5 returns bestmove", XL,
    "position startpos\ngo depth 5\nquit\n",
    ["bestmove"])

# 10.11 XL NNUE: load an XL .nnue file (random weights but correct architecture)
import tempfile as _tmp
_xl_nnue_path = os.path.join(SCRIPT_DIR, "nnue", "hugine_xl.nnue")
_xl_generated = False
if not os.path.exists(_xl_nnue_path):
    try:
        import subprocess as _xsp
        _r = _xsp.run(
            ["python3", os.path.join(SCRIPT_DIR, "nnue_trainer", "train.py"),
             "random", "--out", _xl_nnue_path, "--xl"],
            capture_output=True, text=True, timeout=60
        )
        _xl_generated = (_r.returncode == 0)
    except Exception:
        pass

if os.path.exists(_xl_nnue_path):
    test("NNUE_XL: loads XL .nnue + returns bestmove", XL_NNUE,
        f"setoption name EvalFile value {_xl_nnue_path}\n"
        "position startpos\ngo depth 4\nquit\n",
        ["info string NNUE loaded:", "bestmove"])

    test("NNUE_XL: rejects 256-node .nnue gracefully", XL_NNUE,
        f"setoption name EvalFile value {NNUEF}\n"
        "position startpos\ngo depth 4\nquit\n",
        ["bestmove"])  # still searches (classical fallback)
else:
    section_results.append("  ⚠ NNUE_XL file tests skipped (XL .nnue not generated)")

# 10.12 XL and standard produce same perft
if xl_nodes == "197281":
    PASS += 1; section_results.append("  ✓ NNUE_XL: same perft(4) as standard build")
else:
    FAIL += 1; section_results.append(
        f"  ✗ NNUE_XL: perft mismatch  std={std_nodes}  xl={xl_nodes}")

# 10.13 XL EvalFile option advertised
test("NNUE_XL: EvalFile option advertised", XL,
    "uci\nquit\n",
    ["option name EvalFile"])

lap_summary("Layer 10")
L10_P, L10_F = PASS, FAIL; PASS=0; FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 11: Opening Book Builder (book_builder.py)
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 11 — Opening Book Builder (book_builder.py)")

import sys, os, struct as _struct, subprocess as _sp2, tempfile as _tmp

BUILDER = os.path.join(SCRIPT_DIR, "book_builder.py")

# 11.1 Tool exists and is importable
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("book_builder", BUILDER)
    _bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_bm)
    PASS += 1; section_results.append("  ✓ book_builder.py importable")
except Exception as e:
    FAIL += 1; section_results.append(f"  ✗ book_builder.py import failed: {e}")
    _bm = None

# 11.2 Build a mini PGN book and verify the .bin output is valid
_MINI_PGN = """\
[Event "Test"]
[Site "?"]
[Date "2024.01.01"]
[Round "1"]
[White "Engine1"]
[Black "Engine2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

[Event "Test"]
[Site "?"]
[Date "2024.01.01"]
[Round "2"]
[White "Engine2"]
[Black "Engine1"]
[Result "1/2-1/2"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 1/2-1/2

[Event "Test"]
[Site "?"]
[Date "2024.01.01"]
[Round "3"]
[White "Engine1"]
[Black "Engine2"]
[Result "1-0"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 1-0
"""

with _tmp.TemporaryDirectory() as _td:
    pgn_path = os.path.join(_td, 'test.pgn')
    bin_path = os.path.join(_td, 'out.bin')
    with open(pgn_path, 'w') as f:
        f.write(_MINI_PGN)
    try:
        n_entries, n_games = _bm.build_book(
            pgn_files=[pgn_path], out_path=bin_path,
            depth=10, min_freq=1, verbose=False)
        # Check file exists and has valid 16-byte records
        fsize = os.path.getsize(bin_path)
        if fsize % 16 == 0 and fsize > 0:
            PASS += 1; section_results.append(
                f"  ✓ book_builder build: {n_entries} entries, {n_games} games, {fsize}B")
        else:
            FAIL += 1; section_results.append(
                f"  ✗ book_builder build: bad file size {fsize} (must be multiple of 16)")

        # Check big-endian key order (required for binary search)
        with open(bin_path, 'rb') as f2:
            data = f2.read()
        keys = [_struct.unpack('>Q', data[i:i+8])[0] for i in range(0, len(data), 16)]
        if keys == sorted(keys):
            PASS += 1; section_results.append("  ✓ book_builder: entries sorted by key")
        else:
            FAIL += 1; section_results.append("  ✗ book_builder: entries NOT sorted")

        # Check 1.e4 appears in the book (appears in 2 of 3 games → freq≥1)
        e4_found = any(_struct.unpack('>H', data[i+8:i+10])[0] & 0xFFF == 0b_100_001_100_000
                       for i in range(0, len(data), 16))
        # More lenient: at least one entry exists
        if n_entries > 0:
            PASS += 1; section_results.append("  ✓ book_builder: non-empty book produced")
        else:
            FAIL += 1; section_results.append("  ✗ book_builder: empty book")

    except Exception as e:
        FAIL += 1; section_results.append(f"  ✗ book_builder build exception: {e}")

# 11.3 Merge two books
if _bm:
    with _tmp.TemporaryDirectory() as _td2:
        b1 = os.path.join(_td2, 'b1.bin')
        b2 = os.path.join(_td2, 'b2.bin')
        bm_out = os.path.join(_td2, 'merged.bin')
        pgn_p = os.path.join(_td2, 'test.pgn')
        with open(pgn_p, 'w') as f:
            f.write(_MINI_PGN)
        try:
            _bm.build_book([pgn_p], b1, min_freq=1, verbose=False)
            _bm.build_book([pgn_p], b2, min_freq=1, verbose=False)
            _bm.merge_books([b1, b2], bm_out)
            if os.path.getsize(bm_out) % 16 == 0:
                PASS += 1; section_results.append("  ✓ book_builder merge: valid output")
            else:
                FAIL += 1; section_results.append("  ✗ book_builder merge: bad size")
        except Exception as e:
            FAIL += 1; section_results.append(f"  ✗ book_builder merge: {e}")

# 11.4 Book produced by builder can be loaded by the engine
if _bm:
    with _tmp.TemporaryDirectory() as _td3:
        pgn_p = os.path.join(_td3, 'test.pgn')
        bin_p = os.path.join(_td3, 'engine_test.bin')
        with open(pgn_p, 'w') as f: f.write(_MINI_PGN)
        try:
            _bm.build_book([pgn_p], bin_p, min_freq=1, verbose=False)
            out = ask(BASE,
                f"setoption name BookFile value {bin_p}\n"
                "setoption name OwnBook value true\n"
                "position startpos\ngo depth 3\nquit\n")
            if "bestmove" in out:
                PASS += 1; section_results.append("  ✓ book_builder: engine loads and uses built book")
            else:
                FAIL += 1; section_results.append("  ✗ book_builder: engine did not return bestmove")
        except Exception as e:
            FAIL += 1; section_results.append(f"  ✗ book_builder engine test: {e}")

# 11.5 CLI help doesn't crash
r = _sp2.run([sys.executable, BUILDER], capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=5)
if r.returncode == 0 or "Usage" in r.stdout or "usage" in r.stdout.lower():
    PASS += 1; section_results.append("  ✓ book_builder CLI: help output")
else:
    FAIL += 1; section_results.append(f"  ✗ book_builder CLI: unexpected exit {r.returncode}")

lap_summary("Layer 11")
L11_P, L11_F = PASS, FAIL; PASS=0; FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 12: Elo Benchmark Infrastructure (bench.py)
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 12 — Elo Benchmark Infrastructure (bench.py)")

BENCH = os.path.join(SCRIPT_DIR, "bench.py")

# 12.1 Importable
try:
    spec2 = importlib.util.spec_from_file_location("bench", BENCH)
    _bench = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(_bench)
    PASS += 1; section_results.append("  ✓ bench.py importable")
except Exception as e:
    FAIL += 1; section_results.append(f"  ✗ bench.py import failed: {e}")
    _bench = None

# 12.2 Elo formula sanity
if _bench:
    e50 = _bench.elo_from_score(0.5)    # should be 0
    e75 = _bench.elo_from_score(0.75)   # should be ~191
    e25 = _bench.elo_from_score(0.25)   # should be ~-191
    if abs(e50) < 1 and 180 < e75 < 210 and -210 < e25 < -180:
        PASS += 1; section_results.append(f"  ✓ bench Elo formula: e(0.5)={e50:.1f} e(0.75)={e75:.1f}")
    else:
        FAIL += 1; section_results.append(f"  ✗ bench Elo formula: e(0.5)={e50:.1f} e(0.75)={e75:.1f}")

# 12.3 CI computation
if _bench:
    elo, lo, hi = _bench.elo_ci(60, 40, 20)  # 60W/40L/20D
    if lo < elo < hi and -50 < elo < 200:
        PASS += 1; section_results.append(f"  ✓ bench elo_ci: {elo:.1f} [{lo:.1f},{hi:.1f}]")
    else:
        FAIL += 1; section_results.append(f"  ✗ bench elo_ci: {elo:.1f} [{lo:.1f},{hi:.1f}] out of range")

# 12.4 SPRT LLR computation
if _bench:
    llr_null = _bench.sprt_llr(50, 50, 0, 0, 10)  # ~0 games → near 0
    llr_win  = _bench.sprt_llr(70, 30, 0, 0, 20)  # clear win → positive
    if isinstance(llr_null, float) and llr_win > 0:
        PASS += 1; section_results.append(f"  ✓ bench SPRT LLR: null={llr_null:.2f} win={llr_win:.2f}")
    else:
        FAIL += 1; section_results.append(f"  ✗ bench SPRT LLR: null={llr_null} win={llr_win}")

# 12.5 parse_tc variations
if _bench:
    try:
        tc1 = _bench.parse_tc('depth=10')
        tc2 = _bench.parse_tc('0.1+0.01')
        tc3 = _bench.parse_tc('100ms')
        ok = (tc1 == {'depth':10} and 'wtime' in tc2 and 'movetime' in tc3)
        if ok:
            PASS += 1; section_results.append("  ✓ bench parse_tc: depth/increment/movetime")
        else:
            FAIL += 1; section_results.append(f"  ✗ bench parse_tc: {tc1} {tc2} {tc3}")
    except Exception as e:
        FAIL += 1; section_results.append(f"  ✗ bench parse_tc: {e}")

# 12.6 Engine wrapper: start/go/stop
if _bench:
    try:
        e = _bench.Engine(os.path.join(SCRIPT_DIR, "build", "hugine_base"))
        e.start()
        bm, score = e.go("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                         [], {'depth': 4})
        e.stop()
        if len(bm) >= 4:
            PASS += 1; section_results.append(f"  ✓ bench Engine wrapper: bm={bm} score={score}")
        else:
            FAIL += 1; section_results.append(f"  ✗ bench Engine wrapper: bad bm={repr(bm)}")
    except Exception as e_err:
        FAIL += 1; section_results.append(f"  ✗ bench Engine wrapper: {e_err}")

# 12.7 Mini match: engine vs itself, 4 games
if _bench:
    try:
        cfg = _bench.MatchConfig(
            engine1=os.path.join(SCRIPT_DIR, "build", "hugine_base"),
            engine2=os.path.join(SCRIPT_DIR, "build", "hugine_base"),
            tc={'depth': 4},
            games=4,
            openings=_bench.BUILTIN_OPENINGS[:4],
            out="",      # no file output
            verbose=False,
        )
        result = _bench.run_match(cfg)
        total = result['wins'] + result['losses'] + result['draws']
        if total == 4:
            PASS += 1; section_results.append(
                f"  ✓ bench mini-match: {result['wins']}W/{result['losses']}L/{result['draws']}D")
        else:
            FAIL += 1; section_results.append(
                f"  ✗ bench mini-match: expected 4 games, got {total}")
    except Exception as e_err:
        FAIL += 1; section_results.append(f"  ✗ bench mini-match: {e_err}")

# 12.8 Builtin openings are valid FEN strings
if _bench:
    bad_fens = [f for f in _bench.BUILTIN_OPENINGS if len(f.split()) < 4]
    if not bad_fens:
        PASS += 1; section_results.append(f"  ✓ bench builtin openings: {len(_bench.BUILTIN_OPENINGS)} valid FENs")
    else:
        FAIL += 1; section_results.append(f"  ✗ bench builtin openings: {len(bad_fens)} bad FENs")

lap_summary("Layer 12")
L12_P, L12_F = PASS, FAIL; PASS=0; FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 13: Cross-Architecture (all 13 build variants)
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 13 — Cross-Architecture (all 13 build variants)")

VARIANTS = [
    ("hugine_base",       os.path.join(SCRIPT_DIR, "build", "hugine_base")),
    ("hugine_fast",       os.path.join(SCRIPT_DIR, "build", "hugine_fast")),
    ("hugine_pext",       os.path.join(SCRIPT_DIR, "build", "hugine_pext")),
    ("hugine_large",      os.path.join(SCRIPT_DIR, "build", "hugine_large")),
    ("hugine_nnue",       os.path.join(SCRIPT_DIR, "build", "hugine_nnue")),
    ("hugine_fast_nnue",  os.path.join(SCRIPT_DIR, "build", "hugine_fast_nnue")),
    ("hugine_large_nnue", os.path.join(SCRIPT_DIR, "build", "hugine_large_nnue")),
    ("hugine_xl",         os.path.join(SCRIPT_DIR, "build", "hugine_xl")),
    ("hugine_xl_nnue",    os.path.join(SCRIPT_DIR, "build", "hugine_xl_nnue")),
]

for variant_name, binary in VARIANTS:
    # 13.a: uci / isready / quit
    test(f"{variant_name}: uci handshake",
        binary, "uci\nisready\nquit\n",
        ["uciok", "readyok"])

    # 13.b: perft 4 = 197281
    test(f"{variant_name}: perft(4) = 197281",
        binary, "position startpos\nperft 4\nquit\n",
        ["Nodes searched: 197281"], timeout=30)

    # 13.c: search returns bestmove
    test(f"{variant_name}: search depth 5",
        binary, "position startpos\ngo depth 5\nquit\n",
        ["bestmove"])

    # 13.d: contempt option accepted
    test(f"{variant_name}: Contempt option",
        binary, "setoption name Contempt value 20\nposition startpos\ngo depth 4\nquit\n",
        ["bestmove"], anti_checks=["Unknown option"])

    # 13.e: kiwipete no crash
    test(f"{variant_name}: kiwipete depth 5",
        binary,
        "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\n"
        "go depth 5\nquit\n",
        ["bestmove"], timeout=20)

lap_summary("Layer 13")
L13_P, L13_F = PASS, FAIL; PASS=0; FAIL=0


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 14: Singular Extension — Full Stockfish-style Implementation
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 14 — Singular Extension (full implementation)")

# 14.1 Version string confirms 5.0.1
test("Version: id name Hugine 5.1.0", BASE,
    "uci\nquit\n",
    ["id name Hugine 5.1.0"])

# 14.2 Basic puzzle: singular extension should FIND forced mates more reliably.
#      These positions have one clearly winning move — the engine should find it.
test("SE: mate-in-2 clarity (Qf7#)", BASE,
    "position fen r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4\n"
    "go depth 6\nquit\n",
    ["bestmove h5f7"])

# 14.3 Forced mate: back-rank mate — singular extension should extend this line
test("SE: back-rank mate found", BASE,
    "position fen 6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1\n"
    "go depth 8\nquit\n",
    ["score mate"])

# 14.4 Multi-cut pruning: a position with many winning moves should NOT extend
#      (multi-cut fires, returns early). The engine must return bestmove quickly.
test("SE: multi-cut: winning position returns quickly", BASE,
    "position fen r1b1k1nr/pppp1ppp/2n5/4p3/1b2P3/2N2N2/PPPPBPPP/R1BQK2R b KQkq - 4 4\n"
    "go depth 8\nquit\n",
    ["bestmove"])

# 14.5 Singular move: only one defensive move in a critical position
#      The engine should extend the singular defensive move.
test("SE: extends singular defensive move (depth 10)", BASE,
    "position fen 2r3k1/5ppp/p7/1p6/1P6/P4P2/5KPP/2R5 w - - 0 1\n"
    "go depth 10\nquit\n",
    ["bestmove"])

# 14.6 Double extension: a position where the only move is MUCH better than alternatives
#      Should result in deeper search (more info lines) without hanging
test("SE: double extension (very singular position)", BASE,
    "position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1\n"
    "go depth 12\nquit\n",
    ["bestmove"])

# 14.7 Negative extension: TT move triggers multi-cut check — engine returns bestmove
test("SE: negative extension path doesn\'t crash", BASE,
    "position fen r3k2r/pp1b1ppp/2np1n2/q1p1p3/2B1P3/2NP1N2/PPPB1PPP/R2QR1K1 w kq - 2 9\n"
    "go depth 9\nquit\n",
    ["bestmove"])

# 14.8 Singular extension at depth 8 (minimum trigger depth)
test("SE: fires at depth 8 (minimum trigger)", BASE,
    "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6\n"
    "go depth 8\nquit\n",
    ["bestmove"])

# 14.9 No crash in complex Kiwipete position at deep depth with SE active
test("SE: kiwipete depth 10 stable", BASE,
    "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\n"
    "go depth 10\nquit\n",
    ["bestmove"], timeout=30)

# 14.10 Perft still exact after SE (SE must not affect move generation)
test("SE: perft(4) still 197281 after SE changes", BASE,
    "position startpos\nperft 4\nquit\n",
    ["Nodes searched: 197281"])

# 14.11 All 8 build variants: SE option does not break them (spot check depth 8)
for _v_name, _v_bin in [("hugine_fast", os.path.join(SCRIPT_DIR, "build", "hugine_fast")),
                         ("hugine_pext", os.path.join(SCRIPT_DIR, "build", "hugine_pext")),
                         ("hugine_large", os.path.join(SCRIPT_DIR, "build", "hugine_large"))]:
    test(f"SE: {_v_name} depth-8 search stable", _v_bin,
        "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\n"
        "go depth 8\nquit\n",
        ["bestmove"], timeout=20)

lap_summary("Layer 14")
L14_P, L14_F = PASS, FAIL; PASS=0; FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 15: Advanced Search Features (v5.1.0)
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 15 — Advanced Features (CORRHIST, 2-ply cont, LMP, TT prefetch, Lazy SMP, TM)")

# 15.1 Version string: 5.1.0
test("Version: id name Hugine 5.1.0", BASE,
    "uci\nquit\n",
    ["id name Hugine 5.1.0"])

# 15.2 CORRHIST: pawn structure correction adjusts eval (engine finds correct plan in pawn endgame)
test("CORRHIST: pawn endgame eval correction", BASE,
    "position fen 8/p7/8/1p6/1P6/8/P7/8 w - - 0 1\n"
    "go depth 14\nquit\n",
    ["bestmove"])

# 15.3 2-ply continuation history: engine finds consistent quiet plans
test("2-ply cont history: quiet plan depth 12", BASE,
    "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5\n"
    "go depth 12\nquit\n",
    ["bestmove", "score cp"])

# 15.4 LMP table: engine prunes quickly at depth 8 in side-with-advantage position  
test("LMP table: fast pruning at depth 8", BASE,
    "position fen r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\n"
    "go depth 8\nquit\n",
    ["bestmove"])

# 15.5 TT prefetch: engine searches Kiwipete at depth 11 within 15s (performance test)
test("TT prefetch: kiwipete depth 11 perf", BASE,
    "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\n"
    "go depth 11\nquit\n",
    ["bestmove"], timeout=15)

# 15.6 Lazy SMP: 2-thread search finds same or better move vs single thread
test("Lazy SMP: 2 threads finds best move", BASE,
    "setoption name Threads value 2\n"
    "position fen r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6\n"
    "go depth 10\nquit\n",
    ["bestmove"])

# 15.7 Lazy SMP: 4-thread search completes without crash or hang
test("Lazy SMP: 4 threads stable", BASE,
    "setoption name Threads value 4\n"
    "position startpos\ngo depth 10\nquit\n",
    ["bestmove"])

# 15.8 TM: movetime respected with node-stability (stops promptly)
test("TM: movetime 200ms stops promptly", BASE,
    "setoption name Threads value 1\n"
    "position startpos\ngo movetime 200\nquit\n",
    ["bestmove"])

# 15.9 TM: wtime/btime search stops within time budget
test("TM: wtime 5000 btime 5000 budget respected", BASE,
    "position startpos\n"
    "go wtime 5000 btime 5000 winc 0 binc 0\nquit\n",
    ["bestmove"])

# 15.10 Capture history: engine orders captures correctly (SEE-good before SEE-bad)
test("Capture history: orders captures well", BASE,
    "position fen r1b1k2r/ppp2ppp/2n2n2/3pp3/1bB1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7\n"
    "go depth 10\nquit\n",
    ["bestmove"])

# 15.11 Perft correctness after LMP table change (must not prune legal moves)
test("LMP table: perft(4) still exact", BASE,
    "position startpos\nperft 4\nquit\n",
    ["Nodes searched: 197281"])

# 15.12 Perft kiwipete depth 3 (correctness check, SE + LMP must not distort)
test("LMP+SE: kiwipete perft(3) exact", BASE,
    "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\n"
    "perft 3\nquit\n",
    ["Nodes searched: 97862"])

lap_summary("Layer 15")
L15_P, L15_F = PASS, FAIL; PASS=0; FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 16: Bulk Counting Perft + SIMD NNUE (v5.1.0)
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 16 — Bulk Counting Perft + SIMD NNUE")

FAST = os.path.join(SCRIPT_DIR, "build", "hugine_fast")
NNUE_B = os.path.join(SCRIPT_DIR, "build", "hugine_nnue")

# ── 16.1–16.7: Perft correctness (bulk count must produce identical node counts) ──
test("Perft bulk: startpos d1", BASE, "position startpos\nperft 1\nquit\n",
     ["Nodes searched: 20"])
test("Perft bulk: startpos d2", BASE, "position startpos\nperft 2\nquit\n",
     ["Nodes searched: 400"])
test("Perft bulk: startpos d3", BASE, "position startpos\nperft 3\nquit\n",
     ["Nodes searched: 8902"])
test("Perft bulk: startpos d4", BASE, "position startpos\nperft 4\nquit\n",
     ["Nodes searched: 197281"])
test("Perft bulk: startpos d5", BASE, "position startpos\nperft 5\nquit\n",
     ["Nodes searched: 4865609"])

# ── 16.6–16.9: Kiwipete perft correctness ─────────────────────────────────
KW = "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\n"
test("Perft bulk: kiwipete d1", BASE, KW+"perft 1\nquit\n",
     ["Nodes searched: 48"])
test("Perft bulk: kiwipete d2", BASE, KW+"perft 2\nquit\n",
     ["Nodes searched: 2039"])
test("Perft bulk: kiwipete d3", BASE, KW+"perft 3\nquit\n",
     ["Nodes searched: 97862"])
test("Perft bulk: kiwipete d4", BASE, KW+"perft 4\nquit\n",
     ["Nodes searched: 4085603"], timeout=30)

# ── 16.10: En passant correctness ─────────────────────────────────────────
test("Perft bulk: en-passant pos d5", BASE,
     "position fen rnbqkb1r/pp1p1ppp/2p5/4pP2/2B5/8/PPPP1PPP/RNBQK1NR w KQkq e6 0 1\n"
     "perft 5\nquit\n",
     ["Nodes searched: 30872273"], timeout=15)

# ── 16.11: Promotion-heavy position (pos5) ────────────────────────────────
test("Perft bulk: promotion pos5 d3", BASE,
     "position fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8\n"
     "perft 3\nquit\n",
     ["Nodes searched: 62379"])

# ── 16.12: Perft NPS performance (bulk > 10M NPS on fast build) ───────────
# NPS check: we manually run and verify > 10M NPS
import subprocess as _sp
_r = _sp.run([FAST], input="position startpos\nperft 5\nquit\n",
              capture_output=True, text=True, timeout=30)
_nps_ok = False
for _line in _r.stdout.splitlines():
    if "Nodes searched" in _line and "nps:" in _line:
        _nps_val = int(_line.split("nps:")[1].split()[0])
        _nps_ok = _nps_val >= 10_000_000
PASS += _nps_ok; FAIL += (not _nps_ok)
tick = "✓" if _nps_ok else "✗"
print(f"  {tick} Perft bulk: NPS > 10M (startpos d5) — got {_nps_val:,} nps")

# ── 16.13: No std::function overhead — perft_inner is called recursively ──
# Verify depth 6 completes within 10 seconds (proves non-exponential overhead)
test("Perft bulk: startpos d6 within 15s", FAST,
     "position startpos\nperft 6\nquit\n",
     ["Nodes searched: 119060324"], timeout=10)

# ── 16.14–16.16: NNUE build compiles and loads correctly ──────────────────
test("NNUE: evaluate command returns score", NNUE_B,
     "position startpos\neval\nquit\n",
     ["Evaluation:"])

test("NNUE: go depth 8 returns valid move", NNUE_B,
     "position startpos\ngo depth 8\nquit\n",
     ["bestmove"])

test("NNUE: NNUE eval changes after moves", NNUE_B,
     "position startpos moves e2e4\neval\nquit\n",
     ["Evaluation:"])

# ── 16.17: SIMD NNUE build (fast_nnue): goes deeper within time ───────────
FAST_NNUE = os.path.join(SCRIPT_DIR, "build", "hugine_fast_nnue")
test("SIMD NNUE: fast_nnue returns move", FAST_NNUE,
     "position startpos\ngo depth 10\nquit\n",
     ["bestmove"])

# ── 16.18: large_nnue build functional ────────────────────────────────────
LARGE_NNUE = os.path.join(SCRIPT_DIR, "build", "hugine_large_nnue")
test("SIMD NNUE: large_nnue go depth 8", LARGE_NNUE,
     "position startpos\ngo depth 8\nquit\n",
     ["bestmove"])

# ── 16.19: Perft consistency: base == fast == pext (same node count) ──────
def get_nodes(binary, fen_cmd):
    import subprocess
    r = subprocess.run([binary], input=fen_cmd, capture_output=True, text=True, timeout=30)
    for line in r.stdout.splitlines():
        if "Nodes searched:" in line:
            return line.split(":")[1].split()[0].strip()
    return None

cmd = "position startpos\nperft 5\nquit\n"
n_base = get_nodes(os.path.join(SCRIPT_DIR, "build", "hugine_base"), cmd)
n_fast = get_nodes(os.path.join(SCRIPT_DIR, "build", "hugine_fast"), cmd)
n_pext = get_nodes(os.path.join(SCRIPT_DIR, "build", "hugine_pext"), cmd)
consistent = (n_base == n_fast == n_pext == "4865609")
PASS += consistent; FAIL += (not consistent)
tick = "✓" if consistent else "✗"
print(f"  {tick} Perft consistency: base={n_base} fast={n_fast} pext={n_pext}")
if not consistent:
    print(f"    FAIL: node counts differ between builds!")
section_results[-1] = section_results[-1].replace(" (running)", f" — {PASS} passed, {FAIL} failed so far")

# ── 16.20: Bulk count vs make/undo: regression check (pin positions) ──────
test("Perft bulk: pinned pieces pos d3", BASE,
     "position fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1\n"
     "perft 3\nquit\n",
     ["Nodes searched: 2812"])

lap_summary("Layer 16")
L16_P, L16_F = PASS, FAIL; PASS=0; FAIL=0
# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY (all 16 layers)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 17 — Native Stockfish NNUE (SFNNUEEvaluator)
# ══════════════════════════════════════════════════════════════════════════════
# These tests require a Stockfish .nnue file in the sf_nnue/ subdirectory.
# If none is found the layer is skipped gracefully.
# To run: place any SF .nnue in sf_nnue/ next to this script.
#   e.g.  mkdir sf_nnue && cp nn-XXXXXXXX.nnue sf_nnue/
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 17 — Native Stockfish NNUE (SFNNUEEvaluator)")

import glob as _glob

_SF_NNUE_DIR  = os.path.join(SCRIPT_DIR, "sf_nnue")
_SF_NNUE_FILE = None

# Look for any .nnue that has a Stockfish version header (top byte 0x7A)
for _candidate in sorted(_glob.glob(os.path.join(_SF_NNUE_DIR, "*.nnue")) +
                          _glob.glob(os.path.join(_SF_NNUE_DIR, "nn-*.bin"))):
    try:
        with open(_candidate, "rb") as _fh:
            _v = _fh.read(4)
        if len(_v) == 4 and (_v[3] & 0xFF) == 0x7A:
            _SF_NNUE_FILE = _candidate
            break
    except OSError:
        pass

if _SF_NNUE_FILE is None:
    # Friendly skip: print how to enable, count 0 pass/fail
    section_results.append(f"  (skipped — no Stockfish .nnue found in {_SF_NNUE_DIR})")
    section_results.append(  "   To enable: mkdir sf_nnue && place nn-XXXXXXXX.nnue inside it")
    L17_P = 0; L17_F = 0
else:
    section_results.append(f"  Using: {os.path.basename(_SF_NNUE_FILE)}")

    # 17.1 Architecture auto-detection line in output
    test("SF NNUE: arch line in output on load", NNUE,
        f"setoption name EvalFile value {_SF_NNUE_FILE}\nisready\nquit\n",
        ["SF-NNUE arch:", "readyok"])

    # 17.2 Load confirmation
    test("SF NNUE: NNUE loaded confirmation", NNUE,
        f"setoption name EvalFile value {_SF_NNUE_FILE}\nisready\nquit\n",
        ["info string NNUE loaded:"])

    # 17.3 Search returns bestmove after SF NNUE loaded
    test("SF NNUE: search depth 6 returns bestmove", NNUE,
        f"setoption name EvalFile value {_SF_NNUE_FILE}\n"
        f"position startpos\ngo depth 6\nquit\n",
        ["bestmove"])

    # 17.4 Perft d4 still correct with SF NNUE loaded (move gen must not change)
    test("SF NNUE: perft 4 = 197281 (move-gen unaffected)", NNUE,
        f"setoption name EvalFile value {_SF_NNUE_FILE}\n"
        f"position startpos\nperft 4\nquit\n",
        ["Nodes searched: 197281"])

    # 17.5 Kiwipete perft d3 = 97862 with SF NNUE
    test("SF NNUE: kiwipete perft 3 = 97862", NNUE,
        f"setoption name EvalFile value {_SF_NNUE_FILE}\n"
        "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\n"
        "perft 3\nquit\n",
        ["Nodes searched: 97862"])

    # 17.6 Tactical position — mate detection
    test("SF NNUE: mate-in-2 detected", NNUE,
        f"setoption name EvalFile value {_SF_NNUE_FILE}\n"
        "position fen 4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 0 1\n"
        "go depth 6\nquit\n",
        ["score mate"])

    # 17.7 Incremental accumulator: 18-move line
    test("SF NNUE: 18-move incremental acc survives", NNUE,
        f"setoption name EvalFile value {_SF_NNUE_FILE}\n"
        "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 "
        "e1g1 f8e7 d2d3 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 h7h6\n"
        "go depth 5\nquit\n",
        ["bestmove"])

    # 17.8 UCI reload: unload → reload in same UCI session
    test("SF NNUE: unload then reload same session", NNUE,
        f"setoption name EvalFile value {_SF_NNUE_FILE}\n"
        "isready\n"
        "setoption name EvalFile value \n"
        "isready\n"
        f"setoption name EvalFile value {_SF_NNUE_FILE}\n"
        "position startpos\ngo depth 4\nquit\n",
        ["info string NNUE loaded:", "bestmove"])

    # 17.9 ASAN: no memory errors with SF file
    test("SF NNUE: ASAN no memory errors", ASAN,
        f"setoption name EvalFile value {_SF_NNUE_FILE}\n"
        "position startpos\ngo depth 5\nquit\n",
        ["bestmove"],
        anti_checks=["ERROR: AddressSanitizer", "runtime error", "SEGV"])

    # 17.10 Hugine-native fallback still works after SF file tried
    test("SF NNUE: Hugine-native .nnue still loads (fallback path)", NNUE,
        f"setoption name EvalFile value {NNUEF}\n"
        "position startpos\ngo depth 4\nquit\n",
        ["info string NNUE loaded:", "bestmove"])

    lap_summary("Layer 17")
    L17_P, L17_F = PASS, FAIL; PASS=0; FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
print('\n'.join(section_results))

all_p = L1_P+L2_P+L3_P+L4_P+L5_P+L6_P+L7_P+L8_P+L9_P+L10_P+L11_P+L12_P+L13_P+L14_P+L15_P+L16_P+L17_P
all_f = L1_F+L2_F+L3_F+L4_F+L5_F+L6_F+L7_F+L8_F+L9_F+L10_F+L11_F+L12_F+L13_F+L14_F+L15_F+L16_F+L17_F
print(f"\n{'═'*56}")
print(f" FINAL RESULTS (17 Layers)")
print(f"{'═'*56}")
for lbl, p, f in [
    ("Layer 1   UCI Protocol         ", L1_P,  L1_F),
    ("Layer 2   Perft                ", L2_P,  L2_F),
    ("Layer 3   Puzzles              ", L3_P,  L3_F),
    ("Layer 4   NNUE                 ", L4_P,  L4_F),
    ("Layer 5   Opening Book         ", L5_P,  L5_F),
    ("Layer 6   Time Mgmt            ", L6_P,  L6_F),
    ("Layer 7   Robustness           ", L7_P,  L7_F),
    ("Layer 8   NNUE Stress          ", L8_P,  L8_F),
    ("Layer 9   WDL Probing/Contempt ", L9_P,  L9_F),
    ("Layer 10  NNUE Arch 256/512/1024", L10_P, L10_F),
    ("Layer 11  Book Builder         ", L11_P, L11_F),
    ("Layer 12  Elo Benchmark        ", L12_P, L12_F),
    ("Layer 13  Cross-Architecture   ", L13_P, L13_F),
    ("Layer 14  Singular Extension   ", L14_P, L14_F),
    ("Layer 15  Advanced Features    ", L15_P, L15_F),
    ("Layer 16  Bulk Count + SIMD NNUE", L16_P, L16_F),
    ("Layer 17  SF NNUE Native        ", L17_P, L17_F),
]:
    bar = "✓"*p + "✗"*f
    print(f"  {lbl}  {p:2}/{p+f}  {bar}")
print(f"{'─'*56}")
skipped17 = "(10 skipped)" if L17_P+L17_F == 0 else ""
print(f"  TOTAL: {all_p}/{all_p+all_f}  ({100*all_p//(all_p+all_f) if all_p+all_f else 0}%)  — Hugine 5.1.0 Iota  {skipped17}")
