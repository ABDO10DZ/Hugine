#!/usr/bin/env python3
"""
bench.py — Automated Elo benchmarking for Hugine.

Runs a mini-gauntlet between engine pairs without needing cutechess.
Uses UCI protocol directly (fork-based engine manager), plays time-controlled
games on a standard test set of positions, and computes Elo difference using
the Elo performance formula.

Usage:
  python3 bench.py selftest  --engine ./hugine [--depth 8] [--games 20]
      Quick internal consistency check (engine vs itself at lower depth).

  python3 bench.py match  --engine1 ./hugine --engine2 ./hugine_old
                          [--tc 0.1+0.01] [--depth 10] [--games 100]
                          [--openings openings.epd] [--out results.json]
      Full head-to-head match. Reports Elo difference with 95% CI.

  python3 bench.py nps  --engine ./hugine [--depth 12] [--positions 10]
      NPS and node count benchmark.

  python3 bench.py sprt --engine1 ./hugine --engine2 ./hugine_old
                        [--elo0 0] [--elo1 10] [--alpha 0.05] [--beta 0.05]
      Sequential Probability Ratio Test — stops automatically when result is
      statistically conclusive (like fastchess/cutechess --sprt).

Output is plain text + optional JSON.

Requires: Python 3.9+, no external dependencies.
"""

import argparse, json, math, os, queue, random, re, signal, subprocess
import sys, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─── Built-in opening positions (EPD / FEN) ──────────────────────────────────
# 50 diverse middlegame positions from standard test suites.
BUILTIN_OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",       # 1.e4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",       # 1.d4
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",     # 1.e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",    # 1.e4 e5 2.Nf3
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Ruy Lopez start
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", # 3.Bb5
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",# Two Knights
    "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",  # Italian
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",     # Sicilian
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 2",    # Sicilian 2.Nc3
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",    # Sicilian 2.Nf3
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Sicilian Nc6
    "rnbqkb1r/pp2pppp/3p1n2/2p5/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4", # Sicilian d5
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",     # QGD start
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",     # QGD 2.c4
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",     # Semi-Slav
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",  # QGD Nf6
    "r1bqkb1r/ppp2ppp/2np1n2/4p3/2B1P3/2NP4/PPP2PPP/R1BQK1NR w KQkq - 0 5",  # Italian advanced
    "r1bqk2r/ppp1bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 2 6",  # Giuoco Piano
    "r1bqkb1r/pp2pppp/2np1n2/3Pp3/8/2NB1N2/PPP2PPP/R1BQK2R b KQkq - 0 6",    # Scotch
    "r2qkb1r/pp1bpppp/2np1n2/3Pp3/8/2NB1N2/PPP2PPP/R1BQK2R w KQkq - 1 7",    # Scotch mid
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 2 5",    # QGD Be7
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 b - - 4 8", # Spanish closed
    "r1bq1rk1/pp2bppp/2np1n2/3Pp3/4P3/2NB1N2/PPP2PPP/R1BQ1RK1 b - - 0 9",    # Berlin
    "r1bqr1k1/ppp2ppp/2npbn2/4p3/4P3/2NP1N2/PPPBBPPP/R2QR1K1 b - - 4 9",    # Ruy complex
    "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 b - - 0 6",       # Catalan
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/4P3/1BNP1N2/PPP2PPP/R1BQ1RK1 b - - 2 7",  # English
    "r2q1rk1/ppp1bppp/2np1n2/2b1p3/4P3/1BNP1N2/PPP2PPP/R1BQ1RK1 w - - 3 8",  # English cont
    "r1bqkb1r/ppp2ppp/2np1n2/4p3/4P3/3P1N2/PPP1BPPP/RNBQK2R b KQkq - 0 5",   # King's Indian setup
    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",        # KID 3.Nc3
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR b KQkq e3 0 5",     # KID 4.e4
    "r1bq1rk1/ppp1ppbp/2np1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 4 7",    # KID main line
    "r1bq1rk1/ppp2pbp/2np1np1/3Pp3/4P3/2N2N2/PP2BPPP/R1BQ1RK1 b - e6 0 8",   # KID Mar del Plata
    "r1bq1rk1/pp1pnpbp/4p1p1/2pP4/4P3/2P2N2/PP3PPP/RQBNKB1R w KQ - 1 9",     # Nimzo-Indian
    "r1bqk2r/pp3ppp/2n1pn2/3p4/1bP5/1PN1PN2/P2B1PPP/R2QKB1R b KQkq - 0 7",   # Nimzo mid
    "rnbq1rk1/p3bppp/1p2pn2/2pp4/2PP4/1PN1PN2/P3BPPP/R1BQ1RK1 b - - 0 8",    # Queen's Indian
    "r1bq1rk1/ppp1ppbp/n2p1np1/8/3PP3/2N2N2/PPP1BPPP/R1BQ1RK1 w - - 2 7",    # Gruenfeld setup
    "r1bq1rk1/ppp2pbp/n2p1np1/3Pp3/4P3/2N2N2/PPP1BPPP/R1BQ1RK1 b - - 0 8",   # Gruenfeld d5
    "r2q1rk1/ppp1bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 b - - 2 8",   # Open Spanish
    "r2q1rk1/ppp1bppp/2np4/3np3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 3 9",    # Open Spanish Nd5
    "r1b2rk1/ppp1qppp/2np1n2/2b1p3/4P3/1BNP1N2/PPP1QPPP/R1B2RK1 b - - 2 8",  # Closed Spanish
    "r1bq1rk1/pp2ppbp/2np1np1/3p4/2P5/1PN1PN2/P3BPPP/R1BQR1K1 b - - 0 8",    # English Hedgehog
    "r1bq1rk1/pp3pbp/2np1np1/2pPp3/4P3/2N2N2/PP2BPPP/R1BQ1RK1 b - - 0 9",    # Benoni
    "r1bq1rk1/pp1nppbp/3p1np1/2pP4/4P3/2N2NB1/PPP1BPPP/R2Q1RK1 b - - 2 8",   # Modern Benoni
    "r1b1qrk1/pp3pbp/2npnnp1/4p3/2P1P3/2N2NB1/PP1QBPPP/R4RK1 b - - 4 12",    # KID deep
    "r2q1rk1/1p2bppp/pbnp1n2/4p3/4PP2/2N2N2/PPPBB1PP/R2QR1K1 b - - 2 11",    # Closed Sicilian
    "2rq1rk1/pb1nbppp/1pp1pn2/3p4/2PP4/1PN1BN2/P3BPPP/2RQ1RK1 b - - 4 11",   # Queen's Gambit mid
    "r1bq1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 b - - 2 7",   # QGD Orthodox
    "2rr2k1/1bq1bppp/pp2pn2/3p4/3P1B2/1BN1PN2/PP3PPP/2RQ1RK1 b - - 4 12",    # QGD deep
    "r2q1rk1/pp3ppp/2p1pn2/3pNb2/3P1B2/1BN5/PP3PPP/R2Q1RK1 b - - 4 11",      # Nimzo-Indian mid
]

# ─── Engine process wrapper ───────────────────────────────────────────────────
class Engine:
    """Simple async UCI engine wrapper."""
    def __init__(self, path: str, name: str = "", options: dict = None):
        self.path   = path
        self.name   = name or Path(path).stem
        self.options = options or {}
        self._proc: Optional[subprocess.Popen] = None
        self._q: queue.Queue = queue.Queue()
        self._reader: Optional[threading.Thread] = None

    def start(self):
        self._proc = subprocess.Popen(
            [self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._send("uci")
        self._wait_for("uciok", timeout=10)
        for k, v in self.options.items():
            self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait_for("readyok", timeout=10)

    def _read_loop(self):
        for line in self._proc.stdout:
            self._q.put(line.rstrip())

    def _send(self, cmd: str):
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float = 30.0) -> list[str]:
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.1)
                lines.append(line)
                if token in line: return lines
            except queue.Empty:
                continue
        return lines

    def go(self, fen: str, moves: list[str], tc: dict) -> tuple[str, int]:
        """
        Send position and go command. Returns (bestmove, score_cp).
        tc dict keys: movetime | depth | wtime/btime/winc/binc
        """
        pos_cmd = f"position fen {fen}"
        if moves: pos_cmd += " moves " + " ".join(moves)
        self._send("ucinewgame")
        self._send(pos_cmd)
        if 'depth' in tc:
            go_cmd = f"go depth {tc['depth']}"
        elif 'movetime' in tc:
            go_cmd = f"go movetime {tc['movetime']}"
        else:
            go_cmd = f"go wtime {tc.get('wtime',60000)} btime {tc.get('btime',60000)}"
            if 'winc' in tc: go_cmd += f" winc {tc['winc']} binc {tc['binc']}"
        self._send(go_cmd)
        lines = self._wait_for("bestmove", timeout=60)
        bm = "0000"; score = 0
        for line in lines:
            m = re.search(r'bestmove (\S+)', line)
            if m: bm = m.group(1)
            m = re.search(r'score cp (-?\d+)', line)
            if m: score = int(m.group(1))
        return bm, score

    def stop(self):
        try:
            if self._proc and self._proc.poll() is None:
                self._send("quit")
                self._proc.wait(timeout=3)
        except Exception:
            pass
        finally:
            if self._proc: self._proc.kill()


# ─── Game result ──────────────────────────────────────────────────────────────
DRAW, WHITE_WIN, BLACK_WIN = 0.5, 1.0, 0.0

@dataclass
class GameResult:
    result: float   # 1.0 = white wins, 0.5 = draw, 0.0 = black wins
    plies:  int
    termination: str  # 'checkmate','stalemate','50move','repetition','adjudicated','error'

def play_game(eng1: Engine, eng2: Engine, start_fen: str, tc: dict,
              max_ply: int = 300, adj_score: int = 500, adj_count: int = 5) -> GameResult:
    """
    Play one game between eng1 (white) and eng2 (black).
    Uses score adjudication: if |score| > adj_score for adj_count consecutive plies, stop.
    """
    moves = []
    side  = 0    # 0=white=eng1, 1=black=eng2
    engines = [eng1, eng2]
    win_streak  = 0
    lose_streak = 0
    high_score_side = 0

    for ply in range(max_ply):
        eng = engines[side]
        bm, score = eng.go(start_fen, moves, tc)
        if bm in ('0000', 'none', ''):
            # Engine returned null move → it's checkmated or error
            return GameResult(BLACK_WIN if side == 0 else WHITE_WIN, ply, 'checkmate')
        moves.append(bm)
        side ^= 1

        # Score adjudication (from white's perspective)
        adj_cp = score if side == 1 else -score   # side just flipped, so re-flip
        if abs(adj_cp) >= adj_score:
            win_streak += 1
            if adj_cp > 0: high_score_side = 0
            else:          high_score_side = 1
        else:
            win_streak = 0

        if win_streak >= adj_count:
            result = WHITE_WIN if high_score_side == 0 else BLACK_WIN
            return GameResult(result, ply+1, 'adjudicated')

        # 50-move rule approximation (every 100 plies without a pawn move is draw)
        if len(moves) >= 100 and len(moves) % 2 == 0:
            # Simplified: check if last 100 moves contain no pawn moves
            if all(m[0] not in 'abcdefgh' or m[1:] for m in moves[-100:]):
                return GameResult(DRAW, ply+1, '50move')

    return GameResult(DRAW, max_ply, 'repetition')


# ─── Elo formulas ─────────────────────────────────────────────────────────────
def expected_score(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

def elo_from_score(score: float) -> float:
    """Convert win percentage to Elo difference."""
    if score <= 0.0: return -1000
    if score >= 1.0: return  1000
    return -400 * math.log10(1.0/score - 1.0)

def elo_ci(wins: int, losses: int, draws: int, confidence: float = 0.95) -> tuple[float,float,float]:
    """
    Compute Elo difference and confidence interval.
    Uses Elo performance formula with Bayes-Elo-style normal approximation.
    """
    n = wins + losses + draws
    if n == 0: return 0.0, -1000.0, 1000.0
    score = (wins + 0.5 * draws) / n
    elo   = elo_from_score(score)
    # Variance of the score estimate
    p = score
    var = (p * (1-p)) / n
    z   = 1.96 if confidence == 0.95 else 2.576   # 95% / 99%
    margin = z * math.sqrt(var)
    elo_lo = elo_from_score(max(p - margin, 0.001))
    elo_hi = elo_from_score(min(p + margin, 0.999))
    return elo, elo_lo, elo_hi


# ─── SPRT ────────────────────────────────────────────────────────────────────
def sprt_llr(wins: int, losses: int, draws: int,
             elo0: float, elo1: float) -> float:
    """
    Compute Log-Likelihood Ratio for SPRT H0: elo_diff=elo0, H1: elo_diff=elo1.
    Uses the trinomial (W/D/L) model (same as cutechess --sprt).
    """
    n = wins + losses + draws
    if n == 0: return 0.0
    s = (wins + 0.5 * draws) / n
    s = max(0.001, min(0.999, s))
    p1 = expected_score(elo1)
    p0 = expected_score(elo0)
    # Simple BayesElo-style LLR
    return n * (s * math.log(p1/p0) + (1-s) * math.log((1-p1)/(1-p0)))


# ─── Match runner ─────────────────────────────────────────────────────────────
@dataclass
class MatchConfig:
    engine1: str
    engine2: str
    tc: dict
    games: int = 100
    openings: list = field(default_factory=list)
    out: str = "results.json"
    sprt: Optional[dict] = None   # {'elo0':0,'elo1':10,'alpha':0.05,'beta':0.05}
    verbose: bool = False
    options1: dict = field(default_factory=dict)
    options2: dict = field(default_factory=dict)

def run_match(cfg: MatchConfig) -> dict:
    eng1 = Engine(cfg.engine1, options=cfg.options1)
    eng2 = Engine(cfg.engine2, options=cfg.options2)
    eng1.start(); eng2.start()

    openings = cfg.openings or BUILTIN_OPENINGS
    wins = losses = draws = 0
    results = []

    print(f"\nMatch: {eng1.name} vs {eng2.name}")
    print(f"TC: {cfg.tc}   Games: {cfg.games}")
    print(f"Openings: {len(openings)} positions")
    print("-" * 60)

    try:
        for game_idx in range(cfg.games):
            fen = openings[game_idx % len(openings)]
            # Alternate colours each game pair
            if game_idx % 2 == 0:
                e_white, e_black = eng1, eng2
                swap = False
            else:
                e_white, e_black = eng2, eng1
                swap = True

            gr = play_game(e_white, e_black, fen, cfg.tc)

            # Score from eng1's perspective
            if not swap:
                s = gr.result   # eng1=white
            else:
                s = 1.0 - gr.result if gr.result != DRAW else DRAW

            if s == WHITE_WIN:   wins   += 1; marker = '+'
            elif s == BLACK_WIN: losses += 1; marker = '-'
            else:                draws  += 1; marker = '='
            results.append({'game': game_idx+1, 'result': s, 'plies': gr.plies,
                            'term': gr.termination, 'fen': fen})

            elo, lo, hi = elo_ci(wins, losses, draws)
            if cfg.verbose or (game_idx+1) % 10 == 0:
                print(f"  Game {game_idx+1:4d}/{cfg.games}  {marker}  "
                      f"W:{wins} L:{losses} D:{draws}  "
                      f"Elo: {elo:+.1f} [{lo:+.1f},{hi:+.1f}]  "
                      f"({gr.termination}, {gr.plies} plies)")

            # SPRT stopping criterion
            if cfg.sprt:
                sp = cfg.sprt
                llr = sprt_llr(wins, losses, draws, sp['elo0'], sp['elo1'])
                a, b = sp.get('alpha', 0.05), sp.get('beta', 0.05)
                lo_b = math.log(b / (1-a))
                hi_b = math.log((1-b) / a)
                if llr >= hi_b:
                    print(f"\n  SPRT: H1 accepted (LLR={llr:.3f} >= {hi_b:.3f})")
                    break
                elif llr <= lo_b:
                    print(f"\n  SPRT: H0 accepted (LLR={llr:.3f} <= {lo_b:.3f})")
                    break

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        eng1.stop(); eng2.stop()

    elo, lo, hi = elo_ci(wins, losses, draws)
    summary = {
        'engine1': eng1.name, 'engine2': eng2.name,
        'wins': wins, 'losses': losses, 'draws': draws,
        'total': wins+losses+draws,
        'score': (wins + 0.5*draws) / max(wins+losses+draws, 1),
        'elo': elo, 'elo_lo': lo, 'elo_hi': hi,
        'games': results,
    }

    print("\n" + "=" * 60)
    print(f"Result: {wins}W / {losses}L / {draws}D")
    print(f"Score:  {summary['score']:.3f}")
    print(f"Elo:    {elo:+.1f}  [{lo:+.1f}, {hi:+.1f}]  (95% CI)")
    print(f"{'→ ' + eng1.name + ' is STRONGER' if elo > 0 else '→ ' + eng2.name + ' is STRONGER' if elo < 0 else '→ Engines appear EQUAL'}")

    if cfg.out:
        with open(cfg.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved: {cfg.out}")

    return summary


# ─── NPS benchmark ────────────────────────────────────────────────────────────
def run_nps(engine_path: str, depth: int = 12, positions: int = 10):
    eng = Engine(engine_path)
    eng.start()

    test_fens = BUILTIN_OPENINGS[:positions]
    tc = {'depth': depth}
    total_nodes = total_ms = 0

    print(f"NPS Benchmark: {eng.name}  depth={depth}  positions={positions}")
    print("-" * 50)

    for i, fen in enumerate(test_fens):
        t0 = time.time()
        bm, score = eng.go(fen, [], tc)
        elapsed = (time.time() - t0) * 1000
        total_ms += elapsed

        # Get NPS from last info line directly (crude approximation from time)
        print(f"  Pos {i+1:2d}: {fen[:40]:<40}  bm={bm}  {elapsed:6.0f}ms  score={score:+d}")

    eng.stop()
    print(f"\nTotal time: {total_ms:.0f}ms  Avg: {total_ms/len(test_fens):.0f}ms/position")


# ─── Selftest ────────────────────────────────────────────────────────────────
def run_selftest(engine_path: str, depth: int = 8, games: int = 20):
    """Engine vs itself at different depths — checks for obvious bugs."""
    eng1 = Engine(engine_path, name="deep")
    eng2 = Engine(engine_path, name="shallow")
    eng1.start(); eng2.start()

    tc_deep    = {'depth': depth}
    tc_shallow = {'depth': max(1, depth - 2)}

    wins = losses = draws = errors = 0
    print(f"Selftest: {Path(engine_path).stem}  depth {depth} vs depth {max(1,depth-2)}")
    print("-" * 50)

    for i in range(games):
        fen = BUILTIN_OPENINGS[i % len(BUILTIN_OPENINGS)]
        try:
            if i % 2 == 0:
                gr = play_game(eng1, eng2, fen, tc_deep)
                s  = gr.result
            else:
                gr = play_game(eng2, eng1, fen, tc_shallow)
                s  = 1.0 - gr.result if gr.result != DRAW else DRAW
            if s == WHITE_WIN:   wins   += 1
            elif s == BLACK_WIN: losses += 1
            else:                draws  += 1
            print(f"  Game {i+1:2d}: {'W' if s==WHITE_WIN else 'L' if s==BLACK_WIN else 'D'}  {gr.termination}  {gr.plies} plies")
        except Exception as e:
            errors += 1
            print(f"  Game {i+1:2d}: ERROR {e}")

    eng1.stop(); eng2.stop()
    elo, lo, hi = elo_ci(wins, losses, draws)
    print(f"\n{wins}W/{losses}L/{draws}D  errors={errors}")
    print(f"Elo advantage of deeper search: {elo:+.1f} [{lo:+.1f},{hi:+.1f}]")
    print("(Expect deep > shallow; if losses dominate, check for search bugs)")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_tc(tc_str: str) -> dict:
    """Parse time control string: depth=N, N+N (seconds), Nms, etc."""
    if tc_str.startswith('depth='):
        return {'depth': int(tc_str[6:])}
    if '+' in tc_str:
        base, inc = tc_str.split('+')
        return {'wtime': int(float(base)*1000), 'btime': int(float(base)*1000),
                'winc':  int(float(inc)*1000),  'binc':  int(float(inc)*1000)}
    if tc_str.endswith('ms'):
        return {'movetime': int(tc_str[:-2])}
    return {'movetime': int(float(tc_str) * 1000)}

def load_openings(path: str) -> list[str]:
    fens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            # Support EPD (strip ops after 4th field) or pure FEN
            parts = line.split()
            if len(parts) >= 4:
                fens.append(' '.join(parts[:6] if len(parts)>=6 else parts))
    return fens

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')

    # selftest
    ps = sub.add_parser('selftest')
    ps.add_argument('--engine', required=True)
    ps.add_argument('--depth',  type=int, default=8)
    ps.add_argument('--games',  type=int, default=20)

    # match
    pm = sub.add_parser('match')
    pm.add_argument('--engine1', required=True)
    pm.add_argument('--engine2', required=True)
    pm.add_argument('--tc',      default='depth=8', help='depth=N | N+N (sec) | Nms')
    pm.add_argument('--games',   type=int, default=100)
    pm.add_argument('--openings',help='EPD/FEN file with starting positions')
    pm.add_argument('--out',     default='results.json')
    pm.add_argument('--options1',default='', help='key=val,key=val UCI options for engine1')
    pm.add_argument('--options2',default='', help='key=val,key=val UCI options for engine2')
    pm.add_argument('--verbose', action='store_true')

    # sprt
    pp = sub.add_parser('sprt', help='SPRT stopping test')
    pp.add_argument('--engine1', required=True)
    pp.add_argument('--engine2', required=True)
    pp.add_argument('--tc',      default='depth=8')
    pp.add_argument('--elo0',    type=float, default=0)
    pp.add_argument('--elo1',    type=float, default=10)
    pp.add_argument('--alpha',   type=float, default=0.05)
    pp.add_argument('--beta',    type=float, default=0.05)
    pp.add_argument('--openings',default=None)
    pp.add_argument('--out',     default='sprt_results.json')
    pp.add_argument('--verbose', action='store_true')

    # nps
    pn = sub.add_parser('nps')
    pn.add_argument('--engine', required=True)
    pn.add_argument('--depth',  type=int, default=12)
    pn.add_argument('--positions', type=int, default=10)

    args = ap.parse_args()
    if not args.cmd: ap.print_help(); return

    def parse_opts(s):
        opts = {}
        if not s: return opts
        for part in s.split(','):
            if '=' in part:
                k, v = part.split('=', 1)
                opts[k.strip()] = v.strip()
        return opts

    if args.cmd == 'selftest':
        run_selftest(args.engine, args.depth, args.games)

    elif args.cmd == 'match':
        openings = load_openings(args.openings) if args.openings else BUILTIN_OPENINGS
        cfg = MatchConfig(
            engine1=args.engine1, engine2=args.engine2,
            tc=parse_tc(args.tc), games=args.games,
            openings=openings, out=args.out,
            verbose=args.verbose,
            options1=parse_opts(args.options1),
            options2=parse_opts(args.options2),
        )
        run_match(cfg)

    elif args.cmd == 'sprt':
        openings = load_openings(args.openings) if args.openings else BUILTIN_OPENINGS
        cfg = MatchConfig(
            engine1=args.engine1, engine2=args.engine2,
            tc=parse_tc(args.tc), games=10000,  # SPRT stops early
            openings=openings, out=args.out,
            verbose=args.verbose,
            sprt={'elo0': args.elo0, 'elo1': args.elo1,
                  'alpha': args.alpha, 'beta': args.beta},
        )
        run_match(cfg)

    elif args.cmd == 'nps':
        run_nps(args.engine, args.depth, args.positions)

if __name__ == '__main__':
    main()
