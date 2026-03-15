#!/usr/bin/env python3
"""
Hugine NNUE Trainer
===================
Full training pipeline for Hugine's HalfKP NNUE network.

Architecture (default / --large / --xl):
  40960 → 256 (→ 512 → 1024) → 32 (→ 64 → 128) → 32 → 1
  All hidden layers use ClippedReLU [0, 127].

As of Hugine 5.1.0 the engine also natively loads any Stockfish .nnue file
(HalfKP-256, HalfKA-256, HalfKAv2-512/1024/1536, COMPRESSED_LEB128) via the
SFNNUEEvaluator back-end.  Use `inspect` to examine any .nnue file — Hugine-
native or Stockfish — without running the engine.

Usage
-----
  # Generate random test weights (for engine pipeline testing only)
  python3 train.py random --out hugine.nnue

  # Generate self-play training data
  python3 train.py generate --engine ./hugine_fast --games 5000 --depth 8 --out data/train.txt

  # Train the network (requires torch + numpy)
  python3 train.py train --data data/train.txt --epochs 100 --out hugine.nnue

  # Validate a Hugine-native .nnue file
  python3 train.py validate --file hugine.nnue

  # Inspect ANY .nnue file (Hugine-native OR Stockfish SF10-SF16+)
  python3 train.py inspect --file nn-XXXXXX.nnue

Data format (train.txt)
-----------------------
  One line per position:
    <fen> | <score_cp>
  Example:
    rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2 | 15

Requirements
------------
  pip install numpy torch   (for training)
  pip install chess         (for FEN parsing / self-play generation)

Hugine-native .nnue Binary Format
-----------------------------------
  Header: 7 × uint32  (little-endian)
    magic=0x5A5A5A5A, version, FT_INPUTS, FT_SIZE, L1_SIZE, L2_SIZE, out_dim=1
  FT weights : FT_INPUTS × FT_SIZE  int8
  FT bias    : FT_SIZE              int16
  L1 weights : FT_SIZE  × L1_SIZE   int8
  L1 bias    : L1_SIZE              int16
  L2 weights : L1_SIZE  × L2_SIZE   int8
  L2 bias    : L2_SIZE              int16
  Out weights: L2_SIZE              int8
  Out bias   : 1                    int16

Stockfish .nnue files load natively in the engine — no conversion needed.
Use `inspect` to see the detected architecture of any file.
"""

import argparse, os, struct, sys, random, subprocess, time

# ── Architecture constants (Hugine-native) ────────────────────────────────────
MAGIC         = 0x5A5A5A5A
VERSION_STD   = 2    # 256-node build
VERSION_LARGE = 3    # 512-node build (NNUE_LARGE)
VERSION_XL    = 4    # 1024-node build (NNUE_XL)

FT_INPUTS   = 40960  # HalfKP: 2 * 64 * 64 * 5

# ── Stockfish .nnue architecture fingerprints ─────────────────────────────────
# These are the known architecture-hash → description mappings used by the
# engine's SFNNUEEvaluator to auto-detect the SF network variant.
SF_ARCH_HASHES = {
    0x5D69D7B8: ("HalfKP-256",        40960, 256,  2, (2,  8, 32)),
    0xF8E55352: ("HalfKA-256",         40960, 256,  2, (2,  8, 32)),
    0x3E5AA6EE: ("HalfKAv2-1024",      45056, 1024, 2, (2,  8, 32)),
    0x7F23558C: ("HalfKAv2-512",       45056, 512,  2, (2,  8, 32)),
    0x7F234CB8: ("HalfKAv2-1024-alt",  45056, 1024, 2, (2,  8, 32)),
    0x1C103072: ("SFNNv9/HalfKAv2",    45056, 1536, 8, (8, 32,  1)),
    0xD3CEE169: ("SFNNv9/HalfKAv2-alt",45056, 1536, 8, (8, 32,  1)),
}
SF_VERSION_MARKER = 0x7A000000  # top byte of SF version field

FT_SIZE_STD = 256;  L1_SIZE_STD = 32
FT_SIZE_LRG = 512;  L1_SIZE_LRG = 64
FT_SIZE_XL  = 1024; L1_SIZE_XL  = 128
L2_SIZE     = 32    # same for both

FT_SCALE     = 128  # must match NNUEEvaluator::FT_SCALE
HIDDEN_SCALE = 64   # must match NNUEEvaluator::HIDDEN_SCALE


# ── HalfKP feature index (mirrors hugine.cpp exactly) ─────────────────────────
def halfkp_index(king_color, king_sq, piece_color, piece_sq, pt):
    """pt: 1=P 2=N 3=B 4=R 5=Q  (0=King → -1)
    color: 0=White 1=Black
    """
    if pt == 0:
        return -1
    same = 0 if (piece_color == king_color) else 1
    return (same * 64 * 64 * 5) + (king_sq * 64 + piece_sq) * 5 + (pt - 1)


def fen_to_features(fen):
    """Return (active_feature_indices, is_black_to_move).
    Uses python-chess if available, otherwise a built-in FEN parser.
    """
    try:
        import chess
        board = chess.Board(fen)
        PT = {chess.PAWN:1, chess.KNIGHT:2, chess.BISHOP:3,
              chess.ROOK:4, chess.QUEEN:5, chess.KING:0}
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk is None or bk is None:
            return [], board.turn == chess.BLACK
        feats = []
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p is None or p.piece_type == chess.KING:
                continue
            idx = halfkp_index(0, wk, 0 if p.color == chess.WHITE else 1, sq, PT[p.piece_type])
            if idx != -1:
                feats.append(idx)
        return feats, board.turn == chess.BLACK
    except ImportError:
        pass
    # Built-in fallback
    parts = fen.split()
    side = parts[1] if len(parts) > 1 else 'w'
    FP = {'P':(0,1),'N':(0,2),'B':(0,3),'R':(0,4),'Q':(0,5),'K':(0,0),
           'p':(1,1),'n':(1,2),'b':(1,3),'r':(1,4),'q':(1,5),'k':(1,0)}
    pm = {}; rank, file = 7, 0
    for ch in parts[0]:
        if ch == '/': rank -= 1; file = 0
        elif ch.isdigit(): file += int(ch)
        else: pm[rank*8+file] = FP.get(ch,(0,0)); file += 1
    wk = next((s for s,(c,t) in pm.items() if c==0 and t==0), None)
    if wk is None:
        return [], side == 'b'
    feats = []
    for sq,(pc,pt) in pm.items():
        if pt == 0: continue
        idx = halfkp_index(0, wk, pc, sq, pt)
        if idx != -1 and 0 <= idx < FT_INPUTS:
            feats.append(idx)
    return feats, side == 'b'


# ── File I/O ──────────────────────────────────────────────────────────────────
def write_nnue(path, weights, large=False, xl=False):
    """Write a Hugine binary .nnue from weight dict.
    All arrays should be float32 numpy arrays already at quantised scale.
    """
    import numpy as np
    if xl:
        FT_SIZE, L1_SIZE, version = FT_SIZE_XL, L1_SIZE_XL, VERSION_XL
    elif large:
        FT_SIZE, L1_SIZE, version = FT_SIZE_LRG, L1_SIZE_LRG, VERSION_LARGE
    else:
        FT_SIZE, L1_SIZE, version = FT_SIZE_STD, L1_SIZE_STD, VERSION_STD

    i8  = lambda a: np.clip(np.round(a), -128,   127).astype(np.int8)
    i16 = lambda a: np.clip(np.round(a), -32768, 32767).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        for v in [MAGIC, version, FT_INPUTS, FT_SIZE, L1_SIZE, L2_SIZE, 1]:
            f.write(struct.pack('<I', v))
        f.write(i8 (weights['ft_w'].reshape(FT_INPUTS, FT_SIZE)).tobytes())
        f.write(i16(weights['ft_b']).tobytes())
        f.write(i8 (weights['l1_w'].reshape(FT_SIZE,  L1_SIZE )).tobytes())
        f.write(i16(weights['l1_b']).tobytes())
        f.write(i8 (weights['l2_w'].reshape(L1_SIZE,  L2_SIZE )).tobytes())
        f.write(i16(weights['l2_b']).tobytes())
        f.write(i8 (weights['out_w']).tobytes())
        f.write(struct.pack('<h', int(round(float(weights['out_b'])))))
    sz = os.path.getsize(path)
    print(f"  Written: {path}  ({sz:,} bytes = {sz/1024/1024:.2f} MB)")


# ── random ────────────────────────────────────────────────────────────────────
def cmd_random(args):
    """Random-weight .nnue for engine pipeline testing. Not useful for play strength."""
    xl    = getattr(args, 'xl', False)
    large = args.large or xl
    if xl:
        FT_SIZE, L1_SIZE, arch = FT_SIZE_XL, L1_SIZE_XL, 'XL 1024'
    elif large:
        FT_SIZE, L1_SIZE, arch = FT_SIZE_LRG, L1_SIZE_LRG, 'LARGE 512'
    else:
        FT_SIZE, L1_SIZE, arch = FT_SIZE_STD, L1_SIZE_STD, 'std 256'
    print(f"Generating random {arch} .nnue ...")
    try:
        import numpy as np
        rng = np.random.default_rng(42)
        s = 0.01
        write_nnue(args.out, dict(
            ft_w =(rng.standard_normal((FT_INPUTS,FT_SIZE))*s).astype('f4')*FT_SCALE,
            ft_b = np.zeros(FT_SIZE, 'f4'),
            l1_w =(rng.standard_normal((FT_SIZE, L1_SIZE))*s).astype('f4')*HIDDEN_SCALE,
            l1_b = np.zeros(L1_SIZE, 'f4'),
            l2_w =(rng.standard_normal((L1_SIZE, L2_SIZE))*s).astype('f4')*HIDDEN_SCALE,
            l2_b = np.zeros(L2_SIZE, 'f4'),
            out_w=(rng.standard_normal(L2_SIZE)*s).astype('f4')*HIDDEN_SCALE,
            out_b=0.0,
        ), large=large, xl=xl)
    except ImportError:
        # Pure-Python fallback
        if xl:
            version, FT_SIZE, L1_SIZE = VERSION_XL, FT_SIZE_XL, L1_SIZE_XL
        elif large:
            version, FT_SIZE, L1_SIZE = VERSION_LARGE, FT_SIZE_LRG, L1_SIZE_LRG
        else:
            version, FT_SIZE, L1_SIZE = VERSION_STD, FT_SIZE_STD, L1_SIZE_STD
        random.seed(42)
        ri8  = lambda n: bytes([random.randint(0,255) for _ in range(n)])
        ri16 = lambda n: b''.join(struct.pack('<h',random.randint(-8,8)) for _ in range(n))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        with open(args.out, 'wb') as f:
            for v in [MAGIC, version, FT_INPUTS, FT_SIZE, L1_SIZE, L2_SIZE, 1]:
                f.write(struct.pack('<I', v))
            for step, data in [('FT', ri8(FT_INPUTS*FT_SIZE)+ri16(FT_SIZE)),
                                ('L1', ri8(FT_SIZE*L1_SIZE)+ri16(L1_SIZE)),
                                ('L2', ri8(L1_SIZE*L2_SIZE)+ri16(L2_SIZE)),
                                ('out',ri8(L2_SIZE)+struct.pack('<h',0))]:
                print(f'  {step} ...', end='', flush=True); f.write(data)
        sz = os.path.getsize(args.out)
        print(f"\n  Written: {args.out}  ({sz:,} bytes = {sz/1024/1024:.2f} MB)")


# ── generate ──────────────────────────────────────────────────────────────────
OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 2 3",
    "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
    "r1bqkbnr/pppp1ppp/2n5/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4",
    "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
]


def engine_cmd(engine, inp, timeout=60):
    try:
        r = subprocess.run([engine], input=inp,
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception:
        return ''


def engine_bestmove(engine, fen, depth):
    out = engine_cmd(engine, f"position fen {fen}\ngo depth {depth}\nquit\n")
    for line in out.splitlines():
        if line.startswith('bestmove '):
            bm = line.split()[1]
            return None if bm == '0000' else bm
    return None


def engine_score(engine, fen, depth):
    out = engine_cmd(engine, f"position fen {fen}\ngo depth {depth}\nquit\n")
    score = None
    for line in out.splitlines():
        if 'score cp' in line:
            try: score = int(line.split('score cp')[1].split()[0])
            except (ValueError, IndexError): pass
        elif 'score mate' in line:
            try:
                m = int(line.split('score mate')[1].split()[0])
                score = 30000 if m > 0 else -30000
            except (ValueError, IndexError): pass
    return score


def apply_move(fen, uci):
    import chess
    b = chess.Board(fen); b.push(chess.Move.from_uci(uci)); return b.fen()


def cmd_generate(args):
    if not os.path.exists(args.engine):
        print(f"ERROR: engine not found: {args.engine}"); sys.exit(1)
    has_chess = True
    try:
        import chess  # noqa
    except ImportError:
        has_chess = False
        print("WARNING: python-chess not installed (pip install chess).")
        print("         Will evaluate opening positions only (no self-play).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    count = 0; t0 = time.time()
    print(f"Generating {args.games} games  depth={args.depth}  → {args.out}")

    with open(args.out, 'w') as fout:
        for gi in range(args.games):
            opening = OPENINGS[gi % len(OPENINGS)]
            try:
                fen = opening
                for _ in range(args.maxply):
                    bm = engine_bestmove(args.engine, fen, args.depth)
                    if bm is None: break
                    score = engine_score(args.engine, fen, args.depth)
                    if score is not None:
                        score = max(-3000, min(3000, score))
                        fout.write(f"{fen} | {score}\n")
                        count += 1
                    if not has_chess: break
                    fen = apply_move(fen, bm)
            except Exception as e:
                score = engine_score(args.engine, opening, args.depth)
                if score is not None:
                    fout.write(f"{opening} | {score}\n"); count += 1

            if (gi+1) % 25 == 0:
                rate = count / max(time.time()-t0, 0.001)
                print(f"  game {gi+1}/{args.games}  positions={count}  {rate:.0f} pos/s", end='\r')

    elapsed = time.time() - t0
    print(f"\n  Done: {count:,} positions in {elapsed:.1f}s  →  {args.out}")


# ── train ─────────────────────────────────────────────────────────────────────
def cmd_train(args):
    try:
        import torch, torch.nn as nn, torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        import numpy as np
    except ImportError:
        print("ERROR: PyTorch + numpy required.  pip install torch numpy")
        print(f"       For a test file without training: python3 train.py random --out {args.out}")
        sys.exit(1)

    xl    = getattr(args, 'xl', False)
    large = args.large or xl
    if xl:
        FT_SIZE, L1_SIZE, arch = FT_SIZE_XL, L1_SIZE_XL, 'XL 1024'
    elif large:
        FT_SIZE, L1_SIZE, arch = FT_SIZE_LRG, L1_SIZE_LRG, 'LARGE 512'
    else:
        FT_SIZE, L1_SIZE, arch = FT_SIZE_STD, L1_SIZE_STD, 'std 256'
    print(f"Training Hugine NNUE  ({arch})")
    print(f"  data={args.data}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")

    # ── Load data ──────────────────────────────────────────────────────────
    fens, scores = [], []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line: continue
            fen_part, sc_part = line.rsplit('|', 1)
            try:
                fens.append(fen_part.strip())
                scores.append(int(sc_part.strip()))
            except ValueError: pass
    if not fens:
        print("ERROR: no valid lines.  Expected: <fen> | <score_cp>"); sys.exit(1)
    print(f"  Loaded {len(fens):,} positions")

    # ── Dataset ────────────────────────────────────────────────────────────
    class HalfKPDataset(Dataset):
        def __init__(self, fens, scores):
            self.fens = fens
            # WLD sigmoid target
            self.targets = [1.0/(1.0+2.718281828**(-s/400.0)) for s in scores]
        def __len__(self): return len(self.fens)
        def __getitem__(self, idx):
            feats, is_black = fen_to_features(self.fens[idx])
            x = torch.zeros(FT_INPUTS, dtype=torch.float32)
            for fi in feats:
                if 0 <= fi < FT_INPUTS: x[fi] = 1.0
            t = torch.tensor(self.targets[idx], dtype=torch.float32)
            if is_black: t = 1.0 - t
            return x, t

    loader = DataLoader(HalfKPDataset(fens, scores),
                        batch_size=args.batch, shuffle=True,
                        num_workers=min(4, os.cpu_count() or 1), pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────
    class HugineNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.ft  = nn.Linear(FT_INPUTS, FT_SIZE)
            self.l1  = nn.Linear(FT_SIZE,   L1_SIZE)
            self.l2  = nn.Linear(L1_SIZE,   L2_SIZE)
            self.out = nn.Linear(L2_SIZE,   1)
            for m in [self.ft, self.l1, self.l2, self.out]:
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        def crelu(self, x): return torch.clamp(x, 0.0, 1.0)
        def forward(self, x):
            x = self.crelu(self.ft(x))
            x = self.crelu(self.l1(x))
            x = self.crelu(self.l2(x))
            return torch.sigmoid(self.out(x).squeeze(-1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    model = HugineNet().to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCELoss()
    best_loss = float('inf'); best_ep = 0

    # ── Loop ───────────────────────────────────────────────────────────────
    print()
    for ep in range(1, args.epochs+1):
        model.train(); total = 0.0; nb = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        avg = total / max(nb, 1)
        mark = ''
        if avg < best_loss:
            best_loss = avg; best_ep = ep; mark = ' ← best'
            torch.save(model.state_dict(), args.out + '.ckpt')
        print(f"  Epoch {ep:3d}/{args.epochs}  loss={avg:.6f}  lr={sched.get_last_lr()[0]:.1e}{mark}")

    print(f"\n  Best: epoch {best_ep}  loss={best_loss:.6f}")
    if os.path.exists(args.out + '.ckpt'):
        model.load_state_dict(torch.load(args.out + '.ckpt', map_location=device))
        os.remove(args.out + '.ckpt')
    model.eval()

    # ── Export ─────────────────────────────────────────────────────────────
    print("Exporting weights ...")
    with torch.no_grad():
        # nn.Linear stores weight as (out_features, in_features) → transpose for engine
        w = {
            'ft_w' : model.ft.weight.cpu().numpy().T  * FT_SCALE,
            'ft_b' : model.ft.bias.cpu().numpy()      * FT_SCALE,
            'l1_w' : model.l1.weight.cpu().numpy().T  * HIDDEN_SCALE,
            'l1_b' : model.l1.bias.cpu().numpy()      * HIDDEN_SCALE,
            'l2_w' : model.l2.weight.cpu().numpy().T  * HIDDEN_SCALE,
            'l2_b' : model.l2.bias.cpu().numpy()      * HIDDEN_SCALE,
            'out_w': model.out.weight.cpu().numpy()[0] * HIDDEN_SCALE,
            'out_b': float(model.out.bias.cpu().numpy()[0]) * HIDDEN_SCALE,
        }
    write_nnue(args.out, {k: v.astype('float32') if hasattr(v,'astype') else v
                          for k,v in w.items()}, large=large, xl=xl)
    print(f"\nValidate:  python3 train.py validate --file {args.out}")
    print(f"Load:      setoption name EvalFile value {args.out}")


# ── validate ──────────────────────────────────────────────────────────────────
def cmd_validate(args):
    path = args.file
    if not os.path.exists(path):
        print(f"ERROR: {path} not found"); sys.exit(1)
    sz = os.path.getsize(path)
    print(f"Validating: {path}  ({sz:,} bytes = {sz/1024/1024:.2f} MB)")
    with open(path, 'rb') as f:
        u32 = lambda: struct.unpack('<I', f.read(4))[0]
        magic=u32(); ver=u32(); fi=u32(); fs=u32(); l1=u32(); l2=u32(); od=u32()
    checks = [
        ("magic",     magic, lambda v: v==MAGIC,      f"0x{magic:08X}", "0x5A5A5A5A"),
        ("version",   ver,   lambda v: v in (2,3,4),  str(ver),         "2, 3 or 4"),
        ("FT_INPUTS", fi,    lambda v: v==FT_INPUTS,  str(fi),          str(FT_INPUTS)),
        ("FT_SIZE",   fs,    lambda v: v in (256,512,1024), str(fs),   "256, 512 or 1024"),
        ("L1_SIZE",   l1,    lambda v: v in (32,64,128), str(l1),      "32, 64 or 128"),
        ("L2_SIZE",   l2,    lambda v: v==L2_SIZE,    str(l2),          str(L2_SIZE)),
        ("out_dim",   od,    lambda v: v==1,          str(od),          "1"),
    ]
    print()
    ok_all = True
    for name, val, check, got, want in checks:
        ok = check(val); ok_all = ok_all and ok
        print(f"  {'✓' if ok else '✗'} {name:10s} = {got:>10s}  (expected {want})")
    exp_sz = 28 + fi*fs + fs*2 + fs*l1 + l1*2 + l1*l2 + l2*2 + l2 + 2
    ok_sz = (sz == exp_sz); ok_all = ok_all and ok_sz
    print(f"  {'✓' if ok_sz else '✗'} filesize   = {sz:>10,}  (expected {exp_sz:,})")
    print()
    if ok_all:
        arch = "XL 1024" if fs==1024 else "LARGE 512" if fs==512 else "std 256"
        print(f"  ✓  VALID  [{arch} {fs}→{l1}→{l2}→1]")
        print(f"     setoption name EvalFile value {path}")
    else:
        print("  ✗  INVALID — will not load in the engine"); sys.exit(1)


# ── inspect ───────────────────────────────────────────────────────────────────
def _leb128_decode_weights(data, n):
    """Decode n signed values from a COMPRESSED_LEB128 stream (SF16+ format)."""
    values = []
    i = 0
    while len(values) < n and i < len(data):
        val = 0; shift = 0
        while True:
            b = data[i]; i += 1
            val |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        if val & (1 << (shift - 1)):
            val -= (1 << shift)
        values.append(val)
    return values, i


def cmd_inspect(args):
    """Inspect any .nnue file — Hugine-native or Stockfish SF10–SF16+."""
    path = args.file
    if not os.path.exists(path):
        print(f"ERROR: {path} not found"); sys.exit(1)

    sz = os.path.getsize(path)
    print(f"\nInspecting: {path}")
    print(f"  File size: {sz:,} bytes  ({sz / 1024 / 1024:.2f} MB)\n")

    with open(path, 'rb') as f:
        header = f.read(32)

    if len(header) < 8:
        print("  ✗ File too small to have a valid header"); sys.exit(1)

    ver, arch_hash = struct.unpack_from('<II', header, 0)

    # ── Stockfish .nnue? ──────────────────────────────────────────────────────
    if (ver & 0xFF000000) == SF_VERSION_MARKER:
        print("  Type:       Stockfish .nnue  (loads natively in Hugine 5.1.0+)")
        print(f"  SF version: 0x{ver:08X}")
        print(f"  Arch hash:  0x{arch_hash:08X}")

        if arch_hash in SF_ARCH_HASHES:
            name, ft_in, ft_sz, l1_sz, _ = SF_ARCH_HASHES[arch_hash]
            print(f"  Architecture: {name}")
            print(f"  Feature transformer: {ft_in} inputs → {ft_sz} nodes")
            print(f"  L1 size:  {l1_sz}")
        else:
            print(f"  Architecture: unknown hash 0x{arch_hash:08X}")
            print("  (engine will attempt auto-detection by weight count)")

        # Detect compression
        # COMPRESSED_LEB128 files have the arch hash at offset 4 and then
        # a size marker.  Heuristic: count how many non-zero bytes appear in
        # the body — compressed data is denser.
        with open(path, 'rb') as f:
            f.read(8)   # skip version + arch hash
            sample = f.read(256)
        high_byte_ratio = sum(1 for b in sample if b > 127) / max(len(sample), 1)
        compressed = high_byte_ratio < 0.3
        print(f"  Format:     {'COMPRESSED_LEB128' if compressed else 'uncompressed binary'}")
        print()
        print("  Engine usage:")
        print(f"    setoption name EvalFile value {path}")
        print(f"    (SFNNUEEvaluator back-end will be selected automatically)")
        return

    # ── Hugine-native .nnue? ──────────────────────────────────────────────────
    if len(header) < 28:
        print("  ✗ Header too short for Hugine-native format"); sys.exit(1)

    magic, hver, fi, fs, l1, l2, od = struct.unpack_from('<7I', header, 0)

    print("  Type:    Hugine-native .nnue")
    checks = [
        ("magic",     magic, lambda v: v == MAGIC,              f"0x{magic:08X}", "0x5A5A5A5A"),
        ("version",   hver,  lambda v: v in (2, 3, 4),         str(hver),        "2, 3 or 4"),
        ("FT_INPUTS", fi,    lambda v: v == FT_INPUTS,          str(fi),          str(FT_INPUTS)),
        ("FT_SIZE",   fs,    lambda v: v in (256, 512, 1024),   str(fs),          "256, 512 or 1024"),
        ("L1_SIZE",   l1,    lambda v: v in (32, 64, 128),      str(l1),          "32, 64 or 128"),
        ("L2_SIZE",   l2,    lambda v: v == 32,          str(l2),          "32"),
        ("out_dim",   od,    lambda v: v == 1,           str(od),          "1"),
    ]
    ok_all = True
    for name, val, chk, got, want in checks:
        ok = chk(val); ok_all = ok_all and ok
        print(f"  {'✓' if ok else '✗'} {name:10s} = {got:>10s}  (expected {want})")

    exp_sz = 28 + fi * fs + fs * 2 + fs * l1 + l1 * 2 + l1 * l2 + l2 * 2 + l2 + 2
    ok_sz = (sz == exp_sz); ok_all = ok_all and ok_sz
    print(f"  {'✓' if ok_sz else '✗'} filesize   = {sz:>10,}  (expected {exp_sz:,})")
    print()
    arch = "LARGE 512" if fs == 512 else "standard 256"
    if ok_all:
        print(f"  ✓  VALID Hugine-native .nnue  [{arch}  {fi}→{fs}→{l1}→{l2}→1]")
        print(f"     setoption name EvalFile value {path}")
        print(f"     (NNUEEvaluator back-end will be selected automatically)")
    else:
        print("  ✗  INVALID — will not load in the engine"); sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(prog='train.py', description='Hugine NNUE Trainer',
         formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
subcommands:
  random     Random-weight .nnue (engine pipeline testing only)
  generate   Generate training data via engine self-play
  train      Train NNUE from position data (requires torch + numpy)
  validate   Verify a Hugine-native .nnue file will load correctly
  inspect    Inspect any .nnue file (Hugine-native OR Stockfish SF10-SF16+)

examples:
  python3 train.py random   --out hugine.nnue
  python3 train.py random   --out hugine_xl.nnue --xl
  python3 train.py generate --engine ./hugine_fast --games 5000 --out data/train.txt
  python3 train.py train    --data data/train.txt  --epochs 100 --out hugine.nnue
  python3 train.py validate --file hugine.nnue
  python3 train.py inspect  --file nn-XXXXXX.nnue      # any Stockfish .nnue
  python3 train.py inspect  --file hugine.nnue          # Hugine-native
        """)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('random',   help='Random-weight .nnue (no training)')
    p.add_argument('--out',   default='hugine.nnue')
    p.add_argument('--large', action='store_true', help='512-node architecture')
    p.add_argument('--xl',    action='store_true', help='1024-node architecture (NNUE_XL)')

    p = sub.add_parser('generate', help='Generate training data from self-play')
    p.add_argument('--engine', required=True,   help='Path to hugine binary')
    p.add_argument('--games',  type=int, default=1000)
    p.add_argument('--depth',  type=int, default=6)
    p.add_argument('--maxply', type=int, default=80, help='Max half-moves per game')
    p.add_argument('--out',    default='data/train.txt')

    p = sub.add_parser('train',    help='Train from position data')
    p.add_argument('--data',   required=True,            help='Training data file')
    p.add_argument('--out',    default='hugine.nnue')
    p.add_argument('--epochs', type=int,   default=50)
    p.add_argument('--batch',  type=int,   default=1024)
    p.add_argument('--lr',     type=float, default=1e-3)
    p.add_argument('--large',  action='store_true')
    p.add_argument('--xl',     action='store_true', help='1024-node architecture')

    p = sub.add_parser('validate', help='Validate a Hugine-native .nnue file')
    p.add_argument('--file', required=True)

    p = sub.add_parser('inspect',  help='Inspect any .nnue (Hugine-native or Stockfish)')
    p.add_argument('--file', required=True)

    args = ap.parse_args()
    cmds = {
        'random':   cmd_random,
        'generate': cmd_generate,
        'train':    cmd_train,
        'validate': cmd_validate,
        'inspect':  cmd_inspect,
    }
    cmds[args.cmd](args)

if __name__ == '__main__':
    main()
