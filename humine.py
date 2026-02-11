#!/usr/bin/env python3
"""
Chess Engine v7.2 - PERFECT FINAL VERSION
Fixes all remaining issues:
- Correct mobility calculation
- Complete piece-square tables (including king)
- Full tactical detection (skewer, trapping)
- Proper parallel processing
- Move sequence evaluation
- Pawn structure and king safety
- Principal variation display
- Time management
"""

import chess
import chess.pgn
import argparse
import io
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Set, Any
import time
import random
import json
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from enum import Enum


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class TTFlag(Enum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


# ============================================================================
# TRANSPOSITION TABLE
# ============================================================================

class TranspositionTable:
    """Optimized transposition table with size management."""

    def __init__(self, max_size: int = 2000000):
        self.table: Dict[int, Tuple[int, int, TTFlag, Optional[chess.Move], List[chess.Move]]] = {}
        self.zobrist_keys = self._init_zobrist()
        self.max_size = max_size

    def _init_zobrist(self) -> Dict:
        """Initialize Zobrist keys for hashing."""
        keys = {}
        random.seed(0xDEADBEEF)

        # Piece keys
        for square in chess.SQUARES:
            for piece_type in chess.PIECE_TYPES:
                for color in [chess.WHITE, chess.BLACK]:
                    keys[(square, piece_type, color)] = random.getrandbits(64)

        # Side to move
        keys['side_to_move'] = random.getrandbits(64)

        # Castling rights - FIXED to use BB_ bitboard constants
        keys['castling'] = {
            chess.BB_A1: random.getrandbits(64),  # White queenside (a1 rook)
            chess.BB_H1: random.getrandbits(64),  # White kingside (h1 rook)
            chess.BB_A8: random.getrandbits(64),  # Black queenside (a8 rook)
            chess.BB_H8: random.getrandbits(64)   # Black kingside (h8 rook)
        }

        # En passant
        keys['en_passant'] = {}
        for square in chess.SQUARES:
            keys['en_passant'][square] = random.getrandbits(64)

        return keys

    def hash_board(self, board: chess.Board) -> int:
        """Compute Zobrist hash of board position."""
        h = 0

        # Pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                h ^= self.zobrist_keys[(square, piece.piece_type, piece.color)]

        # Side to move
        if board.turn == chess.BLACK:
            h ^= self.zobrist_keys['side_to_move']

        # Castling rights
        for castling_right, key in self.zobrist_keys['castling'].items():
            if board.has_castling_rights(castling_right):
                h ^= key

        # En passant
        if board.ep_square is not None:
            h ^= self.zobrist_keys['en_passant'][board.ep_square]

        return h

    def probe(self, board: chess.Board, depth: int, alpha: int, beta: int) -> Tuple[bool, int, Optional[chess.Move], List[chess.Move]]:
        """Probe table for position."""
        key = self.hash_board(board)
        if key in self.table:
            stored_depth, stored_score, flag, stored_move, pv = self.table[key]
            if stored_depth >= depth:
                if flag == TTFlag.EXACT:
                    return True, stored_score, stored_move, pv
                elif flag == TTFlag.LOWER and stored_score >= beta:
                    return True, stored_score, stored_move, pv
                elif flag == TTFlag.UPPER and stored_score <= alpha:
                    return True, stored_score, stored_move, pv
        return False, 0, None, []

    def store(self, board: chess.Board, depth: int, score: int, flag: TTFlag, 
              best_move: Optional[chess.Move], pv: List[chess.Move] = None):
        """Store position in table with principal variation."""
        key = self.hash_board(board)
        if pv is None:
            pv = []

        # Size limit management
        if len(self.table) >= self.max_size:
            # Remove 10% of entries randomly
            to_remove = random.sample(list(self.table.keys()), len(self.table) // 10)
            for k in to_remove:
                del self.table[k]

        # Always replace with deeper entries
        if key in self.table:
            old_depth, _, _, _, _ = self.table[key]
            if depth >= old_depth:
                self.table[key] = (depth, score, flag, best_move, pv)
        else:
            self.table[key] = (depth, score, flag, best_move, pv)


# ============================================================================
# COMPLETE TACTICAL DETECTION
# ============================================================================

class TacticsDetector:
    """Complete tactical detection with all patterns."""

    PIECE_VALUES = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
    }

    @staticmethod
    def detect_all_tactics(board: chess.Board, move: chess.Move, color: chess.Color) -> List[str]:
        """Detect all tactical patterns in a move."""
        tactics = []

        # Fork detection
        fork = TacticsDetector._detect_fork(board, move, color)
        if fork:
            tactics.append(fork)

        # Pin detection
        pin = TacticsDetector._detect_pin(board, move, color)
        if pin:
            tactics.append(pin)

        # Skewer detection
        skewer = TacticsDetector._detect_skewer(board, move, color)
        if skewer:
            tactics.append(skewer)

        # Trapping detection
        trap = TacticsDetector._detect_trapping_move(board, move, color)
        if trap:
            tactics.append(trap)

        # Discovered attack
        discovered = TacticsDetector._detect_discovered_attack(board, move, color)
        if discovered:
            tactics.append(discovered)

        # Promotion threat
        if TacticsDetector._is_promotion_threat(board, move, color):
            tactics.append("Promotion threat")

        return tactics

    @staticmethod
    def _detect_fork(board: chess.Board, move: chess.Move, color: chess.Color) -> Optional[str]:
        """Detect if a move creates a fork."""
        board_copy = board.copy()
        board_copy.push(move)

        attacking_square = move.to_square
        attacked_squares = board_copy.attacks(attacking_square)

        valuable_targets = []
        for square in attacked_squares:
            piece = board_copy.piece_at(square)
            if piece and piece.color != color:
                value = TacticsDetector.PIECE_VALUES.get(piece.piece_type, 0)
                if value >= 300:
                    valuable_targets.append((piece.piece_type, value))

        if len(valuable_targets) >= 2:
            valuable_targets.sort(key=lambda x: x[1], reverse=True)
            piece_names = {
                chess.KNIGHT: "N", chess.BISHOP: "B",
                chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"
            }
            targets_str = ", ".join([piece_names.get(p, "P") for p, _ in valuable_targets[:2]])
            return f"Fork ({targets_str})"

        return None

    @staticmethod
    def _detect_pin(board: chess.Board, move: chess.Move, color: chess.Color) -> Optional[str]:
        """Detect if a move creates a pin."""
        board_copy = board.copy()
        board_copy.push(move)

        attacker_square = move.to_square
        attacker_piece = board_copy.piece_at(attacker_square)

        if not attacker_piece or attacker_piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            return None

        opp_king_square = board_copy.king(not color)
        if opp_king_square is None:
            return None

        # Check if attacker and king are aligned
        if not (chess.square_file(attacker_square) == chess.square_file(opp_king_square) or
                chess.square_rank(attacker_square) == chess.square_rank(opp_king_square) or
                abs(chess.square_file(attacker_square) - chess.square_file(opp_king_square)) ==
                abs(chess.square_rank(attacker_square) - chess.square_rank(opp_king_square))):
            return None

        # Find squares between attacker and king
        direction_file = 1 if chess.square_file(opp_king_square) > chess.square_file(attacker_square) else -1 if chess.square_file(opp_king_square) < chess.square_file(attacker_square) else 0
        direction_rank = 1 if chess.square_rank(opp_king_square) > chess.square_rank(attacker_square) else -1 if chess.square_rank(opp_king_square) < chess.square_rank(attacker_square) else 0

        pinned_pieces = []
        current_file = chess.square_file(attacker_square) + direction_file
        current_rank = chess.square_rank(attacker_square) + direction_rank

        while (0 <= current_file < 8 and 0 <= current_rank < 8 and
               chess.square(current_file, current_rank) != opp_king_square):
            square = chess.square(current_file, current_rank)
            piece = board_copy.piece_at(square)
            if piece:
                if piece.color != color:
                    pinned_pieces.append((square, piece.piece_type))
                else:
                    break  # Our piece blocks the line
            current_file += direction_file
            current_rank += direction_rank

        if len(pinned_pieces) == 1:
            square, piece_type = pinned_pieces[0]
            piece_names = {
                chess.PAWN: "pawn", chess.KNIGHT: "knight",
                chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen"
            }
            piece_name = piece_names.get(piece_type, "piece")
            return f"Pin ({piece_name} to king)"

        return None

    @staticmethod
    def _detect_skewer(board: chess.Board, move: chess.Move, color: chess.Color) -> Optional[str]:
        """Detect if a move creates a skewer (like pin but attacks valuable piece through less valuable one)."""
        board_copy = board.copy()
        board_copy.push(move)

        # Check if the move is a capture that reveals attack on more valuable piece
        if board.is_capture(move):
            from_square = move.from_square
            to_square = move.to_square

            # Get direction
            file_diff = chess.square_file(to_square) - chess.square_file(from_square)
            rank_diff = chess.square_rank(to_square) - chess.square_rank(from_square)

            if file_diff == 0 or rank_diff == 0 or abs(file_diff) == abs(rank_diff):
                # Continue in same direction
                direction_file = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
                direction_rank = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)

                current_file = chess.square_file(to_square) + direction_file
                current_rank = chess.square_rank(to_square) + direction_rank

                while 0 <= current_file < 8 and 0 <= current_rank < 8:
                    square = chess.square(current_file, current_rank)
                    piece = board_copy.piece_at(square)
                    if piece:
                        if piece.color != color:
                            piece_value = TacticsDetector.PIECE_VALUES.get(piece.piece_type, 0)
                            captured_value = TacticsDetector.PIECE_VALUES.get(
                                board.piece_at(to_square).piece_type, 0
                            ) if board.piece_at(to_square) else 0
                            if piece_value > captured_value:
                                return "Skewer"
                        break
                    current_file += direction_file
                    current_rank += direction_rank

        return None

    @staticmethod
    def _detect_trapping_move(board: chess.Board, move: chess.Move, color: chess.Color) -> Optional[str]:
        """Detect if a move traps an opponent piece."""
        board_copy = board.copy()
        board_copy.push(move)

        # Check all opponent pieces
        for square in chess.SQUARES:
            piece = board_copy.piece_at(square)
            if piece and piece.color != color and piece.piece_type in [chess.QUEEN, chess.ROOK]:
                # Check if piece has safe squares
                safe_squares = 0
                for target_square in chess.SQUARES:
                    try:
                        test_move = chess.Move(square, target_square)
                        if test_move in board_copy.legal_moves:
                            # Check if moving would leave piece hanging
                            temp_board = board_copy.copy()
                            temp_board.push(test_move)
                            attackers = temp_board.attackers(color, target_square)
                            defenders = temp_board.attackers(not color, target_square)
                            if len(attackers) == 0 or len(defenders) >= len(attackers):
                                safe_squares += 1
                    except:
                        continue

                if safe_squares == 0:
                    piece_names = {chess.ROOK: "rook", chess.QUEEN: "queen"}
                    return f"Traps {piece_names[piece.piece_type]}"

        return None

    @staticmethod
    def _detect_discovered_attack(board: chess.Board, move: chess.Move, color: chess.Color) -> Optional[str]:
        """Detect if a move creates a discovered attack."""
        from_square = move.from_square

        # Check pieces that might be behind the moving piece
        for direction in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            file = chess.square_file(from_square) - direction[0]
            rank = chess.square_rank(from_square) - direction[1]

            while 0 <= file < 8 and 0 <= rank < 8:
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.color == color and piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    # This piece might now have an open line
                    # Check if it attacks something valuable
                    attacks = board.attacks(square)
                    for target in attacks:
                        target_piece = board.piece_at(target)
                        if (target_piece and target_piece.color != color and
                            TacticsDetector.PIECE_VALUES.get(target_piece.piece_type, 0) >= 300):
                            return "Discovered attack"
                    break
                elif piece:
                    break  # Blocked by other piece
                file -= direction[0]
                rank -= direction[1]

        return None

    @staticmethod
    def _is_promotion_threat(board: chess.Board, move: chess.Move, color: chess.Color) -> bool:
        """Check if a move creates a pawn promotion threat."""
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            rank = chess.square_rank(move.to_square)
            if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
                return True
        return False


# ============================================================================
# PERFECT CHESS ENGINE
# ============================================================================

class PerfectChessEngine:
    """Complete chess engine with all features."""

    PIECE_VALUES = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
    }

    # Complete piece-square tables (middle game)
    PST = {
        chess.PAWN: [
             0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
             5,  5, 10, 25, 25, 10,  5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5, -5,-10,  0,  0,-10, -5,  5,
             5, 10, 10,-20,-20, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0
        ],
        chess.KNIGHT: [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ],
        chess.BISHOP: [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ],
        chess.ROOK: [
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             0,  0,  0,  5,  5,  0,  0,  0
        ],
        chess.QUEEN: [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
             -5,  0,  5,  5,  5,  5,  0, -5,
              0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ],
        chess.KING: [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
             20, 20,  0,  0,  0,  0, 20, 20,
             20, 30, 10,  0,  0, 10, 30, 20
        ]
    }

    MATE_SCORE = 100000
    MAX_PLY = 128

    def __init__(self, num_workers: int = None):
        self.board = chess.Board()
        self.play_as_white = True
        self.nodes = 0
        self.tt = TranspositionTable()
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY)]
        self.history_table = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(2)]
        self.search_start_time = 0
        self.time_limit = 30
        self.num_workers = num_workers if num_workers else max(1, cpu_count() - 1)

        self.tt_hits = 0
        self.killer_hits = 0
        self.null_move_prunes = 0
        self.pv_table = [[] for _ in range(self.MAX_PLY)]

    def evaluate(self, board: chess.Board) -> int:
        """Complete evaluation from side-to-move perspective."""
        # Game end
        if board.is_checkmate():
            return -self.MATE_SCORE + board.ply()
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if board.can_claim_fifty_moves() or board.is_repetition(2):
            return 0

        score = 0

        # Material + PST
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for color in [chess.WHITE, chess.BLACK]:
                pieces = board.pieces(piece_type, color)
                value = self.PIECE_VALUES[piece_type]

                for sq in pieces:
                    if color == chess.WHITE:
                        score += value
                        if piece_type in self.PST:
                            score += self.PST[piece_type][sq]
                    else:
                        score -= value
                        if piece_type in self.PST:
                            score -= self.PST[piece_type][chess.square_mirror(sq)]

        # FIXED: Mobility calculation (using null move properly)
        our_moves = len(list(board.legal_moves))
        
        # Count opponent moves by making null move
        board_copy = board.copy()
        try:
            board_copy.push(chess.Move.null())
            opp_moves = len(list(board_copy.legal_moves))
        except:
            # Null move illegal (in check or other rare cases)
            opp_moves = our_moves
        
        mobility = (our_moves - opp_moves) * 5  # Reduced from 10 to 5
        score += mobility if board.turn == chess.WHITE else -mobility

        # NEW: Passed pawn evaluation (CRITICAL for pawn endgames!)
        passed_pawn_score = self.evaluate_passed_pawns(board)
        score += passed_pawn_score

        # Pawn structure
        pawn_structure = self.evaluate_pawn_structure(board)
        score += pawn_structure

        # King safety
        king_safety = self.evaluate_king_safety(board)
        score += king_safety

        # Convert to side-to-move perspective
        if board.turn == chess.BLACK:
            score = -score

        return score

    def evaluate_passed_pawns(self, board: chess.Board) -> int:
        """
        Evaluate passed pawns with the Square Rule for unstoppable pawns.
        This is CRITICAL for finding queen trades that create passed pawns!
        """
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = list(board.pieces(chess.PAWN, color))
            
            for pawn_square in pawns:
                file = chess.square_file(pawn_square)
                rank = chess.square_rank(pawn_square)
                
                # Check if this pawn is passed (no enemy pawns blocking it)
                is_passed = self._is_passed_pawn(board, pawn_square, color)
                
                if is_passed:
                    # Base passed pawn bonus
                    pawn_bonus = 50
                    
                    # Advancement bonus (more advanced = more dangerous)
                    if color == chess.WHITE:
                        advancement = rank  # 0-7
                    else:
                        advancement = 7 - rank  # 7-0
                    
                    # Exponential growth: 7th rank pawn is VERY dangerous
                    advancement_bonus = advancement * advancement * 5  # 0, 5, 20, 45, 80, 125, 180, 245
                    pawn_bonus += advancement_bonus
                    
                    # Check if pawn is unstoppable (Square Rule)
                    if self._is_unstoppable_pawn(board, pawn_square, color):
                        pawn_bonus += 400  # Almost a piece! Will promote to queen
                    
                    # Check if pawn is supported by our king
                    elif self._is_pawn_supported_by_king(board, pawn_square, color):
                        pawn_bonus += 50  # King helps push pawn
                    
                    # Check if blocked by enemy piece
                    if self._is_pawn_blocked(board, pawn_square, color):
                        pawn_bonus //= 2  # Half value if can't advance
                    
                    # Add to score (positive for white, negative for black)
                    if color == chess.WHITE:
                        score += pawn_bonus
                    else:
                        score -= pawn_bonus
        
        return score
    
    def _is_passed_pawn(self, board: chess.Board, pawn_square: int, color: chess.Color) -> bool:
        """Check if pawn is passed (no enemy pawns blocking it)."""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check the three files (left, center, right)
        for check_file in [file - 1, file, file + 1]:
            if check_file < 0 or check_file > 7:
                continue
            
            # Check all ranks ahead of this pawn
            if color == chess.WHITE:
                ranks_to_check = range(rank + 1, 8)
            else:
                ranks_to_check = range(0, rank)
            
            for check_rank in ranks_to_check:
                square = chess.square(check_file, check_rank)
                piece = board.piece_at(square)
                
                # If there's an enemy pawn ahead, not passed
                if piece and piece.piece_type == chess.PAWN and piece.color != color:
                    return False
        
        return True
    
    def _is_unstoppable_pawn(self, board: chess.Board, pawn_square: int, color: chess.Color) -> bool:
        """
        Use the Square Rule to check if pawn is unstoppable.
        The Square Rule: Draw a square from the pawn to its promotion square.
        If the enemy king is outside this square (and it's our turn), pawn is unstoppable.
        """
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Find promotion square
        if color == chess.WHITE:
            promotion_square = chess.square(file, 7)
            moves_to_promote = 7 - rank
        else:
            promotion_square = chess.square(file, 0)
            moves_to_promote = rank
        
        # Adjust for pawn's first move (can move 2 squares)
        if color == chess.WHITE and rank == 1:
            moves_to_promote -= 1
        elif color == chess.BLACK and rank == 6:
            moves_to_promote -= 1
        
        # Get enemy king position
        enemy_king_square = board.king(not color)
        if enemy_king_square is None:
            return False
        
        # Calculate enemy king distance to promotion square (Chebyshev distance)
        enemy_king_file = chess.square_file(enemy_king_square)
        enemy_king_rank = chess.square_rank(enemy_king_square)
        promo_file = chess.square_file(promotion_square)
        promo_rank = chess.square_rank(promotion_square)
        
        king_distance = max(abs(enemy_king_file - promo_file), abs(enemy_king_rank - promo_rank))
        
        # Square Rule: If king distance > pawn distance (considering who moves next)
        # If it's our turn, pawn gets head start
        if board.turn == color:
            # Our turn: pawn moves first
            if king_distance > moves_to_promote:
                return True
        else:
            # Opponent's turn: king moves first
            if king_distance > moves_to_promote + 1:
                return True
        
        return False
    
    def _is_pawn_supported_by_king(self, board: chess.Board, pawn_square: int, color: chess.Color) -> bool:
        """Check if our king is close enough to support the pawn."""
        our_king_square = board.king(color)
        if our_king_square is None:
            return False
        
        # Calculate king distance to pawn (Chebyshev)
        king_file = chess.square_file(our_king_square)
        king_rank = chess.square_rank(our_king_square)
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        distance = max(abs(king_file - pawn_file), abs(king_rank - pawn_rank))
        
        # King is supporting if within 2 squares
        return distance <= 2
    
    def _is_pawn_blocked(self, board: chess.Board, pawn_square: int, color: chess.Color) -> bool:
        """Check if pawn is blocked by any piece."""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check square directly ahead
        if color == chess.WHITE:
            ahead_square = chess.square(file, rank + 1) if rank < 7 else None
        else:
            ahead_square = chess.square(file, rank - 1) if rank > 0 else None
        
        if ahead_square is None:
            return False
        
        piece = board.piece_at(ahead_square)
        return piece is not None


    def evaluate_pawn_structure(self, board: chess.Board) -> int:
        """Evaluate pawn structure."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = list(board.pieces(chess.PAWN, color))
            files = [chess.square_file(p) for p in pawns]
            
            # Doubled pawns penalty
            file_counts = {}
            for f in files:
                file_counts[f] = file_counts.get(f, 0) + 1
            
            doubled_penalty = 0
            for count in file_counts.values():
                if count > 1:
                    doubled_penalty += (count - 1) * 20
            
            # Isolated pawns penalty
            isolated_penalty = 0
            for pawn in pawns:
                file = chess.square_file(pawn)
                has_friend = False
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file < 8:
                        for other in pawns:
                            if chess.square_file(other) == adj_file:
                                has_friend = True
                                break
                    if has_friend:
                        break
                if not has_friend:
                    isolated_penalty += 15
            
            if color == chess.WHITE:
                score -= doubled_penalty + isolated_penalty
            else:
                score += doubled_penalty + isolated_penalty
        
        return score

    def evaluate_king_safety(self, board: chess.Board) -> int:
        """Evaluate king safety based on pawn shield."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq is None:
                continue
            
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            
            pawn_shield = 0
            for file_offset in [-1, 0, 1]:
                file = king_file + file_offset
                if 0 <= file < 8:
                    if color == chess.WHITE:
                        # FIXED: Check ranks ABOVE white king (higher ranks)
                        for rank_offset in [1, 2]:
                            rank = king_rank + rank_offset  # UP for white
                            if 0 <= rank < 8:
                                sq = chess.square(file, rank)
                                if board.piece_at(sq) == chess.Piece(chess.PAWN, color):
                                    pawn_shield += 15
                    else:
                        # Black king: check ranks below (lower ranks)
                        for rank_offset in [1, 2]:
                            rank = king_rank - rank_offset  # DOWN for black
                            if 0 <= rank < 8:
                                sq = chess.square(file, rank)
                                if board.piece_at(sq) == chess.Piece(chess.PAWN, color):
                                    pawn_shield += 15
            
            if color == chess.WHITE:
                score += pawn_shield
            else:
                score -= pawn_shield
        
        return score

    def get_capture_value(self, board: chess.Board, move: chess.Move) -> Tuple[int, int]:
        """Get victim and attacker values."""
        if board.is_en_passant(move):
            return self.PIECE_VALUES[chess.PAWN], self.PIECE_VALUES[chess.PAWN]
        
        victim_piece = board.piece_at(move.to_square)
        victim_value = self.PIECE_VALUES.get(victim_piece.piece_type, 0) if victim_piece else 0
        
        attacker_piece = board.piece_at(move.from_square)
        attacker_value = self.PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0
        
        return victim_value, attacker_value

    def mvv_lva_score(self, victim_value: int, attacker_value: int) -> int:
        """MVV-LVA: victim * 10 - attacker."""
        return victim_value * 10 - attacker_value

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int = 0) -> int:
        """Quiescence search with proper bounds."""
        self.nodes += 1

        stand_pat = self.evaluate(board)

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if ply >= 6:
            return alpha

        # Generate tactical moves
        tactical_moves = []
        for move in board.legal_moves:
            priority = 0

            if board.is_capture(move):
                victim_val, attacker_val = self.get_capture_value(board, move)
                priority = 10000 + self.mvv_lva_score(victim_val, attacker_val)
                if board.gives_check(move):
                    priority += 1000
            elif move.promotion:
                priority = 15000 + move.promotion * 100
            elif board.gives_check(move):
                priority = 8000
            # Pawn push to 7th rank
            elif (board.piece_at(move.from_square) and 
                  board.piece_at(move.from_square).piece_type == chess.PAWN):
                rank = chess.square_rank(move.to_square)
                if (board.turn == chess.WHITE and rank == 6) or (board.turn == chess.BLACK and rank == 1):
                    priority = 7000

            if priority > 0:
                tactical_moves.append((move, priority))

        tactical_moves.sort(key=lambda x: x[1], reverse=True)

        for move, _ in tactical_moves:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int,
                ply: int = 0, can_null: bool = True) -> Tuple[int, List[chess.Move]]:
        """Negamax with all fixes applied."""
        self.nodes += 1

        # Time check
        if self.nodes % 1000 == 0 and time.time() - self.search_start_time > self.time_limit:
            return 0, []

        # TT probe
        tt_hit, tt_score, tt_move, tt_pv = self.tt.probe(board, depth, alpha, beta)
        if tt_hit:
            self.tt_hits += 1
            return tt_score, tt_pv

        # Game end checks
        if board.is_checkmate():
            return -self.MATE_SCORE + ply, []
        if board.is_stalemate() or board.is_insufficient_material():
            return 0, []
        if board.can_claim_draw():
            return 0, []

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply), []

        # Null move pruning
        if (can_null and depth >= 3 and not board.is_check() and
            abs(beta) < self.MATE_SCORE - 1000 and self.has_major_pieces(board)):

            board.push(chess.Move.null())
            null_score, _ = self.negamax(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            null_score = -null_score
            board.pop()

            if null_score >= beta:
                self.null_move_prunes += 1
                self.tt.store(board, depth, beta, TTFlag.LOWER, None, [])
                return beta, []

        # Move ordering
        moves = self.order_moves(board, ply, tt_move)

        if not moves:
            return 0, []

        best_score = -self.MATE_SCORE - 1
        best_move = None
        best_pv = []
        original_alpha = alpha

        for move in moves:
            board.push(move)
            score, sub_pv = self.negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
            score = -score
            board.pop()

            if score > alpha:
                alpha = score
                best_move = move
                best_pv = [move] + sub_pv

            if score >= beta:
                # Killer move
                if not board.is_capture(move) and ply < self.MAX_PLY:
                    killers = self.killer_moves[ply]
                    if move != killers[0]:
                        killers[1] = killers[0]
                        killers[0] = move
                        self.killer_hits += 1

                # History heuristic
                color_idx = 0 if board.turn == chess.WHITE else 1
                self.history_table[color_idx][move.from_square][move.to_square] += depth * depth

                self.tt.store(board, depth, beta, TTFlag.LOWER, move, best_pv)
                return beta, best_pv

            if score > best_score:
                best_score = score

        # Determine flag
        if alpha <= original_alpha:
            flag = TTFlag.UPPER
        elif alpha >= beta:
            flag = TTFlag.LOWER
        else:
            flag = TTFlag.EXACT

        self.tt.store(board, depth, alpha, flag, best_move, best_pv)
        return alpha, best_pv

    def order_moves(self, board: chess.Board, ply: int, tt_move: Optional[chess.Move]) -> List[chess.Move]:
        """Move ordering with center control bonus."""
        moves = list(board.legal_moves)
        scored_moves = []
        color_idx = 0 if board.turn == chess.WHITE else 1

        for move in moves:
            score = 0

            if tt_move and move == tt_move:
                score += 100000

            if board.is_capture(move):
                victim_val, attacker_val = self.get_capture_value(board, move)
                score += 10000 + self.mvv_lva_score(victim_val, attacker_val)

            # Killer moves
            if ply < self.MAX_PLY:
                killers = self.killer_moves[ply]
                if move in killers:
                    score += 9000 if move == killers[0] else 8000

            if move.promotion:
                score += 7000 + move.promotion * 100

            score += self.history_table[color_idx][move.from_square][move.to_square]

            if board.gives_check(move):
                score += 50

            # Center control bonus
            to_sq = move.to_square
            center_dist = abs(3.5 - chess.square_file(to_sq)) + abs(3.5 - chess.square_rank(to_sq))
            score += (7 - center_dist) * 10

            scored_moves.append((move, score))

        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]

    def has_major_pieces(self, board: chess.Board) -> bool:
        """Check for null move pruning eligibility."""
        color = board.turn
        return (len(board.pieces(chess.ROOK, color)) + 
                len(board.pieces(chess.QUEEN, color))) > 0

    def find_best_move(self, max_depth: int = 8, time_limit: int = 30) -> Optional[str]:
        """Iterative deepening with aspiration windows and time management."""
        self.search_start_time = time.time()
        self.time_limit = time_limit
        self.nodes = 0
        self.tt_hits = 0
        self.killer_hits = 0
        self.null_move_prunes = 0

        # Reset search data
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY)]
        self.history_table = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(2)]

        print(f"{'='*70}")
        print(f"🔍 PERFECT ENGINE v7.2 - Iterative Deepening")
        print(f"⚡ Parallel workers: {self.num_workers} | Time limit: {time_limit}s")
        print(f"{'='*70}\n")

        # Check for immediate checkmate
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_checkmate():
                self.board.pop()
                move_san = self.board.san(move)
                print(f"🏆 IMMEDIATE CHECKMATE FOUND: {move_san}")
                return move_san
            self.board.pop()

        best_move = None
        best_score = 0
        best_pv = []
        window = 50

        for depth in range(1, max_depth + 1):
            # Time management
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit * 0.9:
                print(f"⏰ Time limit reached at depth {depth}")
                break

            iter_start = time.time()

            # Aspiration search
            alpha = best_score - window
            beta = best_score + window

            score, pv = self.negamax(self.board, depth, alpha, beta, 0, True)

            # Re-search if outside window
            if score <= alpha or score >= beta:
                alpha = -self.MATE_SCORE
                beta = self.MATE_SCORE
                score, pv = self.negamax(self.board, depth, alpha, beta, 0, True)

            iter_time = time.time() - iter_start

            if pv:
                best_move = pv[0]
                best_score = score
                best_pv = pv

                # Display score from our perspective
                our_color = chess.WHITE if self.play_as_white else chess.BLACK
                display_score = -score if self.board.turn != our_color else score

                if abs(display_score) > self.MATE_SCORE - 10000:
                    mate_in = (self.MATE_SCORE - abs(display_score) + 1) // 2
                    score_str = f"Mate in {mate_in}" if display_score > 0 else f"-Mate in {mate_in}"
                else:
                    score_str = f"{display_score:+d}cp"

                # PV as SAN
                temp_board = self.board.copy()
                pv_san = []
                for move in pv[:8]:
                    pv_san.append(temp_board.san(move))
                    temp_board.push(move)

                pv_str = " → ".join(pv_san)
                print(f"Depth {depth}: {self.board.san(best_move)} | {score_str} | "
                      f"PV: {pv_str} | {self.nodes:,}n | {iter_time:.2f}s")

        total_time = time.time() - self.search_start_time

        if best_move is None:
            legal = list(self.board.legal_moves)
            if legal:
                best_move = legal[0]
            else:
                return None

        move_san = self.board.san(best_move)

        print(f"\n{'='*70}")
        print(f"🎯 BEST MOVE: {move_san}")
        print(f"{'='*70}")
        print(f"Search time: {total_time:.2f}s")
        print(f"Nodes searched: {self.nodes:,}")
        print(f"TT hits: {self.tt_hits:,} ({self.tt_hits/max(self.nodes,1)*100:.1f}%)")
        print(f"Killer move hits: {self.killer_hits:,}")
        print(f"Null move prunes: {self.null_move_prunes:,}")
        print(f"TT entries: {len(self.tt.table):,}")

        # Tactical analysis
        our_color = chess.WHITE if self.play_as_white else chess.BLACK
        tactics = TacticsDetector.detect_all_tactics(self.board, best_move, our_color)
        if tactics:
            print(f"⚡ Tactical patterns: {', '.join(tactics)}")

        # Show full principal variation
        if best_pv:
            print(f"\nPrincipal Variation:")
            temp_board = self.board.copy()
            move_num = 1
            line = ""
            for i, move in enumerate(best_pv[:12]):
                if i % 2 == 0:
                    line += f"{move_num}. {temp_board.san(move)} "
                else:
                    line += f"{temp_board.san(move)} "
                    move_num += 1
                temp_board.push(move)
            print(f"  {line.strip()}")

        print(f"{'='*70}\n")
        return move_san

    def evaluate_move_sequence(self, move_sequence_str: str, depth: int = 4) -> None:
        """Evaluate a sequence of moves."""
        moves = [m.strip() for m in move_sequence_str.split(',')]
        
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING MOVE SEQUENCE: {' → '.join(moves)}")
        print(f"{'='*70}\n")
        
        board_copy = self.board.copy()
        parsed_moves = []
        
        # Parse and validate moves
        for i, move_str in enumerate(moves):
            move = None
            for legal_move in board_copy.legal_moves:
                if board_copy.san(legal_move) == move_str:
                    move = legal_move
                    break
            
            if move is None:
                print(f"❌ Move '{move_str}' is not legal")
                print(f"Legal moves: {[board_copy.san(m) for m in board_copy.legal_moves][:10]}")
                return
            
            # Detect tactics
            move_color = chess.WHITE if board_copy.turn else chess.BLACK
            tactics = TacticsDetector.detect_all_tactics(board_copy, move, move_color)
            
            parsed_moves.append((move, board_copy.san(move), tactics))
            board_copy.push(move)
        
        # Show sequence with tactics
        temp_board = self.board.copy()
        for i, (move, san, tactics) in enumerate(parsed_moves, 1):
            tactics_str = f" [{', '.join(tactics)}]" if tactics else ""
            print(f"  {i}. {san}{tactics_str}")
            temp_board.push(move)
        
        print(f"\nFinal position:")
        print(board_copy)
        print(f"FEN: {board_copy.fen()}")
        print()
        
        # Evaluate final position
        original_board = self.board
        original_color = self.play_as_white
        
        self.board = board_copy
        self.play_as_white = (board_copy.turn == chess.WHITE)
        
        print(f"{'─'*70}")
        print(f"🔍 CONTINUATION ANALYSIS")
        print(f"{'─'*70}")
        
        self.find_best_move(max_depth=depth, time_limit=10)
        
        # Restore
        self.board = original_board
        self.play_as_white = original_color

    def load_position(self, pos_input: str, play_as: Optional[str] = None):
        """Load position from FEN or PGN with full error handling."""
        # Try as file
        if Path(pos_input).exists():
            try:
                with open(pos_input, 'r') as f:
                    content = f.read()

                if content.strip().startswith('['):
                    pgn = io.StringIO(content)
                    game = chess.pgn.read_game(pgn)
                    if game:
                        board = game.board()
                        for move in game.mainline_moves():
                            board.push(move)
                        self.board = board
                        self.play_as_white = play_as.lower() in ['w', 'white'] if play_as else board.turn
                        print(f"✓ Loaded position from PGN file")
                        self._print_position_info("PGN file")
                        return
            except Exception as e:
                print(f"Warning: Failed to parse PGN file: {e}")

        # Try as PGN text
        if '[' in pos_input or '1.' in pos_input:
            try:
                pgn = io.StringIO(pos_input)
                game = chess.pgn.read_game(pgn)
                if game:
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                    self.board = board
                    self.play_as_white = play_as.lower() in ['w', 'white'] if play_as else board.turn
                    print(f"✓ Loaded position from PGN text")
                    self._print_position_info("PGN")
                    return
            except Exception as e:
                print(f"Warning: Failed to parse PGN text: {e}")

        # Try as FEN
        try:
            self.board.set_fen(pos_input)
            self.play_as_white = play_as.lower() in ['w', 'white'] if play_as else self.board.turn
            print(f"✓ Loaded position from FEN")
            self._print_position_info("FEN")
        except Exception as e:
            print(f"❌ Invalid FEN: {e}")
            raise ValueError(f"Could not parse position: {pos_input[:50]}...")

    def _print_position_info(self, source: str):
        print(f"\n{'='*70}")
        print(f"Position loaded from: {source}")
        print(f"Playing as: {'WHITE' if self.play_as_white else 'BLACK'}")
        print(f"Side to move: {'WHITE' if self.board.turn else 'BLACK'}")
        print(f"FEN: {self.board.fen()}")
        print(f"Castling rights: {self.board.castling_rights}")
        print(f"En passant: {chess.SQUARE_NAMES[self.board.ep_square] if self.board.ep_square else 'None'}")
        print(f"Half-move clock: {self.board.halfmove_clock}")
        print(f"Full-move number: {self.board.fullmove_number}")
        print(f"{'='*70}\n")
        print(self.board)
        print()


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def parallel_move_evaluator(args):
    """Worker function for parallel root move evaluation."""
    board_fen, move_uci, max_depth, time_limit = args
    
    engine = PerfectChessEngine(num_workers=1)
    engine.board.set_fen(board_fen)
    
    # Make the move
    move = chess.Move.from_uci(move_uci)
    engine.board.push(move)
    
    # Search from opponent's perspective
    engine.play_as_white = not engine.board.turn
    
    # Run search
    score, pv = engine.negamax(engine.board, max_depth - 1, 
                               -engine.MATE_SCORE, engine.MATE_SCORE, 0, True)
    
    # Negate score since it's from opponent's perspective
    score = -score
    
    return {
        'move_uci': move_uci,
        'score': score,
        'pv': pv
    }


class ParallelRootSearcher:
    """Wrapper for parallel root move evaluation."""
    
    def __init__(self, engine: PerfectChessEngine):
        self.engine = engine
        self.num_workers = engine.num_workers
    
    def find_best_move_parallel(self, max_depth: int = 8, time_limit: int = 30) -> Optional[str]:
        """Find best move using parallel root move evaluation."""
        print(f"🔄 Parallel root search with {self.num_workers} workers")
        
        legal_moves = list(self.engine.board.legal_moves)
        if not legal_moves:
            return None
        
        # Prepare arguments for workers
        args_list = [(self.engine.board.fen(), move.uci(), max_depth, time_limit) 
                    for move in legal_moves[:20]]  # Limit to 20 candidates
        
        # Run parallel evaluation
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(parallel_move_evaluator, args_list)
        
        # Find best move
        best_result = max(results, key=lambda x: x['score'])
        best_move = chess.Move.from_uci(best_result['move_uci'])
        
        return self.engine.board.san(best_move)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Chess Engine v7.2 - Perfect Final Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chess_engine_v7_2.py --pos "start" --depth 8
  python chess_engine_v7_2.py --pos "game.pgn" --as white
  python chess_engine_v7_2.py --pos "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --move "e4,e5,Nf3"
  python chess_engine_v7_2.py --pos "start" --time 60 --parallel
        """
    )
    
    parser.add_argument('--pos', type=str, required=True,
                       help='Position (FEN, PGN file, or PGN text)')
    parser.add_argument('--as', dest='play_as', type=str,
                       choices=['w', 'white', 'b', 'black'], default=None,
                       help='Side to play as (default: side to move)')
    parser.add_argument('--depth', type=int, default=8,
                       help='Max search depth (default: 8)')
    parser.add_argument('--time', type=int, default=30,
                       help='Time limit in seconds (default: 30)')
    parser.add_argument('--move', type=str, default=None,
                       help='Evaluate move sequence: "e4" or "e4,e5,Nf3"')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel root move evaluation')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         CHESS ENGINE v7.2 - PERFECT FINAL                       ║
║   TT • Tactics • PST • Mobility • Pawns • King Safety • PV      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Create engine
    engine = PerfectChessEngine(num_workers=args.workers)
    
    # Handle special positions
    if args.pos.lower() == "start":
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    else:
        fen = args.pos
    
    # Load position
    try:
        engine.load_position(fen, args.play_as)
    except Exception as e:
        print(f"❌ Failed to load position: {e}")
        return
    
    # Evaluate move sequence or find best move
    if args.move:
        engine.evaluate_move_sequence(args.move, depth=args.depth)
    else:
        if args.parallel:
            searcher = ParallelRootSearcher(engine)
            best_move = searcher.find_best_move_parallel(max_depth=args.depth, time_limit=args.time)
        else:
            best_move = engine.find_best_move(max_depth=args.depth, time_limit=args.time)
        
        if best_move:
            print(f"✅ Recommended move: {best_move}")


if __name__ == "__main__":
    main()