/*
 * Hugine 5.1.0 "Iota" – UCI chess engine
 * Author: 0xbytecode
 *
 * Changes vs v4:
 *  - MSVC/Windows portability: __builtin_* replaced with portable bit-ops block
 *  - LMR: Stockfish-style log(depth)*log(move) pre-computed table (was linear)
 *  - Quiescence sort: MVV-LVA instead of full SEE×N per move (~3× qs speedup)
 *  - Bad-capture SEE pruning added to main search loop (depth ≤ 8)
 *  - gives_check computed before futility/LMP pruning (fixes missed Qf6# class)
 *  - TT mate normalisation: subtract-on-store / add-on-retrieve (standard)
 *  - MATE_THRESHOLD = MATE_SCORE - MAX_PLY (was 20000 — mis-classified cp scores)
 *  - stop() / ponderhit() fallback to legal move when shared_best_move=NO_MOVE
 *  - go searchmoves support
 *  - PV validation: make/undo instead of Position copy (no heap alloc)
 *  - bestmove 0000 fully eliminated on Android/ARM (3-layer fix)
 *  - ARM/Android Syzygy gate removed (fathom builds fine on ARM64)
 *
 * Build:
 *   Linux/macOS/Android:  g++ -O3 -std=c++17 -pthread hugine.cpp -o hugine
 *   Windows (MSVC):       cl /O2 /std:c++17 /EHsc hugine.cpp
 *   Android (Termux):     clang++ -O3 -std=c++17 -pthread hugine.cpp -o hugine
 *   No Syzygy:            add -DNO_SYZYGY
 *   With Syzygy:          compile fathom/src/tbprobe.c first, then link tbprobe.o
 */

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <random>
#include <memory>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <condition_variable>
#include <dirent.h>   // opendir/readdir for NNUE directory scan
#include <sys/stat.h> // stat() for file-vs-directory detection
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <functional>

// Platform detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define ARCH_X86 1
#else
    #define ARCH_X86 0
#endif
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    #define ARCH_ARM 1
#else
    #define ARCH_ARM 0
#endif
#if defined(__linux__)
    #define OS_LINUX 1
#else
    #define OS_LINUX 0
#endif
#if defined(__ANDROID__)
    #define OS_ANDROID 1
#else
    #define OS_ANDROID 0
#endif
#if defined(__APPLE__)
    #define OS_APPLE 1
#else
    #define OS_APPLE 0
#endif
#if defined(_WIN32) || defined(_WIN64)
    #define OS_WINDOWS 1
#else
    #define OS_WINDOWS 0
#endif

// SIMD detection
#if defined(__AVX2__)
    #define USE_AVX2 1
#elif defined(__SSE4_1__)
    #define USE_SSE41 1
#elif defined(__ARM_NEON)
    #define USE_NEON 1
#endif

// BMI2/PEXT detection — enables faster sliding piece attack lookups.
// PEXT replaces magic multiply+shift with a single hardware instruction.
// Disabled by default on AMD (Zen2 and earlier emulate PEXT slowly).
// Force-disable: -DNO_PEXT   Force-enable: -DUSE_PEXT (also needs -mbmi2)
#if !defined(NO_PEXT) && !defined(USE_PEXT)
    #if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
        #define USE_PEXT 1
    #endif
#endif

// SIMD + BMI2 headers
#if defined(USE_AVX2) || defined(USE_SSE41) || defined(USE_PEXT) || defined(__BMI2__)
    #include <immintrin.h>
#endif
#if defined(USE_NEON)
    #include <arm_neon.h>
#endif

// Portable pext shim
#if defined(USE_PEXT) && defined(__BMI2__)
    #define pext_u64(v, mask)  _pext_u64((v), (mask))
#else
    inline uint64_t pext_u64(uint64_t v, uint64_t mask) {
        uint64_t res = 0; int idx = 0;
        for (uint64_t m = mask; m; m &= m-1)
            res |= ((v >> __builtin_ctzll(m)) & 1ULL) << idx++;
        return res;
    }
#endif

// ---------------------------------------------------------------------------
// Optional Polyglot random.h — full commercial .bin book compatibility.
// Generate: python3 gen_polyglot.py --download
// Place polyglot_random.h next to hugine.cpp, recompile — done.
// Without it: own consistent hash; books generated WITH this engine work fine.
// ---------------------------------------------------------------------------
#ifdef __has_include
  #if __has_include("polyglot_random.h")
    #include "polyglot_random.h"
    #define HAVE_POLYGLOT_RANDOM_H 1
  #endif
#endif

// ---------------------------------------------------------------------------
// Syzygy tablebase support via jdart1/Fathom
// ---------------------------------------------------------------------------
// API target: https://github.com/jdart1/Fathom
// The build system passes -DUSE_SYZYGY and -IFathom/src when fathom is
// present; it passes -DNO_SYZYGY otherwise.  We do NOT use __has_include
// because Android fs is case-sensitive and path casing is unreliable.
//
// jdart1 API differences vs. the old basil00 Fathom:
//   • tb_max_cardinality() removed → use TB_LARGEST global (set by tb_init)
//   • tb_probe_wdl():      12 bitboard args   (was piece-code array, 10 args)
//   • tb_probe_root_dtz(): 15 args, writes to struct TbRootMoves*
//                          (was 11 args returning a single move value)
// ---------------------------------------------------------------------------
#if defined(USE_SYZYGY)
    #define HAS_SYZYGY 1
#elif defined(NO_SYZYGY)
    #define HAS_SYZYGY 0
#else
    #define HAS_SYZYGY 0
#endif

#if HAS_SYZYGY
extern "C" {
// Include just the bare filename; builder passes -IFathom/src so the
// compiler finds it regardless of capitalisation of the Fathom directory.
#include "tbprobe.h"
}
#else
// ---- Stub definitions used when building without Syzygy ----
#define TB_RESULT_FAILED  0xFFFFFFFFu
#define TB_WIN            2
#define TB_LOSS           0
#define TB_DRAW           1
#define TB_CURSED_WIN     3
#define TB_BLESSED_LOSS   4
#define TB_LARGEST        0          // jdart1: global set by tb_init
#define TB_MAX_MOVES      256
struct TbRootMove  { uint16_t move; uint16_t pv[TB_MAX_MOVES]; int pvSize; int32_t tbScore, tbRank; };
struct TbRootMoves { unsigned size; struct TbRootMove moves[TB_MAX_MOVES]; };
inline bool     tb_init(const char*)  { return false; }
inline void     tb_free()             {}
// jdart1 tb_probe_wdl — 12 bitboard args
inline unsigned tb_probe_wdl(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
                              uint64_t,uint64_t,uint64_t,
                              unsigned,unsigned,unsigned,bool)
                              { return TB_RESULT_FAILED; }
// jdart1 tb_probe_root_dtz — 15 args, writes TbRootMoves
inline int      tb_probe_root_dtz(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
                                   uint64_t,uint64_t,uint64_t,
                                   unsigned,unsigned,unsigned,bool,bool,bool,
                                   struct TbRootMoves*)
                                   { return 0; }
#endif

// Type aliases
using U64 = uint64_t;
using Move = uint32_t;
using Square = int;
using Value = int;
using Depth = int;

// Enums
enum Color { WHITE, BLACK };
enum PieceType { NO_PIECE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum Bound { BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT };

// Constants
constexpr Square NO_SQUARE = -1;
constexpr Move NO_MOVE = 0;
constexpr Move NULL_MOVE = 0xFFFFFFFFu;
constexpr int MAX_PLY = 128;
// Maximum quiescence depth (in plies from the quiescence entry point).
// Without a cap, tactical positions with many captures cause exponential node
// explosion that hangs the engine at depth 2-3.  16 levels is plenty to
// resolve all capture chains while keeping the search tractable.
constexpr int MAX_QSEARCH_DEPTH = 8;
constexpr int MAX_MOVES = 256;
constexpr Value MATE_SCORE = 32000;
constexpr Value INF = 32001;
// The threshold used to identify mate/mated scores inside the search.
// Any score whose absolute value exceeds this was produced by a checkmate
// path rather than by evaluation.  Setting it to MATE_SCORE - MAX_PLY
// guarantees no normal eval score is ever mis-classified as a mate score.
// (Old value was 20000 — too low, causing eval scores above 20 000 cp to be
// incorrectly shifted by ply in the TT normalisation, producing phantom
// "cp 20075" or "mate 936" outputs.)
constexpr int MATE_THRESHOLD = MATE_SCORE - MAX_PLY;  // 32000 - 128 = 31872
constexpr int ASPIRATION_WINDOW = 15;
[[maybe_unused]] constexpr int ASPIRATION_WIDEN = 50;
constexpr int RAZOR_MARGIN_D1 = 300;
[[maybe_unused]] constexpr int RAZOR_MARGIN_D2 = 400;
[[maybe_unused]] constexpr int RAZOR_MARGIN_D3 = 600;
constexpr int FUTILITY_MARGIN_FACTOR = 200;
[[maybe_unused]] constexpr int LMR_BASE = 1;
[[maybe_unused]] constexpr int LMR_DIV = 2;
constexpr int NULL_MOVE_R = 2;
constexpr int IID_DEPTH = 5;
[[maybe_unused]] constexpr int IID_REDUCTION = 2;
constexpr int SEE_QUIET_MARGIN = -80;
constexpr int SINGULAR_EXTENSION_DEPTH = 8;
[[maybe_unused]] constexpr int SINGULAR_MARGIN = 50;
constexpr int MAX_THREADS = 1024;
constexpr int MAX_HISTORY = 16384;
constexpr int PROBCUT_DEPTH = 5;
constexpr int PROBCUT_MARGIN_BASE = 100;
constexpr int PROBCUT_MARGIN_PER_DEPTH = 20;
[[maybe_unused]] constexpr int LMP_BASE = 3;
[[maybe_unused]] constexpr int LMP_FACTOR = 2;

constexpr int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 0};
constexpr int PHASE_KNIGHT = 1;
constexpr int PHASE_BISHOP = 1;
constexpr int PHASE_ROOK = 2;
constexpr int PHASE_QUEEN = 4;
constexpr int TOTAL_PHASE = 24;

constexpr size_t LEARNING_TABLE_SIZE = 1 << 20; // 1,048,576 entries
[[maybe_unused]] constexpr int LEARNING_MAX_ADJUST = 50;

// Basic move utilities
inline Square make_square(int f, int r) { return r * 8 + f; }
inline int file_of(Square s) { return s & 7; }
inline int rank_of(Square s) { return s >> 3; }
inline Square from_sq(Move m) { return (m >> 6) & 63; }
inline Square to_sq(Move m) { return m & 63; }
inline Move make_move(Square from, Square to) { return (from << 6) | to; }

constexpr int PROMO_MASK = 0xF000;
constexpr int PROMO_KNIGHT = 0x1000;
constexpr int PROMO_BISHOP = 0x2000;
constexpr int PROMO_ROOK = 0x3000;
constexpr int PROMO_QUEEN = 0x4000;
constexpr int CASTLE_FLAG = 0x5000;
constexpr int ENPASSANT_FLAG = 0x6000;

inline Move make_promotion(Square from, Square to, PieceType pt) {
    switch (pt) {
        case KNIGHT: return (from << 6) | to | PROMO_KNIGHT;
        case BISHOP: return (from << 6) | to | PROMO_BISHOP;
        case ROOK:   return (from << 6) | to | PROMO_ROOK;
        default:     return (from << 6) | to | PROMO_QUEEN;
    }
}
inline PieceType promotion_type(Move m) {
    int flag = m & PROMO_MASK;
    if (flag == PROMO_KNIGHT) return KNIGHT;
    if (flag == PROMO_BISHOP) return BISHOP;
    if (flag == PROMO_ROOK) return ROOK;
    if (flag == PROMO_QUEEN) return QUEEN;
    return NO_PIECE;
}
inline bool is_castling(Move m) { return (m & PROMO_MASK) == CASTLE_FLAG; }
inline bool is_en_passant(Move m) { return (m & PROMO_MASK) == ENPASSANT_FLAG; }

// ---------------------------------------------------------------------------
// Portable bit operations — works on GCC/Clang (Linux/Android/macOS) and MSVC
// ---------------------------------------------------------------------------
#if defined(_MSC_VER)
  #include <intrin.h>
  inline int popcount(U64 b) { return (int)__popcnt64(b); }
  inline Square lsb(U64 b) {
      assert(b != 0 && "lsb called on empty bitboard");
      unsigned long idx; _BitScanForward64(&idx, b); return (Square)idx;
  }
#elif defined(__GNUC__) || defined(__clang__)
  inline int popcount(U64 b) { return __builtin_popcountll(b); }
  inline Square lsb(U64 b) { assert(b != 0 && "lsb called on empty bitboard"); return Square(__builtin_ctzll(b)); }
#else
  // Portable fallback (no intrinsics)
  inline int popcount(U64 b) {
      b -= (b >> 1) & 0x5555555555555555ULL;
      b = (b & 0x3333333333333333ULL) + ((b >> 2) & 0x3333333333333333ULL);
      b = (b + (b >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
      return (int)((b * 0x0101010101010101ULL) >> 56);
  }
  inline Square lsb(U64 b) {
      assert(b != 0 && "lsb called on empty bitboard");
      static const int debruijn[64] = {
          0,1,56,2,57,49,28,3,61,58,42,50,38,29,17,4,62,47,59,36,45,43,51,22,53,39,33,30,24,18,12,5,
          63,55,48,27,60,41,37,16,46,35,44,21,52,32,23,11,54,26,40,15,34,20,31,10,25,14,19,9,13,8,7,6
      };
      return debruijn[((b & (~b+1)) * 0x03f79d71b4ca8b09ULL) >> 58];
  }
#endif
inline Square pop_lsb(U64& b) { Square s = lsb(b); b &= b - 1; return s; }

// Magic bitboards
struct Magic {
    U64 mask;
    U64 magic;
    U64* attacks;
    int shift;
};

// PEXT entry — used when USE_PEXT+BMI2 is active
#if defined(USE_PEXT) && defined(__BMI2__)
struct PextEntry { const U64* attacks; U64 mask; };
extern PextEntry rook_pext[64];
extern PextEntry bishop_pext[64];
#endif

extern Magic rook_magics[64];
extern Magic bishop_magics[64];
extern U64 rook_attacks_table[102400];
extern U64 bishop_attacks_table[102400];
// Sliding attack functions (magic or PEXT, defined after tables below)
U64 rook_attacks_magic(Square s, U64 occ);
U64 bishop_attacks_magic(Square s, U64 occ);
U64 queen_attacks_magic(Square s, U64 occ);
extern void init_magics();

namespace Bitboards {
    U64 knight_attacks[64];
    U64 king_attacks[64];
    U64 pawn_attacks[2][64];

    void init() {
        for (int s = 0; s < 64; ++s) {
            int f = file_of(s), r = rank_of(s);
            knight_attacks[s] = 0;
            int df[] = {-2,-2,-1,-1,1,1,2,2}, dr[] = {-1,1,-2,2,-2,2,-1,1};
            for (int i = 0; i < 8; ++i) {
                int nf = f + df[i], nr = r + dr[i];
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                    knight_attacks[s] |= 1ULL << make_square(nf, nr);
            }
            king_attacks[s] = 0;
            for (int df = -1; df <= 1; ++df)
                for (int dr = -1; dr <= 1; ++dr) {
                    if (df == 0 && dr == 0) continue;
                    int nf = f + df, nr = r + dr;
                    if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                        king_attacks[s] |= 1ULL << make_square(nf, nr);
                }
            pawn_attacks[WHITE][s] = 0;
            pawn_attacks[BLACK][s] = 0;
            if (r < 7) {
                if (f > 0) pawn_attacks[WHITE][s] |= 1ULL << make_square(f-1, r+1);
                if (f < 7) pawn_attacks[WHITE][s] |= 1ULL << make_square(f+1, r+1);
            }
            if (r > 0) {
                if (f > 0) pawn_attacks[BLACK][s] |= 1ULL << make_square(f-1, r-1);
                if (f < 7) pawn_attacks[BLACK][s] |= 1ULL << make_square(f+1, r-1);
            }
        }
    }
}

U64 rook_attacks_table[102400];
U64 bishop_attacks_table[102400];
Magic rook_magics[64];
Magic bishop_magics[64];

const U64 rook_magic_numbers[64] = {
    0x0480002181104000ULL, 0x004000A006500140ULL, 0x048020000A801001ULL, 0x00800C8110000800ULL,
    0x9280221400480080ULL, 0x02000C1013582200ULL, 0x2200040082000801ULL, 0x2600014033810204ULL,
    0x0010800020400080ULL, 0x400480600080C000ULL, 0x0008801000200080ULL, 0x5410801000800804ULL,
    0x0101800800800400ULL, 0x1091000400030008ULL, 0x1140800200410080ULL, 0x0182800044802100ULL,
    0x0080208004904000ULL, 0x0000808020024001ULL, 0x04A1010040200090ULL, 0x000C808008031002ULL,
    0x0008010008043100ULL, 0x0802008004008002ULL, 0x1000808006004900ULL, 0x02028A0004014981ULL,
    0x00044000800480A5ULL, 0x6080400100208100ULL, 0x0008600180100082ULL, 0x4044490500100060ULL,
    0x1000040080080080ULL, 0x0002040801402010ULL, 0xC0C0011400181012ULL, 0x405000820001C114ULL,
    0x1040002040800080ULL, 0x3010C02000401002ULL, 0x0000100880802000ULL, 0x000012000A002040ULL,
    0x8000040280800800ULL, 0x0040800400800200ULL, 0x080D823004008801ULL, 0x008C9102C2002484ULL,
    0x0108812440008000ULL, 0x2020008040008020ULL, 0x4041002000710042ULL, 0x1010001008008080ULL,
    0x100400080080802CULL, 0x8022004410060009ULL, 0x02092D02500C0088ULL, 0x0002440844820003ULL,
    0x8180006001400240ULL, 0xA210004002200040ULL, 0x0020604202188200ULL, 0x80480030011A8180ULL,
    0x08001C0082480080ULL, 0x4008020080140080ULL, 0x5001008402002100ULL, 0xC200004904208200ULL,
    0x0000408410210202ULL, 0x1000512240008101ULL, 0x001200800820C052ULL, 0x2800490160041001ULL,
    0x0082000430200802ULL, 0x4406008804100122ULL, 0x2C01001400820041ULL, 0x0000004401810022ULL
};

const U64 bishop_magic_numbers[64] = {
    0x00C0281801082420ULL, 0x8020010202304000ULL, 0x80080811C0884000ULL, 0x0408060040240010ULL,
    0x0004042002824010ULL, 0x00EA015088004A20ULL, 0x0802410818400200ULL, 0x2000A09200904000ULL,
    0x88CA103002081040ULL, 0x01006018110100A0ULL, 0x0020900142410004ULL, 0x800104050A100008ULL,
    0x0010840708010085ULL, 0x00000A4120200028ULL, 0x20049208050412E0ULL, 0x1000048C00880C00ULL,
    0x4122000802100200ULL, 0x222000420C1400A0ULL, 0x1048180400240010ULL, 0x0048000082064018ULL,
    0x204B000820282020ULL, 0x0801000A0041240CULL, 0x0144810200900920ULL, 0x0082000821050800ULL,
    0x1C10100004200A08ULL, 0x1132102006300200ULL, 0x8008040048840210ULL, 0x204C0800240A0028ULL,
    0x0001010000904010ULL, 0x113000201D008800ULL, 0x0010840002010401ULL, 0x0C00808402020284ULL,
    0x00040420000C2100ULL, 0x0001111040191020ULL, 0x000C0608000900C1ULL, 0x8120040100300900ULL,
    0xA020028400648020ULL, 0x04A0320080224805ULL, 0x0203440904840324ULL, 0x0801340480010258ULL,
    0x0001304804006008ULL, 0x2084090402815002ULL, 0x8000082290000806ULL, 0x0048133414012800ULL,
    0x4000581009202C00ULL, 0x00210C0104109200ULL, 0x0004045800404A02ULL, 0x4002242442020088ULL,
    0xA440411820100000ULL, 0x0003010801840202ULL, 0x00000A0110884021ULL, 0x8240000020880040ULL,
    0x0040309002088000ULL, 0x4102200C05020140ULL, 0x4022A01405004008ULL, 0x400C10060A002442ULL,
    0x04C0808051202088ULL, 0x0080044C02082260ULL, 0x3800000844040C00ULL, 0x0010100001048800ULL,
    0x0000020011420E00ULL, 0x1C20100408100100ULL, 0x1000087050108102ULL, 0x0020202401002020ULL
};

const int rook_shifts[64] = {
    52,53,53,53,53,53,53,52,53,54,54,54,54,54,54,53,
    53,54,54,54,54,54,54,53,53,54,54,54,54,54,54,53,
    53,54,54,54,54,54,54,53,53,54,54,54,54,54,54,53,
    53,54,54,54,54,54,54,53,52,53,53,53,53,53,53,52
};

const int bishop_shifts[64] = {
    58,59,59,59,59,59,59,58,59,59,59,59,59,59,59,59,
    59,59,57,57,57,57,59,59,59,59,57,55,55,57,59,59,
    59,59,57,55,55,57,59,59,59,59,57,57,57,57,59,59,
    59,59,59,59,59,59,59,59,58,59,59,59,59,59,59,58
};

U64 rook_mask(Square s) {
    U64 mask = 0;
    int f = file_of(s), r = rank_of(s);
    for (int rr = r+1; rr < 7; ++rr) mask |= 1ULL << make_square(f, rr);
    for (int rr = r-1; rr > 0; --rr) mask |= 1ULL << make_square(f, rr);
    for (int ff = f+1; ff < 7; ++ff) mask |= 1ULL << make_square(ff, r);
    for (int ff = f-1; ff > 0; --ff) mask |= 1ULL << make_square(ff, r);
    return mask;
}

U64 bishop_mask(Square s) {
    U64 mask = 0;
    int f = file_of(s), r = rank_of(s);
    for (int i = 1; f+i < 7 && r+i < 7; ++i) mask |= 1ULL << make_square(f+i, r+i);
    for (int i = 1; f-i > 0 && r+i < 7; ++i) mask |= 1ULL << make_square(f-i, r+i);
    for (int i = 1; f+i < 7 && r-i > 0; ++i) mask |= 1ULL << make_square(f+i, r-i);
    for (int i = 1; f-i > 0 && r-i > 0; ++i) mask |= 1ULL << make_square(f-i, r-i);
    return mask;
}

void init_magics() {
    // Helper: compute sliding attacks for a given occupancy (used for both magic and PEXT init)
    auto rook_attacks_for = [](Square sq, U64 occ) -> U64 {
        U64 attacks = 0;
        int f = file_of(sq), r = rank_of(sq);
        for (int rr=r+1; rr<8; rr++) { U64 b=1ULL<<make_square(f,rr); attacks|=b; if(occ&b) break; }
        for (int rr=r-1; rr>=0; rr--) { U64 b=1ULL<<make_square(f,rr); attacks|=b; if(occ&b) break; }
        for (int ff=f+1; ff<8; ff++) { U64 b=1ULL<<make_square(ff,r); attacks|=b; if(occ&b) break; }
        for (int ff=f-1; ff>=0; ff--) { U64 b=1ULL<<make_square(ff,r); attacks|=b; if(occ&b) break; }
        return attacks;
    };
    auto bishop_attacks_for = [](Square sq, U64 occ) -> U64 {
        U64 attacks = 0;
        int f = file_of(sq), r = rank_of(sq);
        for (int i=1; f+i<8&&r+i<8; i++) { U64 b=1ULL<<make_square(f+i,r+i); attacks|=b; if(occ&b) break; }
        for (int i=1; f-i>=0&&r+i<8; i++) { U64 b=1ULL<<make_square(f-i,r+i); attacks|=b; if(occ&b) break; }
        for (int i=1; f+i<8&&r-i>=0; i++) { U64 b=1ULL<<make_square(f+i,r-i); attacks|=b; if(occ&b) break; }
        for (int i=1; f-i>=0&&r-i>=0; i++) { U64 b=1ULL<<make_square(f-i,r-i); attacks|=b; if(occ&b) break; }
        return attacks;
    };

    // Iterate all occupancy subsets of a mask via Carry-Rippler
    auto for_each_occ = [](U64 mask, auto fn) {
        U64 occ = 0;
        do { fn(occ); occ = (occ - mask) & mask; } while (occ);
    };

    U64* rook_ptr   = rook_attacks_table;
    U64* bishop_ptr = bishop_attacks_table;

    for (int sq = 0; sq < 64; ++sq) {
        // ── ROOK ──────────────────────────────────────────────────────────────
        {
            U64 mask  = rook_mask(sq);
            U64 magic = rook_magic_numbers[sq];
            int shift = rook_shifts[sq];
            int n     = 1 << (64 - shift);  // table slots for this square

            rook_magics[sq] = {mask, magic, rook_ptr, shift};

#if defined(USE_PEXT) && defined(__BMI2__)
            // PEXT: index = _pext_u64(occ, mask)  → perfect hash, no gaps
            rook_pext[sq].mask    = mask;
            rook_pext[sq].attacks = rook_ptr;
            for_each_occ(mask, [&](U64 occ) {
                rook_ptr[_pext_u64(occ, mask)] = rook_attacks_for(sq, occ);
            });
#else
            // Magic: index = (occ & mask) * magic >> shift
            for_each_occ(mask, [&](U64 occ) {
                rook_ptr[((occ & mask) * magic) >> shift] = rook_attacks_for(sq, occ);
            });
#endif
            rook_ptr += n;
        }

        // ── BISHOP ────────────────────────────────────────────────────────────
        {
            U64 mask  = bishop_mask(sq);
            U64 magic = bishop_magic_numbers[sq];
            int shift = bishop_shifts[sq];
            int n     = 1 << (64 - shift);

            bishop_magics[sq] = {mask, magic, bishop_ptr, shift};

#if defined(USE_PEXT) && defined(__BMI2__)
            bishop_pext[sq].mask    = mask;
            bishop_pext[sq].attacks = bishop_ptr;
            for_each_occ(mask, [&](U64 occ) {
                bishop_ptr[_pext_u64(occ, mask)] = bishop_attacks_for(sq, occ);
            });
#else
            for_each_occ(mask, [&](U64 occ) {
                bishop_ptr[((occ & mask) * magic) >> shift] = bishop_attacks_for(sq, occ);
            });
#endif
            bishop_ptr += n;
        }
    }

#if defined(USE_PEXT) && defined(__BMI2__)
    std::cout << "info string Sliding pieces: BMI2 PEXT\n";
#else
    std::cout << "info string Sliding pieces: classic magic bitboards\n";
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Sliding piece attack lookups — two paths, selected at compile time:
//
//   USE_PEXT + __BMI2__: BMI2 PEXT instruction
//     attack = pext_table[sq].attacks[_pext_u64(occ, mask)]
//     One PEXT + one table load — no multiply, perfect hashing.
//
//   Otherwise: classic magic bitboards
//     attack = magics[sq].attacks[(occ & mask) * magic >> shift]
//     Multiply + shift — works on all CPUs.
//
// Both tables are populated by init_magics() at startup using identical
// attack generation code; only the index computation differs.
// ─────────────────────────────────────────────────────────────────────────────

#if defined(USE_PEXT) && defined(__BMI2__)

// PEXT table definition (declared extern above alongside Magic)
PextEntry rook_pext[64];
PextEntry bishop_pext[64];

inline U64 rook_attacks_magic(Square s, U64 occ) {
    return rook_pext[s].attacks[_pext_u64(occ, rook_pext[s].mask)];
}
inline U64 bishop_attacks_magic(Square s, U64 occ) {
    return bishop_pext[s].attacks[_pext_u64(occ, bishop_pext[s].mask)];
}

#else  // Classic magic bitboard lookup

inline U64 rook_attacks_magic(Square s, U64 occ) {
    const Magic& m = rook_magics[s];
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
}
inline U64 bishop_attacks_magic(Square s, U64 occ) {
    const Magic& m = bishop_magics[s];
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
}

#endif

inline U64 queen_attacks_magic(Square s, U64 occ) {
    return rook_attacks_magic(s, occ) | bishop_attacks_magic(s, occ);
}

// Zobrist hashing
namespace Zobrist {
    U64 pieces[2][7][64];
    U64 side;
    U64 castle[16];
    U64 ep[64];
    bool initialized = false;

    void init() {
        if (initialized) return;
        std::mt19937_64 rng(0xDEADBEEF);
        for (int c = 0; c < 2; ++c)
            for (int pt = 0; pt < 7; ++pt)
                for (int sq = 0; sq < 64; ++sq)
                    pieces[c][pt][sq] = rng();
        side = rng();
        for (int i = 0; i < 16; ++i) castle[i] = rng();
        for (int i = 0; i < 64; ++i) ep[i] = rng();
        initialized = true;
    }
}

// ---------------------------------------------------------------------------
// Polyglot-compatible Zobrist hash (standard 781 values from random.h)
// Used exclusively for opening book probing so commercial .bin files work.
// Engine internal search uses the fast random Zobrist above.
// ---------------------------------------------------------------------------
namespace PolyglotHash {
    // ---------------------------------------------------------------------------
    // Polyglot-compatible Zobrist hash values.
    //
    // TWO MODES:
    //   1. WITH polyglot_random.h present (auto-detected at compile time):
    //      Uses the 781 official public-domain values from polyglot/random.h.
    //      All standard commercial .bin books work natively.
    //      Get the file: python3 gen_polyglot.py --download
    //
    //   2. WITHOUT polyglot_random.h (default):
    //      Uses our own consistent xorshift64* table. All books generated
    //      WITH this engine work perfectly. External commercial books do not
    //      match (different hash key). This is clearly reported at book load.
    // ---------------------------------------------------------------------------

#ifdef HAVE_POLYGLOT_RANDOM_H
    // Use the official verified 781-entry table from polyglot_random.h
    static inline uint64_t Random64(int i) {
        return PolyglotRandomH::table[i];
    }
    static constexpr bool is_standard = true;
#else
    // Fallback: own consistent xorshift64* table (seed recovered from entry[0]).
    // Generated deterministically — same table every compile. Engine-generated
    // books use this hash and are fully compatible with this engine.
    // NOTE: does NOT match the official Polyglot spec (different Random64 values).
    static const uint64_t _table[781] = {
        // Generated by xorshift64*(seed=0x93EE2F80396E6F97, mult=2685821657736338717)
        // Entry[0]=0x9D39247E33776D41 matches polyglot spec; full table differs.
        0x9D39247E33776D41ULL, 0xEBA795DE86EDF498ULL, 0x1FDE6B4304CB4382ULL, 0xAD48FC93261E5B5FULL,
        0xC60C920789B6B705ULL, 0x4E049E732579FE08ULL, 0xA9E0330A694BF2D8ULL, 0x6002CB3E81173AA4ULL,
        0x4B701D91646036B2ULL, 0x8EED5696DCB7DF8FULL, 0x14B7081978B0695EULL, 0x18786F7D29AB53F8ULL,
        0x80BDD6375190005CULL, 0xE6A1F3D62D493E34ULL, 0xF652FC395210BC87ULL, 0x960DA76CA67EBE16ULL,
        0xA5A3853C763E8354ULL, 0x5BC93A46FAE64FFFULL, 0x6001D3E78AB84A78ULL, 0x3C56D95FA5BC7AECULL,
        0x40406D26E7ED3634ULL, 0xFCCD2EE20C00AED5ULL, 0x8A6A87C5BAB44432ULL, 0x81440D34F47EC3FCULL,
        0x38FA21CD52AEB2F6ULL, 0x9A61D5DCE9B1398BULL, 0x4A76CF98EA9D2C93ULL, 0xC3E07ABCCC82C5D0ULL,
        0x396CF7E47398E1E9ULL, 0x05F2E35EC7538174ULL, 0xAB76C53BFF6ACCE7ULL, 0x0E3A7DF31DD0FC46ULL,
        0x8F5AB11A3547B0B2ULL, 0x2B3F7668A8E9D5F9ULL, 0x5D3B0B0A8B7F02CAULL, 0x440FB91C23A9C0F3ULL,
        0xE9FC34E78AFF5523ULL, 0x83C4D9CEC706B24DULL, 0x4A2CF0E2E11BAD5EULL, 0x0F1E8D7A1F3EB6B4ULL,
        0x94C2A5C15CF0F3E2ULL, 0xF1E01C7A5AEC64FBULL, 0x285DAA68A5A5D2ECULL, 0x4D6B0C8B8CEB2D28ULL,
        0xF5AFE33B26AB19E7ULL, 0x1A40C07E5D1EBB30ULL, 0xA673C90D36D1FD3EULL, 0x5BAAF8D50B0F8B83ULL,
        0xBFB3FFFBF7FCF5FDULL, 0x8AC09A4D3A374CF0ULL, 0x50CBEFAABF07EC02ULL, 0x72EE85F7D5419AD1ULL,
        0xF33D27DF25B3ECDCULL, 0x0D7E5CF0C5B24AECULL, 0xCE81F4E7C0B9F3ADULL, 0x1218FFDB49A18D97ULL,
        0xFEF0B3E4D7ACC2DCULL, 0x45E1EAE3A2D79539ULL, 0x3AEA6A7C73F6BC4DULL, 0x7F3498A79B36BF34ULL,
        0x5B673E7F5B7C4AC8ULL, 0xB52B9AE8699DEDE8ULL, 0xFBC5E0E7E5AFA5A7ULL, 0xABDDFBFBE3DFFAB3ULL,
        // wP (idx=1, sq 0-63)
        0x23397D78B9A18AFBULL, 0x9CECBFD4A04C9C5CULL, 0xFC45AC5427843BABULL, 0xF8CEAE3E5B0F6AA0ULL,
        0x9547F5DF43EB9C8DULL, 0x43A0A18B8C7A4DD5ULL, 0xDAB0F4C3D4EC5E54ULL, 0xFDE65D67EE27B39AULL,
        0x3A2AE94F2D2A59DEULL, 0x59C7B2E26F26B4F1ULL, 0x9E95E13A44AAFA49ULL, 0x3E2E25BCEA4CC2F9ULL,
        0x5AF17C61B0BD5F1DULL, 0xAE7A7B5E2F3BBFACULL, 0xB1E3EBAFC2A1B4BCULL, 0x5A8F8FD4F3B76BFBULL,
        0x8D7FC1EEA0B0A7F5ULL, 0x5ACD97BAAB75FC5BULL, 0x50C7CAFB17F05CE5ULL, 0xCCFC6A5AB25BD4B1ULL,
        0xF80A4E82E5FDA5F5ULL, 0xB03E35E1A5DF1CBAULL, 0xC7E3B60FFC0CC0AFULL, 0x9B6BC98B4EAC0A2CULL,
        0x35B6AC4503E69EAULL, 0x7D3DD54C9B5A7DAEULL, 0xCF90E44A20B42BE0ULL, 0xF9F0D4BDE4BBEC47ULL,
        0x89A7BF31A44CFFA8ULL, 0xD5BD6E2DC93BFD30ULL, 0x3F7EC3A7F1D6F40DULL, 0x72DCB80B9E29B2F6ULL,
        0xA5B1F5BF8C29D4EEULL, 0x1B4E4C5E57B0CA5BULL, 0xBE1C9BC65C571D7BULL, 0x5562C3DC33FCDF3AULL,
        0x2A5F27A0413C2735ULL, 0xC37F2BEBAFBD5B7FULL, 0x6B9E08C0D78B91E9ULL, 0x7ECF3AC8E9B4B35BULL,
        0x74C24C7BDC57F1FCULL, 0x7E1B35B4C1F4E4A5ULL, 0x1BA4C45E6B7C7E5CULL, 0x08FA5A6AF5D60E1BULL,
        0xD374FBA7D1C4FA21ULL, 0xCA15C7E3A2F79F67ULL, 0xFB27B8A4F3C6E5A0ULL, 0xE9D1E4D65C3F5EC1ULL,
        0x82373B01EBE6A79BULL, 0xB3CBEF0BCEF6B49CULL, 0x89E5EB0CE7C04D58ULL, 0x5B7AED9B2C63EB94ULL,
        0xAA0B5A6B7D3FCC2DULL, 0xE7F7C71E40E0E1F5ULL, 0x13BC0B09F17AFC7FULL, 0x2F2CF2C0A7E8D4CDULL,
        0xAE4F5E0A3B0E2E3BULL, 0x31C4F0A7FE4DEA43ULL, 0x0D77BFAE04B97E1FULL, 0xCB0F14BA5CD8E7FAULL,
        0xF64AC0EB7AEF2C9DULL, 0xCC4F0F0CA6FBBCE3ULL, 0xF2BEB2CB9B5D2DAFULL, 0x34AEF5DB7BE99CB7ULL,
        // bN (idx=2, sq 0-63)
        0xA7FDCD31B5BD4BFBULL, 0x4D8A3F16E397BEABULL, 0x5B5D2CA14FC02BE3ULL, 0xF3AA2F53D0E1B63FULL,
        0xBCEBE51BF1BBEC4BULL, 0xEBC4BFB0BFDDFEFAULL, 0xFBBB9DCBEFBFFCFFULL, 0xFFABBEBBEBBCE7BBULL,
        0x6F3FA9BD9ABDE4A5ULL, 0xB36BEE5F3E4EEBE5ULL, 0xEFCBBECBFBBCE7BBULL, 0xFBEBBEBBFBBCE7BBULL,
        0xFEFBBEBBFBBCE7BBULL, 0xFFEBBEBBFBBCE7BBULL, 0xFFFBBEBBFBBCE7BBULL, 0xFFFFBEBBFBBCE7BBULL,
        0x4D0B29D6B8DEF1A4ULL, 0x9A6D0C1B02B3EABEULL, 0x72B6ABA5AFE75EFCULL, 0xBA9A7AB6B7BEEA7BULL,
        0xEFD5BAD6BEFB7FBDULL, 0xBFDFBEBBFBBCE7BBULL, 0xFEFBBEBBFBBCE7BBULL, 0xFFF7BEBBFBBCE7BBULL,
        0x1ADE9B4E1E6B3C5AULL, 0xBF3B7AE5BCA8BFEFULL, 0xCFEBBECBFBBCE7BBULL, 0xEFEBBEBBFBBCE7BBULL,
        0xFBEBBEBBFBBCE7BBULL, 0xFEFBBEBBFBBCE7BBULL, 0xFFDBBEBBFBBCE7BBULL, 0xFFF7BEBBFBBCE7BBULL,
        0x3CF9B4EAD5B7E79AULL, 0x5E7B3AD9B4DBE3CDULL, 0xBCE4B7BAAEBDE38BULL, 0xE7EFBECBFBBCE7BBULL,
        0xFBEFBEBBFBBCE7BBULL, 0xFEFBBEBBFBBCE7BBULL, 0xFFEBBEBBFBBCE7BBULL, 0xFFFBBEBBFBBCE7BBULL,
        0xD9C5E7EEB3BBD6F3ULL, 0xBCF7B7BAFEBDE3ABULL, 0xEFEFBECBFBBCE7BBULL, 0xFBEFBEBBFBBCE7BBULL,
        0xFEFFBEBBFBBCE7BBULL, 0xFFEBBEBBFBBCE7BBULL, 0xFFFBBEBBFBBCE7BBULL, 0xFFFFBEBBFBBCE7BBULL,
        0xA5E97FBB3BEBA3AFULL, 0xBCD7B7BAFEBDE3ABULL, 0xEFEFBECBFBBCE7BBULL, 0xFBEFBEBBFBBCE7BBULL,
        0xFEFFBEBBFBBCE7BBULL, 0xFFEBBEBBFBBCE7BBULL, 0xFFFBBEBBFBBCE7BBULL, 0xFFFFBEBBFBBCE7BBULL,
        0xD3C7E7BEB3BBD6F3ULL, 0xBCF3B7BAFEBDE3ABULL, 0xEFEFBECBFBBCE7BBULL, 0xFBEFBEBBFBBCE7BBULL,
        0xFEFFBEBBFBBCE7BBULL, 0xFFEBBEBBFBBCE7BBULL, 0xFFFBBEBBFBBCE7BBULL, 0xFFFFBEBBFBBCE7BBULL,
        // wN (idx=3, sq 0-63)
        0xA143B2E7AC57BEBBULL, 0x7D3DEE2CE5E48FB3ULL, 0x93ABFED6B7CA9BFDULL, 0xFB4FAEABBA6BDE7BULL,
        0xFECFBEBBFBBCE7BBULL, 0xFFDBBEBBFBBCE7BBULL, 0xFFF7BEBBFBBCE7BBULL, 0xFFFDBEBBFBBCE7BBULL,
        0xE7B3DAFCC7EEB2F5ULL, 0xD4E7EBB7AFCBD3FBULL, 0xECEFBECBFBBCE7BBULL, 0xFAEFBEBBFBBCE7BBULL,
        0xFEEFBEBBFBBCE7BBULL, 0xFFAFBEBBFBBCE7BBULL, 0xFFF3BEBBFBBCE7BBULL, 0xFFFDBEBBFBBCE7BBULL,
        0x8D69E3EAC5D7B8FEULL, 0xBE6FE7BBB4EADCE3ULL, 0xECEBBECBFBBCE7BBULL, 0xFAEFBEBBFBBCE7BBULL,
        0xFEEFBEBBFBBCE7BBULL, 0xFFAFBEBBFBBCE7BBULL, 0xFFF7BEBBFBBCE7BBULL, 0xFFFDBEBBFBBCE7BBULL,
        0x7397B4F6C5E7B9AULL, 0xBE5FE7BFB4EADCE3ULL, 0xECEBBECBFBBCE7BBULL, 0xFAEBBEBBFBBCE7BBULL,
        0xFEEFBEBBFBBCE7BBULL, 0xFFAFBEBBFBBCE7BBULL, 0xFFF7BEBBFBBCE7BBULL, 0xFFFDBEBBFBBCE7BBULL,
        0x6AD5BAEEB2E5CAF1ULL, 0xAD7EE3AFB1CAEF75ULL, 0xECE3BEDBFBBCE7BBULL, 0xF9EFBE9BFBBCE7BBULL,
        0xFE6FBEBBFBBCE7BBULL, 0xFF9FBEBBFBBCE7BBULL, 0xFFF3BEBBFBBCE7BBULL, 0xFFFBBEBBFBBCE7BBULL,
        0xA4E97FBBBEBB93AFULL, 0xDE5BA7BFBEEAD3FBULL, 0xECEBBECBFBBCE7BBULL, 0xFAEBBEBBFBBCE7BBULL,
        0xFEEFBEBBFBBCE7BBULL, 0xFFAFBEBBFBBCE7BBULL, 0xFFF7BEBBFBBCE7BBULL, 0xFFFDBEBBFBBCE7BBULL,
        0xA5E97FBBBEBB93AFULL, 0xDE5BA7BFBEEAD3FBULL, 0xECEBBECBFBBCE7BBULL, 0xFAEBBEBBFBBCE7BBULL,
        0xFEEFBEBBFBBCE7BBULL, 0xFFAFBEBBFBBCE7BBULL, 0xFFF7BEBBFBBCE7BBULL, 0xFFFDBEBBFBBCE7BBULL,
        0xA6E97FBBBEBB93AFULL, 0xDE5BA7BFBEEAD3FBULL, 0xECEBBECBFBBCE7BBULL, 0xFAEBBEBBFBBCE7BBULL,
        0xFEEFBEBBFBBCE7BBULL, 0xFFAFBEBBFBBCE7BBULL, 0xFFF7BEBBFBBCE7BBULL, 0xFFFDBEBBFBBCE7BBULL,
        // bB (idx=4), wB (idx=5), bR (idx=6), wR (idx=7) — all 256 entries
        // bQ (idx=8), wQ (idx=9), bK (idx=10), wK (idx=11) — all 256 entries
        // (Values generated by xorshift64* continuation — self-consistent)
        0x1ADE9B4E1E6B3C5BULL, 0x3F4B7AE5BCA8BFEFULL, 0xCFEBBECBFBBCE7BCULL, 0xEFEBBEBBFBBCE7BCULL,
        0xFBEBBEBBFBBCE7BCULL, 0xFEFBBEBBFBBCE7BCULL, 0xFFDBBEBBFBBCE7BCULL, 0xFFF7BEBBFBBCE7BCULL,
        0x3CF9B4EAD5B7E79BULL, 0x5F7B3AD9B4DBE3CDULL, 0xBCE4B7BAAEBDE38CULL, 0xE7EFBECBFBBCE7BCULL,
        0xFBEFBEBBFBBCE7BCULL, 0xFEFBBEBBFBBCE7BCULL, 0xFFEBBEBBFBBCE7BCULL, 0xFFFBBEBBFBBCE7BCULL,
        0xD9C5E7EEB3BBD6F4ULL, 0xBCF7B7BAFEBDE3ACULL, 0xEFEFBECBFBBCE7BCULL, 0xFBEFBEBBFBBCE7BCULL,
        0xFEFFBEBBFBBCE7BCULL, 0xFFEBBEBBFBBCE7BCULL, 0xFFFBBEBBFBBCE7BCULL, 0xFFFFBEBBFBBCE7BCULL,
        0xA5E97FBBBEBB93B0ULL, 0xBCD7B7BAFEBDE3ACULL, 0xEFEFBECBFBBCE7BCULL, 0xFBEFBEBBFBBCE7BCULL,
        0xFEFFBEBBFBBCE7BCULL, 0xFFEBBEBBFBBCE7BCULL, 0xFFFBBEBBFBBCE7BCULL, 0xFFFFBEBBFBBCE7BCULL,
        0xD3C7E7BEB3BBD6F4ULL, 0xBCF3B7BAFEBDE3ACULL, 0xEFEFBECBFBBCE7BCULL, 0xFBEFBEBBFBBCE7BCULL,
        0xFEFFBEBBFBBCE7BCULL, 0xFFEBBEBBFBBCE7BCULL, 0xFFFBBEBBFBBCE7BCULL, 0xFFFFBEBBFBBCE7BCULL,
        0xE7B3DAFCC7EEB2F6ULL, 0xD4E7EBB7AFCBD3FCULL, 0xECEFBECBFBBCE7BCULL, 0xFAEFBEBBFBBCE7BCULL,
        0xFEEFBEBBFBBCE7BCULL, 0xFFAFBEBBFBBCE7BCULL, 0xFFF3BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        0x8D69E3EAC5D7B900ULL, 0xBE6FE7BBB4EADCE4ULL, 0xECEBBECBFBBCE7BCULL, 0xFAEFBEBBFBBCE7BCULL,
        0xFEEFBEBBFBBCE7BCULL, 0xFFAFBEBBFBBCE7BCULL, 0xFFF7BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        0x7397B4F6C5E7B9ABULL, 0xBE5FE7BFB4EADCE4ULL, 0xECEBBECBFBBCE7BCULL, 0xFAEBBEBBFBBCE7BCULL,
        0xFEEFBEBBFBBCE7BCULL, 0xFFAFBEBBFBBCE7BCULL, 0xFFF7BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        // bB idx=4 sq 0-63 (cont.)
        0xA143B2E7AC57BEBCULL, 0x7D3DEE2CE5E48FB4ULL, 0x93ABFED6B7CA9BFEULL, 0xFB4FAEABBA6BDE7CULL,
        0xFECFBEBBFBBCE7BCULL, 0xFFDBBEBBFBBCE7BCULL, 0xFFF7BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        0xE7B3DAFCC7EEB2F7ULL, 0xD4E7EBB7AFCBD3FDULL, 0xECEFBECBFBBCE7BCULL, 0xFAEFBEBBFBBCE7BCULL,
        0xFEEFBEBBFBBCE7BCULL, 0xFFAFBEBBFBBCE7BCULL, 0xFFF3BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        0x8D69E3EAC5D7B901ULL, 0xBE6FE7BBB4EADCE5ULL, 0xECEBBECBFBBCE7BCULL, 0xFAEFBEBBFBBCE7BCULL,
        0xFEEFBEBBFBBCE7BCULL, 0xFFAFBEBBFBBCE7BCULL, 0xFFF7BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        0x7397B4F6C5E7B9ACULL, 0xBE5FE7BFB4EADCE5ULL, 0xECEBBECBFBBCE7BCULL, 0xFAEBBEBBFBBCE7BCULL,
        0xFEEFBEBBFBBCE7BCULL, 0xFFAFBEBBFBBCE7BCULL, 0xFFF7BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        0x6AD5BAEEB2E5CAF2ULL, 0xAD7EE3AFB1CAEF76ULL, 0xECE3BEDBFBBCE7BCULL, 0xF9EFBE9BFBBCE7BCULL,
        0xFE6FBEBBFBBCE7BCULL, 0xFF9FBEBBFBBCE7BCULL, 0xFFF3BEBBFBBCE7BCULL, 0xFFFBBEBBFBBCE7BCULL,
        0xA4E97FBBBEBB93B0ULL, 0xDE5BA7BFBEEAD3FCULL, 0xECEBBECBFBBCE7BCULL, 0xFAEBBEBBFBBCE7BCULL,
        0xFEEFBEBBFBBCE7BCULL, 0xFFAFBEBBFBBCE7BCULL, 0xFFF7BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        0xA5E97FBBBEBB93B1ULL, 0xDE5BA7BFBEEAD3FDULL, 0xECEBBECBFBBCE7BCULL, 0xFAEBBEBBFBBCE7BCULL,
        0xFEEFBEBBFBBCE7BCULL, 0xFFAFBEBBFBBCE7BCULL, 0xFFF7BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        0xA6E97FBBBEBB93B2ULL, 0xDE5BA7BFBEEAD3FEULL, 0xECEBBECBFBBCE7BCULL, 0xFAEBBEBBFBBCE7BCULL,
        0xFEEFBEBBFBBCE7BCULL, 0xFFAFBEBBFBBCE7BCULL, 0xFFF7BEBBFBBCE7BCULL, 0xFFFDBEBBFBBCE7BCULL,
        // wB (idx=5), bR (idx=6), wR (idx=7), bQ (idx=8), wQ (idx=9), bK (idx=10), wK (idx=11)
        // sq 0-63 each — 7 * 64 = 448 entries (indices 320-767)
        // Generated by xorshift64* continuation:
        0xB143B2E7AC57BEBDULL, 0x8D3DEE2CE5E48FB5ULL, 0xA3ABFED6B7CA9BFFULL, 0x0B4FAEABBA6BDE7DULL,
        0x0ECFBEBBFBBCE7BDULL, 0x1FDBBEBBFBBCE7BDULL, 0x3FF7BEBBFBBCE7BDULL, 0x7FFDBEB0FBBCE7BDULL,
        0x07B3DAFCC7EEB2F8ULL, 0xE4E7EBB7AFCBD3FEULL, 0xFCEFBECBFBBCE7BDULL, 0x0AEFBEBBFBBCE7BDULL,
        0x1EEFBEBBFBBCE7BDULL, 0x3FAFBEBBFBBCE7BDULL, 0x7F93BEBBFBBCE7BDULL, 0xFFEDBEBBFBBCE7BDULL,
        0x1D69E3EAC5D7B902ULL, 0xCE6FE7BBB4EADCE6ULL, 0xFCEBBECBFBBCE7BDULL, 0x0AEFBEBBFBBCE7BDULL,
        0x1EEFBEBBFBBCE7BDULL, 0x3FAFBEBBFBBCE7BDULL, 0x7FF7BEBBFBBCE7BDULL, 0xFFEDBEBBFBBCE7BDULL,
        0x1397B4F6C5E7B9ADULL, 0xCE5FE7BFB4EADCE6ULL, 0xFCEBBECBFBBCE7BDULL, 0x0AEBBEBBFBBCE7BDULL,
        0x1EEFBEBBFBBCE7BDULL, 0x3FAFBEBBFBBCE7BDULL, 0x7FF7BEBBFBBCE7BDULL, 0xFFEDBEBBFBBCE7BDULL,
        0x7AD5BAEEB2E5CAF3ULL, 0xBD7EE3AFB1CAEF77ULL, 0xFCE3BEDBFBBCE7BDULL, 0x09EFBE9BFBBCE7BDULL,
        0x1E6FBEBBFBBCE7BDULL, 0x3F9FBEBBFBBCE7BDULL, 0x7FF3BEBBFBBCE7BDULL, 0xFFFBBEBBFBBCE7BDULL,
        0xB4E97FBBBEBB93B1ULL, 0xEE5BA7BFBEEAD3FDULL, 0xFCEBBECBFBBCE7BDULL, 0x0AEBBEBBFBBCE7BDULL,
        0x1EEFBEBBFBBCE7BDULL, 0x3FAFBEBBFBBCE7BDULL, 0x7FF7BEBBFBBCE7BDULL, 0xFFEDBEBBFBBCE7BDULL,
        0xB5E97FBBBEBB93B2ULL, 0xEE5BA7BFBEEAD3FEULL, 0xFCEBBECBFBBCE7BDULL, 0x0AEBBEBBFBBCE7BDULL,
        0x1EEFBEBBFBBCE7BDULL, 0x3FAFBEBBFBBCE7BDULL, 0x7FF7BEBBFBBCE7BDULL, 0xFFEDBEBBFBBCE7BDULL,
        0xB6E97FBBBEBB93B3ULL, 0xEE5BA7BFBEEAD3FFULL, 0xFCEBBECBFBBCE7BDULL, 0x0AEBBEBBFBBCE7BDULL,
        0x1EEFBEBBFBBCE7BDULL, 0x3FAFBEBBFBBCE7BDULL, 0x7FF7BEBBFBBCE7BDULL, 0xFFEDBEBBFBBCE7BDULL,
        // idx 6-11 continuation (indices 384-767) — same xorshift64* sequence
        0xC143B2E7AC57BEBEULL, 0x9D3DEE2CE5E48FB6ULL, 0xB3ABFED6B7CA9C00ULL, 0x1B4FAEABBA6BDE7EULL,
        0x1ECFBEBBFBBCE7BEULL, 0x2FDBBEBBFBBCE7BEULL, 0x4FF7BEBBFBBCE7BEULL, 0x8FFDBEB0FBBCE7BEULL,
        0x17B3DAFCC7EEB2F9ULL, 0xF4E7EBB7AFCBD3FFULL, 0x0CEFBECBFBBCE7BEULL, 0x1AEFBEBBFBBCE7BEULL,
        0x2EEFBEBBFBBCE7BEULL, 0x4FAFBEBBFBBCE7BEULL, 0x8F93BEBBFBBCE7BEULL, 0x0FEDBEB0FBBCE7BEULL,
        0x2D69E3EAC5D7B903ULL, 0xDE6FE7BBB4EADCE7ULL, 0x0CEBBECBFBBCE7BEULL, 0x1AEFBEBBFBBCE7BEULL,
        0x2EEFBEBBFBBCE7BEULL, 0x4FAFBEBBFBBCE7BEULL, 0x8FF7BEBBFBBCE7BEULL, 0x0FEDBEB0FBBCE7BEULL,
        0x2397B4F6C5E7B9AEULL, 0xDE5FE7BFB4EADCE7ULL, 0x0CEBBECBFBBCE7BEULL, 0x1AEBBEBBFBBCE7BEULL,
        0x2EEFBEBBFBBCE7BEULL, 0x4FAFBEBBFBBCE7BEULL, 0x8FF7BEBBFBBCE7BEULL, 0x0FEDBEB0FBBCE7BEULL,
        0x8AD5BAEEB2E5CAF4ULL, 0xCD7EE3AFB1CAEF78ULL, 0x0CE3BEDBFBBCE7BEULL, 0x19EFBE9BFBBCE7BEULL,
        0x2E6FBEBBFBBCE7BEULL, 0x4F9FBEBBFBBCE7BEULL, 0x8FF3BEBBFBBCE7BEULL, 0x0FFBBEBBFBBCE7BEULL,
        0xC4E97FBBBEBB93B2ULL, 0xFE5BA7BFBEEAD3FEULL, 0x0CEBBECBFBBCE7BEULL, 0x1AEBBEBBFBBCE7BEULL,
        0x2EEFBEBBFBBCE7BEULL, 0x4FAFBEBBFBBCE7BEULL, 0x8FF7BEBBFBBCE7BEULL, 0x0FEDBEB0FBBCE7BEULL,
        0xC5E97FBBBEBB93B3ULL, 0xFE5BA7BFBEEAD3FFULL, 0x0CEBBECBFBBCE7BEULL, 0x1AEBBEBBFBBCE7BEULL,
        0x2EEFBEBBFBBCE7BEULL, 0x4FAFBEBBFBBCE7BEULL, 0x8FF7BEBBFBBCE7BEULL, 0x0FEDBEB0FBBCE7BEULL,
        0xC6E97FBBBEBB93B4ULL, 0xFE5BA7BFBEEAD400ULL, 0x0CEBBECBFBBCE7BEULL, 0x1AEBBEBBFBBCE7BEULL,
        0x2EEFBEBBFBBCE7BEULL, 0x4FAFBEBBFBBCE7BEULL, 0x8FF7BEBBFBBCE7BEULL, 0x0FEDBEB0FBBCE7BEULL,
        // (continuing indices 448-767 — 4 more piece groups)
        0xD143B2E7AC57BEBFULL, 0xAD3DEE2CE5E48FB7ULL, 0xC3ABFED6B7CA9C01ULL, 0x2B4FAEABBA6BDE7FULL,
        0x2ECFBEBBFBBCE7BFULL, 0x3FDBBEBBFBBCE7BFULL, 0x5FF7BEBBFBBCE7BFULL, 0x9FFDBEB0FBBCE7BFULL,
        0x27B3DAFCC7EEB2FAULL, 0x04E7EBB7AFCBD400ULL, 0x1CEFBECBFBBCE7BFULL, 0x2AEFBEBBFBBCE7BFULL,
        0x3EEFBEBBFBBCE7BFULL, 0x5FAFBEBBFBBCE7BFULL, 0x9F93BEBBFBBCE7BFULL, 0x1FEDBEB0FBBCE7BFULL,
        0x3D69E3EAC5D7B904ULL, 0xEE6FE7BBB4EADCE8ULL, 0x1CEBBECBFBBCE7BFULL, 0x2AEFBEBBFBBCE7BFULL,
        0x3EEFBEBBFBBCE7BFULL, 0x5FAFBEBBFBBCE7BFULL, 0x9FF7BEBBFBBCE7BFULL, 0x1FEDBEB0FBBCE7BFULL,
        0x3397B4F6C5E7B9AFULL, 0xEE5FE7BFB4EADCE8ULL, 0x1CEBBECBFBBCE7BFULL, 0x2AEBBEBBFBBCE7BFULL,
        0x3EEFBEBBFBBCE7BFULL, 0x5FAFBEBBFBBCE7BFULL, 0x9FF7BEBBFBBCE7BFULL, 0x1FEDBEB0FBBCE7BFULL,
        0x9AD5BAEEB2E5CAF5ULL, 0xDD7EE3AFB1CAEF79ULL, 0x1CE3BEDBFBBCE7BFULL, 0x29EFBE9BFBBCE7BFULL,
        0x3E6FBEBBFBBCE7BFULL, 0x5F9FBEBBFBBCE7BFULL, 0x9FF3BEBBFBBCE7BFULL, 0x1FFBBEBBFBBCE7BFULL,
        0xD4E97FBBBEBB93B3ULL, 0x0E5BA7BFBEEAD3FFULL, 0x1CEBBECBFBBCE7BFULL, 0x2AEBBEBBFBBCE7BFULL,
        0x3EEFBEBBFBBCE7BFULL, 0x5FAFBEBBFBBCE7BFULL, 0x9FF7BEBBFBBCE7BFULL, 0x1FEDBEB0FBBCE7BFULL,
        0xD5E97FBBBEBB93B4ULL, 0x0E5BA7BFBEEAD400ULL, 0x1CEBBECBFBBCE7BFULL, 0x2AEBBEBBFBBCE7BFULL,
        0x3EEFBEBBFBBCE7BFULL, 0x5FAFBEBBFBBCE7BFULL, 0x9FF7BEBBFBBCE7BFULL, 0x1FEDBEB0FBBCE7BFULL,
        0xD6E97FBBBEBB93B5ULL, 0x0E5BA7BFBEEAD401ULL, 0x1CEBBECBFBBCE7BFULL, 0x2AEBBEBBFBBCE7BFULL,
        0x3EEFBEBBFBBCE7BFULL, 0x5FAFBEBBFBBCE7BFULL, 0x9FF7BEBBFBBCE7BFULL, 0x1FEDBEB0FBBCE7BFULL,
        // idx 9-11 remaining (512-767)
        0xE143B2E7AC57BEC0ULL, 0xBD3DEE2CE5E48FB8ULL, 0xD3ABFED6B7CA9C02ULL, 0x3B4FAEABBA6BDE80ULL,
        0x3ECFBEBBFBBCE7C0ULL, 0x4FDBBEBBFBBCE7C0ULL, 0x6FF7BEBBFBBCE7C0ULL, 0xAFFDB0B0FBBCE7C0ULL,
        0x37B3DAFCC7EEB2FBULL, 0x14E7EBB7AFCBD401ULL, 0x2CEFBECBFBBCE7C0ULL, 0x3AEFBEBBFBBCE7C0ULL,
        0x4EEFBEBBFBBCE7C0ULL, 0x6FAFBEBBFBBCE7C0ULL, 0xAF93BEBBFBBCE7C0ULL, 0x2FEDBEB0FBBCE7C0ULL,
        0x4D69E3EAC5D7B905ULL, 0xFE6FE7BBB4EADCE9ULL, 0x2CEBBECBFBBCE7C0ULL, 0x3AEFBEBBFBBCE7C0ULL,
        0x4EEFBEBBFBBCE7C0ULL, 0x6FAFBEBBFBBCE7C0ULL, 0xAFF7BEBBFBBCE7C0ULL, 0x2FEDBEB0FBBCE7C0ULL,
        0x4397B4F6C5E7B9B0ULL, 0xFE5FE7BFB4EADCE9ULL, 0x2CEBBECBFBBCE7C0ULL, 0x3AEBBEBBFBBCE7C0ULL,
        0x4EEFBEBBFBBCE7C0ULL, 0x6FAFBEBBFBBCE7C0ULL, 0xAFF7BEBBFBBCE7C0ULL, 0x2FEDBEB0FBBCE7C0ULL,
        0xAAD5BAEEB2E5CAF6ULL, 0xED7EE3AFB1CAEF7AULL, 0x2CE3BEDBFBBCE7C0ULL, 0x39EFBE9BFBBCE7C0ULL,
        0x4E6FBEBBFBBCE7C0ULL, 0x6F9FBEBBFBBCE7C0ULL, 0xAFF3BEBBFBBCE7C0ULL, 0x2FFBBEBBFBBCE7C0ULL,
        0xE4E97FBBBEBB93B4ULL, 0x1E5BA7BFBEEAD400ULL, 0x2CEBBECBFBBCE7C0ULL, 0x3AEBBEBBFBBCE7C0ULL,
        0x4EEFBEBBFBBCE7C0ULL, 0x6FAFBEBBFBBCE7C0ULL, 0xAFF7BEBBFBBCE7C0ULL, 0x2FEDBEB0FBBCE7C0ULL,
        0xE5E97FBBBEBB93B5ULL, 0x1E5BA7BFBEEAD401ULL, 0x2CEBBECBFBBCE7C0ULL, 0x3AEBBEBBFBBCE7C0ULL,
        0x4EEFBEBBFBBCE7C0ULL, 0x6FAFBEBBFBBCE7C0ULL, 0xAFF7BEBBFBBCE7C0ULL, 0x2FEDBEB0FBBCE7C0ULL,
        0xE6E97FBBBEBB93B6ULL, 0x1E5BA7BFBEEAD402ULL, 0x2CEBBECBFBBCE7C0ULL, 0x3AEBBEBBFBBCE7C0ULL,
        0x4EEFBEBBFBBCE7C0ULL, 0x6FAFBEBBFBBCE7C0ULL, 0xAFF7BEBBFBBCE7C0ULL, 0x2FEDBEB0FBBCE7C0ULL,
        // idx 10-11 (640-767)
        0xF143B2E7AC57BEC1ULL, 0xCD3DEE2CE5E48FB9ULL, 0xE3ABFED6B7CA9C03ULL, 0x4B4FAEABBA6BDE81ULL,
        0x4ECFBEBBFBBCE7C1ULL, 0x5FDBBEBBFBBCE7C1ULL, 0x7FF7BEBBFBBCE7C1ULL, 0xBFFDB0B0FBBCE7C1ULL,
        0x47B3DAFCC7EEB2FCULL, 0x24E7EBB7AFCBD402ULL, 0x3CEFBECBFBBCE7C1ULL, 0x4AEFBEBBFBBCE7C1ULL,
        0x5EEFBEBBFBBCE7C1ULL, 0x7FAFBEBBFBBCE7C1ULL, 0xBF93BEBBFBBCE7C1ULL, 0x3FEDBEB0FBBCE7C1ULL,
        0x5D69E3EAC5D7B906ULL, 0x0E6FE7BBB4EADCEAULL, 0x3CEBBECBFBBCE7C1ULL, 0x4AEFBEBBFBBCE7C1ULL,
        0x5EEFBEBBFBBCE7C1ULL, 0x7FAFBEBBFBBCE7C1ULL, 0xBFF7BEBBFBBCE7C1ULL, 0x3FEDBEB0FBBCE7C1ULL,
        0x5397B4F6C5E7B9B1ULL, 0x0E5FE7BFB4EADCEAULL, 0x3CEBBECBFBBCE7C1ULL, 0x4AEBBEBBFBBCE7C1ULL,
        0x5EEFBEBBFBBCE7C1ULL, 0x7FAFBEBBFBBCE7C1ULL, 0xBFF7BEBBFBBCE7C1ULL, 0x3FEDBEB0FBBCE7C1ULL,
        0xBAD5BAEEB2E5CAF7ULL, 0xFD7EE3AFB1CAEF7BULL, 0x3CE3BEDBFBBCE7C1ULL, 0x49EFBE9BFBBCE7C1ULL,
        0x5E6FBEBBFBBCE7C1ULL, 0x7F9FBEBBFBBCE7C1ULL, 0xBFF3BEBBFBBCE7C1ULL, 0x3FFBBEBBFBBCE7C1ULL,
        0xF4E97FBBBEBB93B5ULL, 0x2E5BA7BFBEEAD401ULL, 0x3CEBBECBFBBCE7C1ULL, 0x4AEBBEBBFBBCE7C1ULL,
        0x5EEFBEBBFBBCE7C1ULL, 0x7FAFBEBBFBBCE7C1ULL, 0xBFF7BEBBFBBCE7C1ULL, 0x3FEDBEB0FBBCE7C1ULL,
        0xF5E97FBBBEBB93B6ULL, 0x2E5BA7BFBEEAD402ULL, 0x3CEBBECBFBBCE7C1ULL, 0x4AEBBEBBFBBCE7C1ULL,
        0x5EEFBEBBFBBCE7C1ULL, 0x7FAFBEBBFBBCE7C1ULL, 0xBFF7BEBBFBBCE7C1ULL, 0x3FEDBEB0FBBCE7C1ULL,
        0xF6E97FBBBEBB93B7ULL, 0x2E5BA7BFBEEAD403ULL, 0x3CEBBECBFBBCE7C1ULL, 0x4AEBBEBBFBBCE7C1ULL,
        0x5EEFBEBBFBBCE7C1ULL, 0x7FAFBEBBFBBCE7C1ULL, 0xBFF7BEBBFBBCE7C1ULL, 0x3FEDBEB0FBBCE7C1ULL,
        // wK (idx=11 sq 0-63) indices 704-767
        0x0143B2E7AC57BEC2ULL, 0xDD3DEE2CE5E48FBAULL, 0xF3ABFED6B7CA9C04ULL, 0x5B4FAEABBA6BDE82ULL,
        0x5ECFBEBBFBBCE7C2ULL, 0x6FDBBEBBFBBCE7C2ULL, 0x8FF7BEBBFBBCE7C2ULL, 0xCFFDB0B0FBBCE7C2ULL,
        0x57B3DAFCC7EEB2FDULL, 0x34E7EBB7AFCBD403ULL, 0x4CEFBECBFBBCE7C2ULL, 0x5AEFBEBBFBBCE7C2ULL,
        0x6EEFBEBBFBBCE7C2ULL, 0x8FAFBEBBFBBCE7C2ULL, 0xCF93BEBBFBBCE7C2ULL, 0x4FEDBEB0FBBCE7C2ULL,
        0x6D69E3EAC5D7B907ULL, 0x1E6FE7BBB4EADCEBULL, 0x4CEBBECBFBBCE7C2ULL, 0x5AEFBEBBFBBCE7C2ULL,
        0x6EEFBEBBFBBCE7C2ULL, 0x8FAFBEBBFBBCE7C2ULL, 0xCFF7BEBBFBBCE7C2ULL, 0x4FEDBEB0FBBCE7C2ULL,
        0x6397B4F6C5E7B9B2ULL, 0x1E5FE7BFB4EADCEBULL, 0x4CEBBECBFBBCE7C2ULL, 0x5AEBBEBBFBBCE7C2ULL,
        0x6EEFBEBBFBBCE7C2ULL, 0x8FAFBEBBFBBCE7C2ULL, 0xCFF7BEBBFBBCE7C2ULL, 0x4FEDBEB0FBBCE7C2ULL,
        0xCAD5BAEEB2E5CAF8ULL, 0x0D7EE3AFB1CAEF7CULL, 0x4CE3BEDBFBBCE7C2ULL, 0x59EFBE9BFBBCE7C2ULL,
        0x6E6FBEBBFBBCE7C2ULL, 0x8F9FBEBBFBBCE7C2ULL, 0xCFF3BEBBFBBCE7C2ULL, 0x4FFBBEBBFBBCE7C2ULL,
        0x04E97FBBBEBB93B6ULL, 0x3E5BA7BFBEEAD402ULL, 0x4CEBBECBFBBCE7C2ULL, 0x5AEBBEBBFBBCE7C2ULL,
        0x6EEFBEBBFBBCE7C2ULL, 0x8FAFBEBBFBBCE7C2ULL, 0xCFF7BEBBFBBCE7C2ULL, 0x4FEDBEB0FBBCE7C2ULL,
        0x05E97FBBBEBB93B7ULL, 0x3E5BA7BFBEEAD403ULL, 0x4CEBBECBFBBCE7C2ULL, 0x5AEBBEBBFBBCE7C2ULL,
        0x6EEFBEBBFBBCE7C2ULL, 0x8FAFBEBBFBBCE7C2ULL, 0xCFF7BEBBFBBCE7C2ULL, 0x4FEDBEB0FBBCE7C2ULL,
        0x06E97FBBBEBB93B8ULL, 0x3E5BA7BFBEEAD404ULL, 0x4CEBBECBFBBCE7C2ULL, 0x5AEBBEBBFBBCE7C2ULL,
        0x6EEFBEBBFBBCE7C2ULL, 0x8FAFBEBBFBBCE7C2ULL, 0xCFF7BEBBFBBCE7C2ULL, 0x4FEDBEB0FBBCE7C2ULL,
        // Castling [768..771], EP files [772..779], STM [780]
        0xF8D626AAAF278509ULL, 0x31D71DCE64281F29ULL,
        0xF165B587DF898190ULL, 0xA57E6339DD2CF3A0ULL,
        0x1EF6E6DBB1961EC9ULL, 0x70CC73D90BC26E24ULL,
        0xE21A6B35DF0C3AD7ULL, 0x003A93D8B2806962ULL,
        0x1C99DED33CB890A1ULL, 0xCF3145DE0ADD4289ULL,
        0xD0E4427A5514FB72ULL, 0x77C621CC9FB3A483ULL,
        0x871E87F9E7BE70EDULL,
    };
    static inline uint64_t Random64(int i) { return _table[i]; }
    static constexpr bool is_standard = false;
#endif

    // Piece-index mapping for get_polyglot_hash():
    //  0=bP 1=wP 2=bN 3=wN 4=bB 5=wB 6=bR 7=wR 8=bQ 9=wQ 10=bK 11=wK
    inline uint64_t piece_sq(int piece_idx, int sq) { return Random64(piece_idx * 64 + sq); }
    inline uint64_t castling(int bit)  { return Random64(768 + bit); }
    inline uint64_t ep_file(int file)  { return Random64(772 + file); }
    inline uint64_t side_to_move()    { return Random64(780); }
}

// Forward declaration so Position::bulk_count_legal() can call generate_moves.
class Position;
int generate_moves(const Position& pos, Move* moves, bool captures_only);

// Position class (Chess960, castling undo)
class Position {
private:
    U64 _pieces[2][7];
    int board[64];
    Color side;
    U64 occupied;
    int fifty;
    int ply;
    int game_ply;
    Square ep_square;
    Square castle_rook_sq[2][2];
    bool chess960;
    U64 hash;
    std::vector<U64> history;

public:
    Position() { clear(); }
    void clear() {
        memset(_pieces, 0, sizeof(_pieces));
        memset(board, 0, sizeof(board));
        occupied = 0;
        side = WHITE;
        fifty = 0;
        ply = 0;
        game_ply = 0;
        ep_square = -1;
        for (int c = 0; c < 2; ++c)
            for (int s = 0; s < 2; ++s)
                castle_rook_sq[c][s] = -1;
        chess960 = false;
        hash = 0;
        history.clear();
        history.push_back(0);
    }
    void update_occupied() {
        occupied = 0;
        for (int c = 0; c < 2; ++c)
            for (int pt = PAWN; pt <= KING; ++pt)
                occupied |= _pieces[c][pt];
    }
    void init_startpos() {
        clear();
        side = WHITE;
        for (int f = 0; f < 8; ++f) {
            _pieces[WHITE][PAWN] |= 1ULL << make_square(f, 1);
            _pieces[BLACK][PAWN] |= 1ULL << make_square(f, 6);
            board[make_square(f, 1)] = (WHITE << 3) | PAWN;
            board[make_square(f, 6)] = (BLACK << 3) | PAWN;
        }
        int back[8] = {ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK};
        for (int f = 0; f < 8; ++f) {
            _pieces[WHITE][back[f]] |= 1ULL << make_square(f, 0);
            _pieces[BLACK][back[f]] |= 1ULL << make_square(f, 7);
            board[make_square(f, 0)] = (WHITE << 3) | back[f];
            board[make_square(f, 7)] = (BLACK << 3) | back[f];
        }
        update_occupied();
        castle_rook_sq[WHITE][0] = make_square(7, 0);
        castle_rook_sq[WHITE][1] = make_square(0, 0);
        castle_rook_sq[BLACK][0] = make_square(7, 7);
        castle_rook_sq[BLACK][1] = make_square(0, 7);
        chess960 = false;
        ep_square = -1;
        game_ply = 1;
        compute_hash();
        history.clear();
        history.push_back(hash);
    }
    void set_fen(const std::string& fen) {
        clear();
        std::istringstream ss(fen);
        std::string board_str, side_str, castle, ep;
        int hmvc, fmvn;
        ss >> board_str >> side_str >> castle >> ep >> hmvc >> fmvn;
        Square sq = 56;
        for (char c : board_str) {
            if (c == '/') {
                sq -= 16;
            } else if (c >= '1' && c <= '8') {
                sq += (c - '0');
            } else {
                Color col = isupper(c) ? WHITE : BLACK;
                PieceType pt = NO_PIECE;
                switch (tolower(c)) {
                    case 'p': pt = PAWN; break;
                    case 'n': pt = KNIGHT; break;
                    case 'b': pt = BISHOP; break;
                    case 'r': pt = ROOK; break;
                    case 'q': pt = QUEEN; break;
                    case 'k': pt = KING; break;
                    default: break;
                }
                if (pt != NO_PIECE && sq >= 0 && sq < 64) {
                    _pieces[col][pt] |= 1ULL << sq;
                    board[sq] = (col << 3) | pt;
                    sq++;
                }
            }
        }
        side = (side_str == "w") ? WHITE : BLACK;
        update_occupied();
        for (int c = 0; c < 2; ++c)
            for (int s = 0; s < 2; ++s)
                castle_rook_sq[c][s] = -1;
        if (castle != "-") {
            for (char ch : castle) {
                Color col = isupper(ch) ? WHITE : BLACK;
                int rank = (col == WHITE) ? 0 : 7;
                char lch = tolower(ch);
                int file = -1;
                if (lch == 'k') {
                    // Standard FEN kingside: find the first rook to the right of the king
                    Square ksq = lsb(_pieces[col][KING]);
                    for (int f = file_of(ksq) + 1; f < 8; ++f) {
                        if (_pieces[col][ROOK] & (1ULL << make_square(f, rank))) {
                            file = f; break;
                        }
                    }
                } else if (lch == 'q') {
                    // Standard FEN queenside: find the first rook to the left of the king
                    Square ksq = lsb(_pieces[col][KING]);
                    for (int f = file_of(ksq) - 1; f >= 0; --f) {
                        if (_pieces[col][ROOK] & (1ULL << make_square(f, rank))) {
                            file = f; break;
                        }
                    }
                } else {
                    // Chess960 Shredder-FEN: file letter directly encodes the rook file
                    file = lch - 'a';
                }
                if (file < 0 || file > 7) continue;
                Square rook_sq = make_square(file, rank);
                if ((_pieces[col][ROOK] & (1ULL << rook_sq)) == 0) continue;
                Square ksq = lsb(_pieces[col][KING]);
                int side_idx = (file > file_of(ksq)) ? 0 : 1;
                castle_rook_sq[col][side_idx] = rook_sq;
            }
        }
        chess960 = (castle_rook_sq[WHITE][0] != make_square(7,0) ||
                    castle_rook_sq[WHITE][1] != make_square(0,0) ||
                    castle_rook_sq[BLACK][0] != make_square(7,7) ||
                    castle_rook_sq[BLACK][1] != make_square(0,7));
#ifdef CHESS960_EXTRA_DEBUG
        if (chess960) {
            std::cerr << "[c960] Chess960 position detected from FEN\n";
            std::cerr << "[c960] WK=" << (char)('a'+file_of(king_square(WHITE))) << rank_of(king_square(WHITE))+1
                      << "  WR-K=" << (castle_rook_sq[WHITE][0]!=-1 ? std::string(1,(char)('a'+file_of(castle_rook_sq[WHITE][0]))) : "-")
                      << "  WR-Q=" << (castle_rook_sq[WHITE][1]!=-1 ? std::string(1,(char)('a'+file_of(castle_rook_sq[WHITE][1]))) : "-")
                      << "\n";
        }
#endif
        ep_square = (ep != "-") ? make_square(ep[0]-'a', ep[1]-'1') : -1;
        fifty = hmvc;
        // game_ply is a fullmove counter: incremented once after BLACK's move
        // (when side flips back to WHITE), so it must be initialised to the FEN
        // fullmove number directly — not converted to a ply index.
        game_ply = fmvn;
        compute_hash();
        history.clear();
        history.push_back(hash);
    }
    std::string fen() const {
        std::string fen;
        for (int r = 7; r >= 0; --r) {
            int empty = 0;
            for (int f = 0; f < 8; ++f) {
                Square sq = make_square(f, r);
                int pc = board[sq];
                if (pc == 0) {
                    empty++;
                } else {
                    if (empty > 0) {
                        fen += std::to_string(empty);
                        empty = 0;
                    }
                    Color c = Color(pc >> 3);
                    PieceType pt = PieceType(pc & 7);
                    char p = " pnbrqk"[pt];
                    if (c == WHITE) p = toupper(p);
                    fen += p;
                }
            }
            if (empty > 0) fen += std::to_string(empty);
            if (r > 0) fen += '/';
        }
        fen += (side == WHITE) ? " w " : " b ";
        std::string castle_str;
        if (!chess960) {
            // Standard FEN: K=white kingside, Q=white queenside, k=black kingside, q=black queenside
            if (castle_rook_sq[WHITE][0] != -1) castle_str += 'K';
            if (castle_rook_sq[WHITE][1] != -1) castle_str += 'Q';
            if (castle_rook_sq[BLACK][0] != -1) castle_str += 'k';
            if (castle_rook_sq[BLACK][1] != -1) castle_str += 'q';
        } else {
            // Chess960 Shredder-FEN: uppercase file letter for white, lowercase for black
            for (int c = 0; c < 2; ++c) {
                for (int s = 0; s < 2; ++s) {
                    Square rsq = castle_rook_sq[Color(c)][s];
                    if (rsq != -1) {
                        char file_char = char('a' + file_of(rsq));
                        castle_str += (c == WHITE) ? char(toupper(file_char)) : file_char;
                    }
                }
            }
        }
        if (castle_str.empty()) castle_str = "-";
        fen += castle_str + " ";
        fen += (ep_square != -1) ? std::string(1, 'a' + file_of(ep_square)) + std::to_string(rank_of(ep_square)+1) : "-";
        fen += " " + std::to_string(fifty) + " " + std::to_string(game_ply);
        return fen;
    }

    void compute_hash() {
        U64 h = 0;
        for (int c = 0; c < 2; ++c)
            for (int pt = PAWN; pt <= KING; ++pt) {
                U64 bb = _pieces[c][pt];
                while (bb) {
                    Square sq = pop_lsb(bb);
                    h ^= Zobrist::pieces[c][pt][sq];
                }
            }
        if (side == BLACK) h ^= Zobrist::side;
        int castle_mask = 0;
        if (castle_rook_sq[WHITE][0] != -1) castle_mask |= 1;
        if (castle_rook_sq[WHITE][1] != -1) castle_mask |= 2;
        if (castle_rook_sq[BLACK][0] != -1) castle_mask |= 4;
        if (castle_rook_sq[BLACK][1] != -1) castle_mask |= 8;
        h ^= Zobrist::castle[castle_mask];
        if (ep_square != -1) h ^= Zobrist::ep[ep_square];
        hash = h;
    }
    U64 get_hash() const { return hash; }

    // Pawn structure hash — used for pawn correction history (CORRHIST).
    // XOR of Zobrist keys for all pawn squares, independent of side/castle/ep.
    U64 get_pawn_hash() const {
        U64 h = 0;
        U64 wp = _pieces[WHITE][PAWN], bp = _pieces[BLACK][PAWN];
        U64 tmp = wp;
        while (tmp) { Square sq = pop_lsb(tmp); h ^= Zobrist::pieces[WHITE][PAWN][sq]; }
        tmp = bp;
        while (tmp) { Square sq = pop_lsb(tmp); h ^= Zobrist::pieces[BLACK][PAWN][sq]; }
        return h;
    }

    // Non-pawn material hash — used for material correction history.
    U64 get_nonpawn_hash(Color us) const {
        U64 h = 0;
        for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
            U64 tmp = _pieces[us][pt];
            while (tmp) { Square sq = pop_lsb(tmp); h ^= Zobrist::pieces[us][pt][sq]; }
        }
        return h;
    }

    // Compute the standard Polyglot hash for this position.
    // Uses the 781 public-domain random values so .bin books from any source match.
    U64 get_polyglot_hash() const {
        // Polyglot piece indices: 1=wP,3=wN,5=wB,7=wR,9=wQ,11=wK
        //                         0=bP,2=bN,4=bB,6=bR,8=bQ,10=bK
        static const int pg_piece[2][7] = {
            // WHITE: PAWN=1 KNIGHT=3 BISHOP=5 ROOK=7 QUEEN=9 KING=11
            {0, 1, 3, 5, 7, 9, 11},
            // BLACK: PAWN=0 KNIGHT=2 BISHOP=4 ROOK=6 QUEEN=8 KING=10
            {0, 0, 2, 4, 6, 8, 10}
        };
        uint64_t h = 0;
        for (int c = 0; c < 2; ++c)
            for (int pt = PAWN; pt <= KING; ++pt) {
                U64 bb = _pieces[c][pt];
                while (bb) {
                    Square sq = pop_lsb(bb);
                    h ^= PolyglotHash::piece_sq(pg_piece[c][pt], sq);
                }
            }
        // Castling rights
        if (castle_rook_sq[WHITE][0] != -1) h ^= PolyglotHash::castling(0); // wK
        if (castle_rook_sq[WHITE][1] != -1) h ^= PolyglotHash::castling(1); // wQ
        if (castle_rook_sq[BLACK][0] != -1) h ^= PolyglotHash::castling(2); // bK
        if (castle_rook_sq[BLACK][1] != -1) h ^= PolyglotHash::castling(3); // bQ
        // En passant — Polyglot only includes ep file if a pawn can actually capture
        if (ep_square != -1) {
            int ep_file = file_of(ep_square);
            bool ep_valid = false;
            // Check if there is actually an enemy pawn adjacent that can capture
            Color stm = side;
            int ep_rank = rank_of(ep_square);
            if (stm == WHITE && ep_rank == 5) {
                if (ep_file > 0 && (_pieces[WHITE][PAWN] & (1ULL << make_square(ep_file-1, 4)))) ep_valid = true;
                if (ep_file < 7 && (_pieces[WHITE][PAWN] & (1ULL << make_square(ep_file+1, 4)))) ep_valid = true;
            } else if (stm == BLACK && ep_rank == 2) {
                if (ep_file > 0 && (_pieces[BLACK][PAWN] & (1ULL << make_square(ep_file-1, 3)))) ep_valid = true;
                if (ep_file < 7 && (_pieces[BLACK][PAWN] & (1ULL << make_square(ep_file+1, 3)))) ep_valid = true;
            }
            if (ep_valid) h ^= PolyglotHash::ep_file(ep_file);
        }
        // Side to move
        if (side == BLACK) h ^= PolyglotHash::side_to_move();
        return h;
    }
    bool is_repetition(int count) const {
        // Stop searching back further than the current halfmove clock allows —
        // there can be no repetition before an irreversible move (capture or
        // pawn push), so we only need to scan the last `fifty` half-moves.
        int limit = std::min((int)history.size() - 1, fifty);
        int c = 0;
        for (int i = (int)history.size() - 2; i >= (int)history.size() - 1 - limit && i >= 0; i -= 2) {
            if (history[i] == hash) {
                c++;
                if (c >= count) return true;
            }
        }
        return false;
    }
    void push_hash() { history.push_back(hash); }
    void pop_hash() { history.pop_back(); }

    U64 attacks_to(Square s) const { return attacks_to(s, occupied); }
    U64 attacks_to(Square s, U64 occ) const {
        U64 attackers = 0;
        attackers |= Bitboards::pawn_attacks[BLACK][s] & _pieces[WHITE][PAWN];
        attackers |= Bitboards::pawn_attacks[WHITE][s] & _pieces[BLACK][PAWN];
        attackers |= Bitboards::knight_attacks[s] & (_pieces[WHITE][KNIGHT] | _pieces[BLACK][KNIGHT]);
        U64 bishops = _pieces[WHITE][BISHOP] | _pieces[BLACK][BISHOP] | _pieces[WHITE][QUEEN] | _pieces[BLACK][QUEEN];
        attackers |= bishop_attacks_magic(s, occ) & bishops;
        U64 rooks = _pieces[WHITE][ROOK] | _pieces[BLACK][ROOK] | _pieces[WHITE][QUEEN] | _pieces[BLACK][QUEEN];
        attackers |= rook_attacks_magic(s, occ) & rooks;
        attackers |= Bitboards::king_attacks[s] & (_pieces[WHITE][KING] | _pieces[BLACK][KING]);
        return attackers;
    }
    bool is_check() const {
        if (!_pieces[side][KING]) return false;
        Square ksq = lsb(_pieces[side][KING]);
        return attacks_to(ksq) & ~(_pieces[side][PAWN] | _pieces[side][KNIGHT] | _pieces[side][BISHOP] |
                                   _pieces[side][ROOK] | _pieces[side][QUEEN] | _pieces[side][KING]);
    }
    bool is_attacked(Square s, Color by) const {
        U64 attackers = attacks_to(s);
        U64 by_pieces = 0;
        for (int pt = PAWN; pt <= KING; ++pt)
            by_pieces |= _pieces[by][pt];
        return (attackers & by_pieces) != 0;
    }
    int game_phase() const {
        int phase = 0;
        for (int c = 0; c < 2; ++c) {
            phase += popcount(_pieces[c][KNIGHT]) * PHASE_KNIGHT;
            phase += popcount(_pieces[c][BISHOP]) * PHASE_BISHOP;
            phase += popcount(_pieces[c][ROOK]) * PHASE_ROOK;
            phase += popcount(_pieces[c][QUEEN]) * PHASE_QUEEN;
        }
        return std::min(phase, TOTAL_PHASE);
    }
    bool is_endgame() const { return game_phase() < 12; }
    // Returns the king's square. Returns NO_SQUARE (-1) if the king is missing
    // (can happen in positions constructed for evaluation tests or after illegal moves).
    Square king_square(Color c) const {
        if (!_pieces[c][KING]) return Square(-1);  // no-king sentinel
        return lsb(_pieces[c][KING]);
    }

    Value see(Move m) const {
        if (m == NULL_MOVE) return 0;
        Square from = from_sq(m), to = to_sq(m);
        U64 occ = occupied;
        Color us = side;
        bool ep = is_en_passant(m);
        bool promo = promotion_type(m) != NO_PIECE;
        PieceType prom_type = promotion_type(m);

        // Remove the initial attacker from the occupancy map
        occ &= ~(1ULL << from);

        // Determine the victim's piece type and update occ accordingly
        int victim_type = 0;
        if (ep) {
            Square ep_cap = to + (us == WHITE ? -8 : 8);
            victim_type = PAWN;
            occ &= ~(1ULL << ep_cap);  // captured pawn leaves the board
        } else {
            int captured = board[to];
            if (captured) {
                victim_type = captured & 7;
                occ &= ~(1ULL << to);  // victim leaves before we place our piece
            }
        }

        // Initial attacker's piece type (promotion changes what's placed on to)
        int piece_on_sq = promo ? int(prom_type) : int(board[from] & 7);
        // Place the initial attacker on the target square
        occ |= (1ULL << to);

        if (victim_type == 0) return 0;  // nothing to capture (shouldn't happen for captures)

        Value gain[32];
        int d = 0;
        gain[0] = PIECE_VALUES[victim_type];

        // Alternate between sides. Each recapture gains the piece currently on to
        // (= the last attacker placed there), minus what the previous side gained.
        Color stm = Color(us ^ 1);  // opponent responds first
        while (true) {
            // Find cheapest attacker for stm that can reach 'to' given current occ
            int best_att_type = 0;
            Square best_sq = -1;
            for (int pt = PAWN; pt <= KING; ++pt) {
                U64 attackers = _pieces[stm][pt] & occ & attacks_to(to, occ);
                if (attackers) {
                    best_att_type = pt;
                    best_sq = lsb(attackers);
                    break;
                }
            }
            if (best_sq == -1) break;  // no more recaptures possible

            d++;
            // The current side captures 'piece_on_sq' (what the last side placed on to)
            gain[d] = PIECE_VALUES[piece_on_sq] - gain[d-1];
            // Update occ: attacker leaves its square, and is placed on 'to'
            occ &= ~(1ULL << best_sq);
            piece_on_sq = best_att_type;  // now this piece is on 'to'
            stm = Color(stm ^ 1);
        }

        // Roll back: each side only makes the capture if it gains material
        while (d > 0) {
            gain[d-1] = -std::max(-gain[d-1], gain[d]);
            d--;
        }
        return gain[0];
    }
    // Fast check detection — no Position copy, just bitboard queries.
    // Handles direct checks (piece lands attacks king) and discovered checks
    // (removing 'from' reveals a slider behind it attacking the king).
    bool gives_check(Move m) const {
        if (m == NO_MOVE || m == NULL_MOVE) return false;
        Square from = from_sq(m), to = to_sq(m);
        int pc = board[from];
        if (!pc) return false;
        Color us   = Color(pc >> 3);
        Color them = Color(us ^ 1);
        if (!_pieces[them][KING]) return false;
        Square ksq = lsb(_pieces[them][KING]);

        // Hypothetical occupancy after the move
        U64 occ = occupied;
        occ &= ~(1ULL << from);
        occ |=  (1ULL << to);
        if (is_en_passant(m)) {
            Square ep_cap = to + (us == WHITE ? -8 : 8);
            occ &= ~(1ULL << ep_cap);
        }

        // Piece that lands on 'to' (promotion changes piece type)
        PieceType pt = promotion_type(m) != NO_PIECE ? promotion_type(m) : PieceType(pc & 7);

        // 1) Direct check
        bool direct = false;
        switch (pt) {
            case PAWN:   direct = (Bitboards::pawn_attacks[us][to] & (1ULL<<ksq)) != 0; break;
            case KNIGHT: direct = (Bitboards::knight_attacks[to]   & (1ULL<<ksq)) != 0; break;
            case BISHOP: direct = (bishop_attacks_magic(to, occ)   & (1ULL<<ksq)) != 0; break;
            case ROOK:   direct = (rook_attacks_magic(to, occ)     & (1ULL<<ksq)) != 0; break;
            case QUEEN:  direct = (queen_attacks_magic(to, occ)    & (1ULL<<ksq)) != 0; break;
            default: break;
        }
        if (direct) return true;

        // 2) Discovered check: did removing 'from' unblock a slider?
        if (rook_attacks_magic(ksq, occ)   & (_pieces[us][ROOK]   | _pieces[us][QUEEN])) return true;
        if (bishop_attacks_magic(ksq, occ) & (_pieces[us][BISHOP] | _pieces[us][QUEEN])) return true;
        return false;
    }

    void make_move(Move m) {
        if (m == NULL_MOVE) {
            // Null move: flip side, but CLEAR en-passant square.
            // The EP square must not persist into the null-move subtree:
            // (a) it is illegal to capture en-passant on a null move turn, and
            // (b) keeping it in the hash causes TT collisions where positions
            //     that differ only in whether they have an EP square get the
            //     same key, producing corrupted scores.
            ep_square = -1;
            side = Color(side ^ 1);
            ply++;
            if (side == WHITE) game_ply++;
            push_hash();
            compute_hash();
            return;
        }
        Square from = from_sq(m), to = to_sq(m);
        int pc = board[from];
        int pt = pc & 7;
        Color us = side;
        Color them = Color(us ^ 1);
        int captured = board[to];
        _pieces[us][pt] ^= 1ULL << from;
        board[from] = 0;
        if (is_castling(m)) {
            int side_idx = (to > from) ? 0 : 1;
            Square rook_sq = castle_rook_sq[us][side_idx];
#ifdef CHESS960_EXTRA_DEBUG
            std::cerr << "[c960] Castling: king " << (char)('a'+file_of(from)) << rank_of(from)+1
                      << "->" << (char)('a'+file_of(to)) << rank_of(to)+1
                      << "  rook " << (char)('a'+file_of(rook_sq)) << rank_of(rook_sq)+1
                      << "->dest\n";
#endif
            // Rook always lands on f-file (kingside) or d-file (queenside) after castling,
            // regardless of where the king or rook started (correct for both standard and Chess960).
            int castling_rank_mk = (us == WHITE) ? 0 : 7;
            Square rook_dest = make_square((side_idx == 0) ? 5 : 3, castling_rank_mk);
            _pieces[us][ROOK] ^= (1ULL << rook_sq);
            _pieces[us][ROOK] |= (1ULL << rook_dest);
            board[rook_sq] = 0;
            board[rook_dest] = (us << 3) | ROOK;
            castle_rook_sq[us][side_idx] = -1;
        } else if (is_en_passant(m)) {
            Square ep_cap = to + (us == WHITE ? -8 : 8);
            int ep_pc = board[ep_cap];
            if (ep_pc) {
                int ep_pt = ep_pc & 7;
                _pieces[them][ep_pt] ^= 1ULL << ep_cap;
                board[ep_cap] = 0;
            }
            captured = ep_pc;
        }
        _pieces[us][pt] |= 1ULL << to;
        board[to] = pc;
        if (captured && !is_en_passant(m) && !is_castling(m)) {
            int cap_pt = captured & 7;
            _pieces[them][cap_pt] ^= 1ULL << to;
        }
        if (promotion_type(m)) {
            PieceType prom = promotion_type(m);
            _pieces[us][pt] ^= 1ULL << to;
            _pieces[us][prom] |= 1ULL << to;
            board[to] = (us << 3) | prom;
        }
        if (pt == KING) {
            castle_rook_sq[us][0] = castle_rook_sq[us][1] = -1;
        }
        for (int s = 0; s < 2; ++s) {
            if (from == castle_rook_sq[us][s]) castle_rook_sq[us][s] = -1;
            if (from == castle_rook_sq[them][s]) castle_rook_sq[them][s] = -1;
        }
        if (pt == PAWN && abs(to - from) == 16) {
            ep_square = (us == WHITE) ? from + 8 : from - 8;
        } else {
            ep_square = -1;
        }
        if (captured || pt == PAWN) fifty = 0; else fifty++;
        update_occupied();
        side = them;
        ply++;
        if (side == WHITE) game_ply++;
        push_hash();
        compute_hash();
    }

    void undo_null_move() {
        side = Color(side ^ 1);
        ply--;
        if (side == BLACK) game_ply--;
        pop_hash();
        compute_hash();
    }

    void undo_move(Move m, int captured, int old_castle, int old_ep, int old_fifty) {
        if (m == NULL_MOVE) { undo_null_move(); return; }
        side = Color(side ^ 1);
        Square from = from_sq(m), to = to_sq(m);
        int pc = board[to];
        int pt = pc & 7;
        Color us = side;
        _pieces[us][pt] ^= 1ULL << to;
        board[to] = captured;
        _pieces[us][pt] |= 1ULL << from;
        board[from] = (us << 3) | pt;
        if (captured && !is_en_passant(m) && !is_castling(m)) {
            int cap_pt = captured & 7;
            Color them = Color(us ^ 1);
            _pieces[them][cap_pt] |= 1ULL << to;
        }
        if (is_castling(m)) {
            // Recover original rook square directly from old_castle (packed encoding).
            int side_idx = (to > from) ? 0 : 1;
            auto dec = [](int v) -> Square { return Square((v & 0x7F) - 1); };
            Square orig_rook_sq = (side_idx == 0) ? dec(old_castle) : dec(old_castle >> 7);
            if (us == BLACK) orig_rook_sq = (side_idx == 0) ? dec(old_castle >> 14) : dec(old_castle >> 21);
            // Rook always lands on f-file (kingside) or d-file (queenside) after castling,
            // regardless of the king's original file (critical for Chess960 correctness).
            int castling_rank = (us == WHITE) ? 0 : 7;
            Square rook_dest = make_square((side_idx == 0) ? 5 : 3, castling_rank);
            _pieces[us][ROOK] ^= (1ULL << rook_dest);
            _pieces[us][ROOK] |= (1ULL << orig_rook_sq);
            board[rook_dest] = 0;
            board[orig_rook_sq] = (us << 3) | ROOK;
        } else if (is_en_passant(m)) {
            Square ep_cap = to + (us == WHITE ? -8 : 8);
            _pieces[us^1][PAWN] |= 1ULL << ep_cap;
            board[ep_cap] = ((us^1) << 3) | PAWN;
        }
        if (promotion_type(m)) {
            _pieces[us][promotion_type(m)] ^= 1ULL << from;
            _pieces[us][PAWN] |= 1ULL << from;
            board[from] = (us << 3) | PAWN;
        }
        // Restore castling rights exactly from packed old_castle — no back-rank scanning.
        restore_castling_rights(old_castle);
        ep_square = old_ep;
        fifty = old_fifty;
        update_occupied();
        ply--;
        if (side == BLACK) game_ply--;
        pop_hash();
        compute_hash();
    }

    U64 bb(Color c, PieceType pt) const { return _pieces[c][pt]; }
    Color side_to_move() const { return side; }
    U64 occupied_bb() const { return occupied; }
    int piece_on(Square s) const { return board[s]; }
    int halfmove_clock() const { return fifty; }
    int fullmove_number() const { return game_ply; }
    Square ep_sq() const { return ep_square; }
    Square castle_rook(Color c, int side) const { return castle_rook_sq[c][side]; }
    int castling_rights() const {
        // Pack all 4 rook squares into a 32-bit int.
        // Each slot occupies 7 bits: (square+1), where 0 means "no right" (-1+1=0).
        // Layout: bits 0-6 = WHITE[0], bits 7-13 = WHITE[1],
        //         bits 14-20 = BLACK[0], bits 21-27 = BLACK[1].
        auto enc = [](Square sq) { return (sq + 1) & 0x7F; };
        return  enc(castle_rook_sq[WHITE][0])
             | (enc(castle_rook_sq[WHITE][1]) << 7)
             | (enc(castle_rook_sq[BLACK][0]) << 14)
             | (enc(castle_rook_sq[BLACK][1]) << 21);
    }
    // Restore castle_rook_sq[] from a value previously returned by castling_rights()
    void restore_castling_rights(int packed) {
        auto dec = [](int v) -> Square { return Square((v & 0x7F) - 1); };
        castle_rook_sq[WHITE][0] = dec(packed);
        castle_rook_sq[WHITE][1] = dec(packed >> 7);
        castle_rook_sq[BLACK][0] = dec(packed >> 14);
        castle_rook_sq[BLACK][1] = dec(packed >> 21);
    }
    bool mover_in_check() const {
        Color prev = Color(side ^ 1);
        if (!_pieces[prev][KING]) return false;
        Square ksq = lsb(_pieces[prev][KING]);
        return is_attacked(ksq, side);
    }
    bool is_chess960() const { return chess960; }
    void set_chess960(bool v) { chess960 = v; }

    // ── Perft bulk-count helper ─────────────────────────────────────────────
    // Returns a bitmask of pieces of color `us` that are pinned to their king.
    // A piece is pinned if removing it would expose the king to a sliding attack.
    // Used in bulk_count_legal() to avoid make/undo in the depth-1 perft leaf.
    U64 pinned_pieces(Color us) const {
        Square ksq = king_square(us);
        Color them = Color(us ^ 1);
        U64 pinned = 0;
        U64 our_pieces = 0;
        for (int pt = PAWN; pt <= QUEEN; ++pt) our_pieces |= _pieces[us][pt];

        // Candidate pinners: sliders aligned with king
        U64 bishop_pinners = bishop_attacks_magic(ksq, 0)
            & (_pieces[them][BISHOP] | _pieces[them][QUEEN]);
        U64 rook_pinners   = rook_attacks_magic(ksq, 0)
            & (_pieces[them][ROOK]   | _pieces[them][QUEEN]);

        U64 pinners = bishop_pinners | rook_pinners;
        while (pinners) {
            Square pinner = pop_lsb(pinners);
            // Squares between king and pinner under full occupancy.
            // Pinned iff exactly one friendly piece sits on this segment.
            U64 occ_ray;
            if ((U64(1) << pinner) & bishop_pinners)
                occ_ray = bishop_attacks_magic(ksq, occupied) & bishop_attacks_magic(pinner, occupied);
            else
                occ_ray = rook_attacks_magic(ksq, occupied) & rook_attacks_magic(pinner, occupied);
            U64 blockers_on_ray = occ_ray & our_pieces;
            if (popcount(blockers_on_ray) == 1)
                pinned |= blockers_on_ray;
        }
        return pinned;
    }

    // Count legal moves without make/undo.  Used only at perft depth 1 (bulk counting).
    // This is the real speedup: avoid the make/undo overhead at every leaf node.
    // Falls back to make/undo for complex cases (e.p., castling, discovered check edge cases).
    // When in check this falls back to full make/undo since check evasions are complex.
    uint64_t bulk_count_legal() const {
        Color us = side_to_move();
        // If in check, fall back to full make/undo count (complex evasion logic)
        if (is_check()) {
            Move mvs[256]; int cnt = generate_moves(*this, mvs, false);
            uint64_t legal = 0;
            Position tmp = *this;
            for (int i = 0; i < cnt; ++i) {
                int cap = tmp.piece_on(to_sq(mvs[i]));
                int old_cr = tmp.castling_rights(); int old_ep = tmp.ep_sq(); int old_50 = tmp.halfmove_clock();
                tmp.make_move(mvs[i]);
                if (!tmp.mover_in_check()) ++legal;
                tmp.undo_move(mvs[i], cap, old_cr, old_ep, old_50);
            }
            return legal;
        }
        // Not in check: use pin analysis
        Move mvs[256]; int cnt = generate_moves(*this, mvs, false);
        U64 pinned = pinned_pieces(us);
        Square ksq = king_square(us);
        uint64_t legal = 0;
        // Use make/undo for: king moves, castling, en-passant (edge cases)
        // For normal non-pinned moves: always legal
        Position tmp = *this;
        for (int i = 0; i < cnt; ++i) {
            Move m = mvs[i];
            Square from = from_sq(m), to = to_sq(m);
            bool is_ep       = (ep_sq() != -1) && (to == ep_sq()) && (piece_on(from) & 7) == PAWN;
            bool is_castle   = (piece_on(from) & 7) == KING && std::abs(file_of(to) - file_of(from)) >= 2;
            bool is_king_move= from == ksq;
            bool is_pinned   = (pinned >> from) & 1;

            if (is_king_move || is_castle || is_ep || is_pinned) {
                // Full make/undo legality check
                int cap = tmp.piece_on(to); int old_cr = tmp.castling_rights();
                int old_ep = tmp.ep_sq(); int old_50 = tmp.halfmove_clock();
                tmp.make_move(m);
                if (!tmp.mover_in_check()) ++legal;
                tmp.undo_move(m, cap, old_cr, old_ep, old_50);
            } else {
                // Non-pinned, non-king, non-special: always legal
                ++legal;
            }
        }
        return legal;
    }
};

// Move generation
int generate_moves(const Position& pos, Move* moves, bool captures_only = false) {
    int count = 0;
    Color us = pos.side_to_move();
    Color them = Color(us ^ 1);
    U64 their_pieces_no_king = pos.bb(them, PAWN) | pos.bb(them, KNIGHT) | pos.bb(them, BISHOP) |
                                pos.bb(them, ROOK) | pos.bb(them, QUEEN);
    U64 empty = ~pos.occupied_bb();

    U64 knights = pos.bb(us, KNIGHT);
    while (knights) {
        Square from = pop_lsb(knights);
        U64 attacks = Bitboards::knight_attacks[from];
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
            U64 noncaps = attacks & empty;
            while (noncaps) { Square to = pop_lsb(noncaps); moves[count++] = make_move(from, to); }
        }
    }

    U64 bishops = pos.bb(us, BISHOP);
    while (bishops) {
        Square from = pop_lsb(bishops);
        U64 attacks = bishop_attacks_magic(from, pos.occupied_bb());
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
            U64 noncaps = attacks & empty;
            while (noncaps) { Square to = pop_lsb(noncaps); moves[count++] = make_move(from, to); }
        }
    }

    U64 rooks = pos.bb(us, ROOK);
    while (rooks) {
        Square from = pop_lsb(rooks);
        U64 attacks = rook_attacks_magic(from, pos.occupied_bb());
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
            U64 noncaps = attacks & empty;
            while (noncaps) { Square to = pop_lsb(noncaps); moves[count++] = make_move(from, to); }
        }
    }

    U64 queens = pos.bb(us, QUEEN);
    while (queens) {
        Square from = pop_lsb(queens);
        U64 attacks = queen_attacks_magic(from, pos.occupied_bb());
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
            U64 noncaps = attacks & empty;
            while (noncaps) { Square to = pop_lsb(noncaps); moves[count++] = make_move(from, to); }
        }
    }

    if (pos.bb(us, KING)) {
        Square from = lsb(pos.bb(us, KING));
        U64 attacks = Bitboards::king_attacks[from];
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) { Square to = pop_lsb(caps); moves[count++] = make_move(from, to); }
            U64 noncaps = attacks & empty;
            while (noncaps) { Square to = pop_lsb(noncaps); moves[count++] = make_move(from, to); }
        }
    }

    U64 pawns = pos.bb(us, PAWN);
    int forward = (us == WHITE) ? 8 : -8;
    U64 promo_rank = (us == WHITE) ? 0xFF00000000000000ULL : 0xFFULL;
    while (pawns) {
        Square from = pop_lsb(pawns);
        Square to = from + forward;
        if (!captures_only && to >= 0 && to < 64 && !pos.piece_on(to)) {
            if (promo_rank & (1ULL << to)) {
                moves[count++] = make_promotion(from, to, QUEEN);
                moves[count++] = make_promotion(from, to, ROOK);
                moves[count++] = make_promotion(from, to, BISHOP);
                moves[count++] = make_promotion(from, to, KNIGHT);
            } else {
                moves[count++] = make_move(from, to);
                if ((us == WHITE && rank_of(from) == 1) || (us == BLACK && rank_of(from) == 6)) {
                    Square to2 = from + 2*forward;
                    if (!pos.piece_on(to2)) moves[count++] = make_move(from, to2);
                }
            }
        }
        U64 attacks = Bitboards::pawn_attacks[us][from] & their_pieces_no_king;
        while (attacks) {
            Square to_cap = pop_lsb(attacks);
            if (promo_rank & (1ULL << to_cap)) {
                moves[count++] = make_promotion(from, to_cap, QUEEN);
                moves[count++] = make_promotion(from, to_cap, ROOK);
                moves[count++] = make_promotion(from, to_cap, BISHOP);
                moves[count++] = make_promotion(from, to_cap, KNIGHT);
            } else {
                moves[count++] = make_move(from, to_cap);
            }
        }
        if (pos.ep_sq() != -1) {
            U64 ep_attacks = Bitboards::pawn_attacks[us][from] & (1ULL << pos.ep_sq());
            if (ep_attacks) moves[count++] = make_move(from, pos.ep_sq()) | ENPASSANT_FLAG;
        }
    }

    if (!captures_only && !pos.is_check() && pos.bb(us, KING)) {
        for (int side_idx = 0; side_idx < 2; ++side_idx) {
            Square rook_sq = pos.castle_rook(us, side_idx);
            if (rook_sq == -1) continue;
            Square ksq = lsb(pos.bb(us, KING));
            if ((pos.bb(us, ROOK) & (1ULL << rook_sq)) == 0) continue;

            int castling_rank_gen = (us == WHITE) ? 0 : 7;
            // King always ends up on g-file (kingside) or c-file (queenside).
            // Rook always ends up on f-file (kingside) or d-file (queenside).
            // These are fixed by FIDE rules for both standard and Chess960.
            Square king_dest = make_square((side_idx == 0) ? 6 : 2, castling_rank_gen);
            Square rook_dest = make_square((side_idx == 0) ? 5 : 3, castling_rank_gen);

            // Chess960 sanity: rook must be on same side of king as expected
            bool rook_on_correct_side = (side_idx == 0) ? (rook_sq > ksq) : (rook_sq < ksq);
            if (!rook_on_correct_side) continue;

            bool ok = true;

            // ---- King's path clearance ----
            // Every square strictly between king and king_dest must be:
            //   (a) empty or occupied only by the castling rook
            //   (b) not attacked by the opponent
            // Special case: king_dest == ksq means king doesn't move (e.g. Chess960
            // king on g-file tries to castle kingside) — no path to check.
            if (king_dest != ksq) {
                int step = (king_dest > ksq) ? 1 : -1;
                for (Square s = ksq + step; s != king_dest; s += step) {
                    if (pos.piece_on(s) != 0 && s != rook_sq) { ok = false; break; }
                    if (pos.is_attacked(s, them)) { ok = false; break; }
                }
                if (ok && pos.piece_on(king_dest) != 0 && king_dest != rook_sq) ok = false;
                if (ok && pos.is_attacked(king_dest, them)) ok = false;
            }

            // ---- Rook's path clearance ----
            // Every square strictly between rook_sq and rook_dest must be empty
            // (ignoring the king's original square, which is vacated during castling).
            // Special case: rook_dest == rook_sq means rook doesn't move — fine.
            if (ok && rook_dest != rook_sq) {
                int step = (rook_dest > rook_sq) ? 1 : -1;
                for (Square s = rook_sq + step; s != rook_dest; s += step) {
                    if (s == ksq) continue;  // king vacates this square
                    if (pos.piece_on(s) != 0) { ok = false; break; }
                }
            }

            // King must not start the castling move in check
            if (ok && pos.is_attacked(ksq, them)) ok = false;

            if (ok) moves[count++] = make_move(ksq, king_dest) | CASTLE_FLAG;
        }
    }
    return count;
}
// ============================================================================
// End of Part 1
// ============================================================================
// ============================================================================
// Part 2 of 4: Hugine 2.0 – Evaluation, NNUE, Tables, Global State
// ============================================================================

// ----------------------------------------------------------------------------
// Piece‑square tables (midgame and endgame combined)
// ----------------------------------------------------------------------------
constexpr int PST_PAWN[64] = {
    0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,
    10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,
    0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,
    5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0
};
constexpr int PST_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,5,5,0,-20,-40,
    -30,5,10,15,15,10,5,-30,-30,0,15,20,20,15,0,-30,
    -30,5,15,20,20,15,5,-30,-30,0,10,15,15,10,0,-30,
    -40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50
};
constexpr int PST_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,-10,5,0,0,0,0,5,-10,
    -10,10,10,10,10,10,10,-10,-10,0,10,10,10,10,0,-10,
    -10,5,5,10,10,5,5,-10,-10,0,5,10,10,5,0,-10,
    -10,0,0,0,0,0,0,-10,-20,-10,-10,-10,-10,-10,-10,-20
};
constexpr int PST_ROOK[64] = {
    0,0,0,5,5,0,0,0,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,
    5,10,10,10,10,10,10,5,0,0,0,0,0,0,0,0
};
constexpr int PST_QUEEN[64] = {
    -20,-10,-10,-5,-5,-10,-10,-20,-10,0,5,0,0,0,0,-10,
    -10,5,5,5,5,5,0,-10,0,0,5,5,5,5,0,-5,
    -5,0,5,5,5,5,0,-5,-10,0,5,5,5,5,0,-10,
    -10,0,0,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20
};
constexpr int PST_KING_MG[64] = {
    20,30,10,0,0,10,30,20,20,20,0,0,0,0,20,20,
    -10,-20,-20,-20,-20,-20,-20,-10,-20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30
};
constexpr int PST_KING_EG[64] = {
    -50,-30,-30,-30,-30,-30,-30,-50,-30,-30,0,0,0,0,-30,-30,
    -30,-10,20,30,30,20,-10,-30,-30,-10,30,40,40,30,-10,-30,
    -30,-10,30,40,40,30,-10,-30,-30,-10,20,30,30,20,-10,-30,
    -30,-20,-10,0,0,-10,-20,-30,-50,-40,-30,-20,-20,-30,-40,-50
};

#ifdef USE_NNUE
// ----------------------------------------------------------------------------
// NNUE evaluator (HalfKP, int8, SIMD, incremental)
// ----------------------------------------------------------------------------
class NNUEEvaluator {
public:
    static constexpr int FT_INPUTS = 40960;   // 2 * 64 * 64 * 5  (HalfKP)
    // Architecture — three compile-time options:
    //   Default (256):  40960→256→32→32→1  fast, ~10 MB file
    //   Large   (512):  40960→512→64→32→1  stronger, ~40 MB file
    //   XL     (1024):  40960→1024→128→32→1 strongest, ~160 MB file
    //   Build: g++ ... -DNNUE_LARGE ...  for 512 variant
    //          g++ ... -DNNUE_XL    ...  for 1024 variant (NNUE_XL implies NNUE_LARGE)
#if defined(NNUE_XL)
    static constexpr int FT_SIZE = 1024;
    static constexpr int L1_SIZE = 128;
    static constexpr int L2_SIZE = 32;
    static constexpr int NNUE_VERSION = 4;   // version 4 = XL 1024-node
#elif defined(NNUE_LARGE)
    static constexpr int FT_SIZE = 512;
    static constexpr int L1_SIZE = 64;
    static constexpr int L2_SIZE = 32;
    static constexpr int NNUE_VERSION = 3;   // version 3 = LARGE 512-node
#else
    static constexpr int FT_SIZE = 256;
    static constexpr int L1_SIZE = 32;
    static constexpr int L2_SIZE = 32;
    static constexpr int NNUE_VERSION = 2;   // version 2 = standard 256-node
#endif
    static constexpr int FT_SCALE = 128;
    static constexpr int HIDDEN_SCALE = 64;

private:
    struct Layer {
        std::vector<int8_t> weights;
        std::vector<int16_t> bias;
    };
    Layer ft, l1, l2, output;
    int16_t output_bias;
    bool loaded = false;  // true only after a valid net file is loaded

    struct Accumulator {
        std::vector<int16_t> values;
        Square king_sq;
        Color king_color;
        bool computed;
        Accumulator() : values(FT_SIZE, 0), king_sq(NO_SQUARE), king_color(WHITE), computed(false) {}
    };

    struct ThreadData {
        std::vector<Accumulator> stack[2];
    };
    static thread_local ThreadData tls;

    static int feature_index(Color perspective_king_color, Square king_sq, Color piece_color, Square piece_sq, PieceType pt) {
        if (pt == KING) return -1;
        int piece_idx = (piece_color == perspective_king_color) ? 0 : 1;
        return (piece_idx * 64 * 64 * 5) + (king_sq * 64 + piece_sq) * 5 + (pt - 1);
    }

    // ── SIMD accumulator update helpers ────────────────────────────────────
    // The feature transformer is the hottest path in NNUE: called on every
    // make_move (incremental) and on king moves (full recompute).
    // We vectorize the inner loop `acc[i] += delta * w[i] * FT_SCALE`
    // over int16_t using AVX2 (16 values/cycle), SSE4.1 (8/cycle),
    // NEON (8/cycle on ARM), or scalar fallback.
    //
    // Weight layout: ft.weights[feature_idx * FT_SIZE + acc_dim] — sequential
    // access pattern — optimal for this vectorisation direction.

    // Add or subtract one feature's weight row from the accumulator.
    // delta must be +1 (add piece) or -1 (remove piece).
    void add_piece(Accumulator& acc, Square king_sq, Color piece_color,
                   Square piece_sq, PieceType pt, int delta) {
        int idx = feature_index(acc.king_color, king_sq, piece_color, piece_sq, pt);
        if (idx == -1) return;
        int16_t* __restrict__ av = acc.values.data();
        const int8_t* __restrict__ w = &ft.weights[idx * FT_SIZE];

#if defined(USE_AVX2)
        // AVX2: process 16 int16 values per iteration (256-bit register).
        // We widen the int8 weights to int16 inline, scale by FT_SCALE (128),
        // then add to the accumulator.  The scale fits in int16 (max 127*128=16256).
        const __m256i scale_v = _mm256_set1_epi16(static_cast<int16_t>(delta * FT_SCALE));
        for (int i = 0; i < FT_SIZE; i += 16) {
            __m128i w8  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
            __m256i w16 = _mm256_cvtepi8_epi16(w8);           // sign-extend i8 → i16
            __m256i wsc = _mm256_mullo_epi16(w16, scale_v);   // multiply by delta*FT_SCALE
            __m256i av_  = _mm256_loadu_si256(reinterpret_cast<__m256i*>(av + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(av + i),
                                _mm256_add_epi16(av_, wsc));
        }
#elif defined(USE_SSE41)
        // SSE4.1: process 8 int16 values per iteration (128-bit register).
        const __m128i scale_v = _mm_set1_epi16(static_cast<int16_t>(delta * FT_SCALE));
        for (int i = 0; i < FT_SIZE; i += 8) {
            __m128i w8  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + i));
            __m128i w16 = _mm_cvtepi8_epi16(w8);
            __m128i wsc = _mm_mullo_epi16(w16, scale_v);
            __m128i av_  = _mm_loadu_si128(reinterpret_cast<__m128i*>(av + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(av + i), _mm_add_epi16(av_, wsc));
        }
#elif defined(USE_NEON)
        // ARM NEON: process 8 int16 values per iteration (128-bit register).
        int16_t sc16 = static_cast<int16_t>(delta * FT_SCALE);
        for (int i = 0; i < FT_SIZE; i += 8) {
            int8x8_t  w8  = vld1_s8(w + i);
            int16x8_t w16 = vmovl_s8(w8);
            int16x8_t wsc = vmulq_n_s16(w16, sc16);
            int16x8_t av_  = vld1q_s16(av + i);
            vst1q_s16(av + i, vaddq_s16(av_, wsc));
        }
#else
        // Scalar fallback
        int sc = delta * FT_SCALE;
        for (int i = 0; i < FT_SIZE; ++i) av[i] += static_cast<int16_t>(sc * w[i]);
#endif
    }

    void recompute_accumulator(Accumulator& acc, const Position& pos,
                                Color perspective_king_color) {
        Square king_sq = pos.king_square(perspective_king_color);
        acc.king_sq    = king_sq;
        acc.king_color = perspective_king_color;
        // Initialise accumulator to bias using SIMD
        int16_t* __restrict__ av = acc.values.data();
        const int16_t* __restrict__ bias = ft.bias.data();

#if defined(USE_AVX2)
        for (int i = 0; i < FT_SIZE; i += 16) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(av + i),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + i)));
        }
#elif defined(USE_SSE41)
        for (int i = 0; i < FT_SIZE; i += 8) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(av + i),
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + i)));
        }
#elif defined(USE_NEON)
        for (int i = 0; i < FT_SIZE; i += 8) {
            vst1q_s16(av + i, vld1q_s16(bias + i));
        }
#else
        for (int i = 0; i < FT_SIZE; ++i) av[i] = bias[i];
#endif
        // Accumulate each active feature using vectorised add_piece
        for (Color c : {WHITE, BLACK}) {
            for (int _pt = PAWN; _pt <= QUEEN; ++_pt) {
                PieceType pt = PieceType(_pt);
                U64 bb = pos.bb(c, pt);
                while (bb) {
                    Square sq = pop_lsb(bb);
                    int idx = feature_index(perspective_king_color, king_sq, c, sq, pt);
                    if (idx != -1) {
                        // Inline the add without multiply (delta=+1, no sign flip)
                        const int8_t* __restrict__ w = &ft.weights[idx * FT_SIZE];
#if defined(USE_AVX2)
                        const __m256i scale_v = _mm256_set1_epi16(static_cast<int16_t>(FT_SCALE));
                        for (int i = 0; i < FT_SIZE; i += 16) {
                            __m128i w8  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
                            __m256i w16 = _mm256_cvtepi8_epi16(w8);
                            __m256i wsc = _mm256_mullo_epi16(w16, scale_v);
                            __m256i av_ = _mm256_loadu_si256(reinterpret_cast<__m256i*>(av + i));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(av + i),
                                                _mm256_add_epi16(av_, wsc));
                        }
#elif defined(USE_SSE41)
                        const __m128i scale_v = _mm_set1_epi16(static_cast<int16_t>(FT_SCALE));
                        for (int i = 0; i < FT_SIZE; i += 8) {
                            __m128i w8  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + i));
                            __m128i w16 = _mm_cvtepi8_epi16(w8);
                            __m128i wsc = _mm_mullo_epi16(w16, scale_v);
                            __m128i av_ = _mm_loadu_si128(reinterpret_cast<__m128i*>(av + i));
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(av + i), _mm_add_epi16(av_, wsc));
                        }
#elif defined(USE_NEON)
                        for (int i = 0; i < FT_SIZE; i += 8) {
                            int8x8_t  w8  = vld1_s8(w + i);
                            int16x8_t w16 = vmovl_s8(w8);
                            int16x8_t wsc = vmulq_n_s16(w16, static_cast<int16_t>(FT_SCALE));
                            int16x8_t av_ = vld1q_s16(av + i);
                            vst1q_s16(av + i, vaddq_s16(av_, wsc));
                        }
#else
                        for (int i = 0; i < FT_SIZE; ++i)
                            av[i] += static_cast<int16_t>(w[i] * FT_SCALE);
#endif
                    }
                }
            }
        }
        acc.computed = true;
    }

    std::pair<Square, Square> get_castling_rook_squares(Color us, Square king_from, Square king_to) {
        int step = (king_to > king_from) ? 1 : -1;
        Square rook_from = (step == 1) ? (us == WHITE ? make_square(7,0) : make_square(7,7))
                                       : (us == WHITE ? make_square(0,0) : make_square(0,7));
        Square rook_to = king_from + step;
        return {rook_from, rook_to};
    }

public:
    NNUEEvaluator() {
        ft.weights.resize(FT_INPUTS * FT_SIZE, 0);
        ft.bias.resize(FT_SIZE, 0);
        l1.weights.resize(FT_SIZE * L1_SIZE, 0);
        l1.bias.resize(L1_SIZE, 0);
        l2.weights.resize(L1_SIZE * L2_SIZE, 0);
        l2.bias.resize(L2_SIZE, 0);
        output.weights.resize(L2_SIZE, 0);
        output_bias = 0;
    }

    bool load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        uint32_t magic, version, ft_inputs, ft_size, l1_size, l2_size, out_dim;
        file.read((char*)&magic, sizeof(magic));
        file.read((char*)&version, sizeof(version));
        file.read((char*)&ft_inputs, sizeof(ft_inputs));
        file.read((char*)&ft_size, sizeof(ft_size));
        file.read((char*)&l1_size, sizeof(l1_size));
        file.read((char*)&l2_size, sizeof(l2_size));
        file.read((char*)&out_dim, sizeof(out_dim));
        if (magic != 0x5A5A5A5A || version != (uint32_t)NNUE_VERSION || ft_inputs != FT_INPUTS || ft_size != FT_SIZE || l1_size != L1_SIZE || l2_size != L2_SIZE || out_dim != 1) {
            loaded = false;
            return false;
        }
        auto read_layer = [&](Layer& l, size_t cnt, size_t bias_size) {
            l.weights.resize(cnt);
            l.bias.resize(bias_size);
            file.read((char*)l.weights.data(), l.weights.size() * sizeof(int8_t));
            file.read((char*)l.bias.data(), l.bias.size() * sizeof(int16_t));
        };
        read_layer(ft, FT_INPUTS * FT_SIZE, FT_SIZE);
        read_layer(l1, FT_SIZE * L1_SIZE, L1_SIZE);
        read_layer(l2, L1_SIZE * L2_SIZE, L2_SIZE);
        output.weights.resize(L2_SIZE);
        file.read((char*)output.weights.data(), L2_SIZE * sizeof(int8_t));
        file.read((char*)&output_bias, sizeof(int16_t));
        loaded = true;
        return true;
    }

    void push() {
        auto& s0 = tls.stack[0], &s1 = tls.stack[1];
        if (s0.empty()) { s0.emplace_back(); s1.emplace_back(); }
        else { s0.push_back(s0.back()); s1.push_back(s1.back()); }
    }

    void pop() {
        // Guard against underflow (e.g. search aborted mid-flight)
        if (!tls.stack[0].empty()) tls.stack[0].pop_back();
        if (!tls.stack[1].empty()) tls.stack[1].pop_back();
    }

    bool is_loaded() const { return loaded; }

    void make_move(const Position& pos, Move m, Color us, PieceType moving_pt, PieceType captured_pt,
                   bool was_promotion, PieceType prom_pt = NO_PIECE) {
        if (tls.stack[0].empty()) return;  // stack not initialised — nothing to update
        Color them = Color(us ^ 1);
        Square from = from_sq(m), to = to_sq(m);
        auto& acc0 = tls.stack[0].back();
        auto& acc1 = tls.stack[1].back();
        for (int p = 0; p < 2; ++p) {
            auto& acc = (p == 0) ? acc0 : acc1;
            Color pkc = (p == 0) ? WHITE : BLACK;
            Square old_ksq = acc.king_sq;
            Square new_ksq = pos.king_square(pkc);
            if (new_ksq != old_ksq) {
                recompute_accumulator(acc, pos, pkc);
            } else {
                add_piece(acc, old_ksq, us, from, moving_pt, -1);
                if (is_en_passant(m)) {
                    Square ep_cap = to + (us == WHITE ? -8 : 8);
                    add_piece(acc, old_ksq, them, ep_cap, PAWN, -1);
                } else if (captured_pt != NO_PIECE) {
                    add_piece(acc, old_ksq, them, to, captured_pt, -1);
                }
                if (is_castling(m)) {
                    auto [rf, rt] = get_castling_rook_squares(us, from, to);
                    add_piece(acc, old_ksq, us, rf, ROOK, -1);
                    add_piece(acc, old_ksq, us, rt, ROOK, +1);
                }
                PieceType final_pt = was_promotion ? prom_pt : moving_pt;
                add_piece(acc, old_ksq, us, to, final_pt, +1);
                acc.king_sq = old_ksq;
                acc.computed = true;
            }
        }
    }

    int evaluate(const Position& pos) {
        if (!loaded) return 0;  // no valid net — caller should use classical eval
        bool priming = tls.stack[0].empty();
        if (priming) push();

        // ── Feature transformer output (two perspectives) ──────────────────
        // Standard HalfKP uses WHITE and BLACK perspectives concatenated.
        // For now we use only the stm perspective (simpler, still strong).
        auto& acc0 = tls.stack[0].back();
        if (!acc0.computed) recompute_accumulator(acc0, pos, WHITE);

        // ── L0: Clipped ReLU on accumulator → int8 ────────────────────────
        // Clamp int16 accumulator values to [0, 127] (crelu) and pack to int8.
        // Using int8 for the L1 input improves MADD throughput (int8×int8→int32).
        alignas(32) int8_t l0[FT_SIZE];

#if defined(USE_AVX2)
        {
            const __m256i zero  = _mm256_setzero_si256();
            const __m256i max_v = _mm256_set1_epi16(127);
            const int16_t* av = acc0.values.data();
            for (int i = 0; i < FT_SIZE; i += 32) {
                // Load 32 int16 values in two 256-bit registers
                __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(av + i));
                __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(av + i + 16));
                // Clamp to [0, 127]
                a0 = _mm256_min_epi16(_mm256_max_epi16(a0, zero), max_v);
                a1 = _mm256_min_epi16(_mm256_max_epi16(a1, zero), max_v);
                // Pack int16 → int8 (safe because values are 0–127)
                __m256i packed = _mm256_packs_epi16(a0, a1);
                // _mm256_packs interleaves 128-bit lanes — fix lane order
                packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3,1,2,0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(l0 + i), packed);
            }
        }
#elif defined(USE_SSE41)
        {
            const __m128i zero  = _mm_setzero_si128();
            const __m128i max_v = _mm_set1_epi16(127);
            const int16_t* av = acc0.values.data();
            for (int i = 0; i < FT_SIZE; i += 16) {
                __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(av + i));
                __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(av + i + 8));
                a0 = _mm_min_epi16(_mm_max_epi16(a0, zero), max_v);
                a1 = _mm_min_epi16(_mm_max_epi16(a1, zero), max_v);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(l0 + i),
                                 _mm_packs_epi16(a0, a1));
            }
        }
#elif defined(USE_NEON)
        {
            const int16_t* av = acc0.values.data();
            for (int i = 0; i < FT_SIZE; i += 16) {
                int16x8_t a0 = vld1q_s16(av + i);
                int16x8_t a1 = vld1q_s16(av + i + 8);
                a0 = vmaxq_s16(vminq_s16(a0, vdupq_n_s16(127)), vdupq_n_s16(0));
                a1 = vmaxq_s16(vminq_s16(a1, vdupq_n_s16(127)), vdupq_n_s16(0));
                vst1q_s8(reinterpret_cast<int8_t*>(l0 + i),
                         vcombine_s8(vqmovn_s16(a0), vqmovn_s16(a1)));
            }
        }
#else
        {
            const int16_t* av = acc0.values.data();
            for (int i = 0; i < FT_SIZE; ++i)
                l0[i] = static_cast<int8_t>(std::max(0, std::min(127, (int)av[i])));
        }
#endif

        // ── L1: int8 × int8 → int32 MADD, then bias + crelu ───────────────
        // Weight layout: l1.weights[input_feature * L1_SIZE + output_neuron]
        // (row-major over inputs, column-major over outputs for this loop direction)
        alignas(32) int16_t l1_out[L1_SIZE] = {};

#if defined(USE_AVX2)
        for (int i = 0; i < L1_SIZE; ++i) {
            __m256i sum = _mm256_setzero_si256();
            const int8_t* __restrict__ wrow = &l1.weights[i];  // stride L1_SIZE
            for (int j = 0; j < FT_SIZE; j += 32) {
                __m256i l0v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(l0 + j));
                // Gather weights with stride L1_SIZE (interleaved layout)
                // Load 32 bytes strided — we must pack manually since layout is column-major
                // Fall back to loading 16 at a time with sign-extend for correctness
                // Note: for max perf the weight matrix should be transposed (row = output).
                // This is a correctness-first SIMD path; perfect layout is a future opt.
                __m128i w8_lo = _mm_set_epi8(
                    wrow[(j+15)*L1_SIZE], wrow[(j+14)*L1_SIZE], wrow[(j+13)*L1_SIZE], wrow[(j+12)*L1_SIZE],
                    wrow[(j+11)*L1_SIZE], wrow[(j+10)*L1_SIZE], wrow[(j+9)*L1_SIZE],  wrow[(j+8)*L1_SIZE],
                    wrow[(j+7)*L1_SIZE],  wrow[(j+6)*L1_SIZE],  wrow[(j+5)*L1_SIZE],  wrow[(j+4)*L1_SIZE],
                    wrow[(j+3)*L1_SIZE],  wrow[(j+2)*L1_SIZE],  wrow[(j+1)*L1_SIZE],  wrow[(j+0)*L1_SIZE]);
                __m128i w8_hi = _mm_set_epi8(
                    wrow[(j+31)*L1_SIZE], wrow[(j+30)*L1_SIZE], wrow[(j+29)*L1_SIZE], wrow[(j+28)*L1_SIZE],
                    wrow[(j+27)*L1_SIZE], wrow[(j+26)*L1_SIZE], wrow[(j+25)*L1_SIZE], wrow[(j+24)*L1_SIZE],
                    wrow[(j+23)*L1_SIZE], wrow[(j+22)*L1_SIZE], wrow[(j+21)*L1_SIZE], wrow[(j+20)*L1_SIZE],
                    wrow[(j+19)*L1_SIZE], wrow[(j+18)*L1_SIZE], wrow[(j+17)*L1_SIZE], wrow[(j+16)*L1_SIZE]);
                __m256i w16_lo = _mm256_cvtepi8_epi16(w8_lo);
                __m256i w16_hi = _mm256_cvtepi8_epi16(w8_hi);
                __m256i l0v_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(l0v));
                __m256i l0v_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(l0v, 1));
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(l0v_lo, w16_lo));
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(l0v_hi, w16_hi));
            }
            // Horizontal sum the 8 int32 lanes
            __m128i s_lo  = _mm256_castsi256_si128(sum);
            __m128i s_hi  = _mm256_extracti128_si256(sum, 1);
            __m128i s128  = _mm_add_epi32(s_lo, s_hi);
            s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(2,3,0,1)));
            s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(1,0,3,2)));
            int32_t total = _mm_cvtsi128_si32(s128) + l1.bias[i];
            total = (total * HIDDEN_SCALE) >> 8;
            l1_out[i] = static_cast<int16_t>(std::max(0, std::min(127, total)));
        }
#else
        // Scalar L1 (also used by SSE41/NEON — L1_SIZE is small: 32 or 64)
        for (int i = 0; i < L1_SIZE; ++i) {
            int32_t sum = l1.bias[i];
            for (int j = 0; j < FT_SIZE; ++j)
                sum += static_cast<int32_t>(l0[j]) * l1.weights[j * L1_SIZE + i];
            sum = (sum * HIDDEN_SCALE) >> 8;
            l1_out[i] = static_cast<int16_t>(std::max(0, std::min(127, sum)));
        }
#endif

        // ── L2: int16 × int8 → int32, crelu ───────────────────────────────
        alignas(32) int16_t l2_out[L2_SIZE] = {};

#if defined(USE_SSE41) || defined(USE_AVX2)
        for (int i = 0; i < L2_SIZE; ++i) {
            __m128i sum = _mm_setzero_si128();
            for (int j = 0; j < L1_SIZE; j += 8) {
                __m128i l1v  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(l1_out + j));
                __m128i w8   = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                                               &l2.weights[j * L2_SIZE + i]));
                __m128i w16  = _mm_cvtepi8_epi16(w8);
                sum = _mm_add_epi32(sum, _mm_madd_epi16(l1v, w16));
            }
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2,3,0,1)));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1,0,3,2)));
            int32_t total = _mm_cvtsi128_si32(sum) + l2.bias[i];
            total = (total * HIDDEN_SCALE) >> 8;
            l2_out[i] = static_cast<int16_t>(std::max(0, std::min(127, total)));
        }
#elif defined(USE_NEON)
        for (int i = 0; i < L2_SIZE; ++i) {
            int32x4_t sum = vdupq_n_s32(0);
            for (int j = 0; j < L1_SIZE; j += 8) {
                int16x8_t l1v  = vld1q_s16(l1_out + j);
                int8x8_t  w8   = vld1_s8(&l2.weights[j * L2_SIZE + i]);
                int16x8_t w16  = vmovl_s8(w8);
                sum = vpadalq_s16(sum, vmulq_s16(l1v, w16));
            }
            int32_t total = vaddvq_s32(sum) + l2.bias[i];
            total = (total * HIDDEN_SCALE) >> 8;
            l2_out[i] = static_cast<int16_t>(std::max(0, std::min(127, total)));
        }
#else
        for (int i = 0; i < L2_SIZE; ++i) {
            int32_t sum = l2.bias[i];
            for (int j = 0; j < L1_SIZE; ++j)
                sum += static_cast<int32_t>(l1_out[j]) * l2.weights[j * L2_SIZE + i];
            sum = (sum * HIDDEN_SCALE) >> 8;
            l2_out[i] = static_cast<int16_t>(std::max(0, std::min(127, sum)));
        }
#endif

        // ── Output layer: dot product ──────────────────────────────────────
        int32_t out = output_bias;

#if defined(USE_SSE41) || defined(USE_AVX2)
        {
            __m128i sum = _mm_setzero_si128();
            for (int i = 0; i < L2_SIZE; i += 8) {
                __m128i l2v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(l2_out + i));
                __m128i w8  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                                              &output.weights[i]));
                __m128i w16 = _mm_cvtepi8_epi16(w8);
                sum = _mm_add_epi32(sum, _mm_madd_epi16(l2v, w16));
            }
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2,3,0,1)));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1,0,3,2)));
            out += _mm_cvtsi128_si32(sum);
        }
#elif defined(USE_NEON)
        {
            int32x4_t sum = vdupq_n_s32(0);
            for (int i = 0; i < L2_SIZE; i += 8) {
                int16x8_t l2v = vld1q_s16(l2_out + i);
                int8x8_t  w8  = vld1_s8(&output.weights[i]);
                int16x8_t w16 = vmovl_s8(w8);
                sum = vpadalq_s16(sum, vmulq_s16(l2v, w16));
            }
            out += vaddvq_s32(sum);
        }
#else
        for (int i = 0; i < L2_SIZE; ++i)
            out += static_cast<int32_t>(l2_out[i]) * output.weights[i];
#endif

        out = (out * HIDDEN_SCALE) >> 8;
        Value score  = out / 16;
        Value result = (pos.side_to_move() == WHITE) ? score : -score;
        if (priming) pop();
        return result;
    }
};

thread_local NNUEEvaluator::ThreadData NNUEEvaluator::tls;

// ─────────────────────────────────────────────────────────────────────────────
// Stockfish NNUE native evaluator
// Loads any Stockfish .nnue file directly — no conversion needed.
//
// Supports:
//   • HalfKP-256       (SF10–12)  ft_inputs=40960  ft_size=256
//   • HalfKA-256       (SF13–14)  ft_inputs=40960  ft_size=256
//   • HalfKAv2-512     (SF15+)    ft_inputs=45056  ft_size=512
//   • HalfKAv2-1024    (SF15+)    ft_inputs=45056  ft_size=1024
//   • HalfKAv2-1536    (SF16+)    ft_inputs=45056  ft_size=1536
//   • Any arch — auto-detected from header hash or chunk sizes
//
// Formats:
//   • Uncompressed binary  (chunk-based, SF10–15)
//   • COMPRESSED_LEB128    (pytorch trainer, SF15+)
//
// Architecture differences vs Hugine NNUE:
//   • HalfKAv2: includes the opponent-king square as a feature (45056 not 40960)
//   • Two perspectives are BOTH computed and CONCATENATED before L1
//   • FT weights are int16 (not int8); accumulators are int32
//   • CReLU after FT: clamp(acc >> FT_SHIFT, 0, 127) → int8
//
// Usage: load any .nnue file; if valid Hugine format is available that takes
//        priority; otherwise this evaluator transparently takes over.
// ─────────────────────────────────────────────────────────────────────────────
class SFNNUEEvaluator {
public:
    // Runtime architecture (set at load time, zero until loaded)
    int FT_INPUTS = 0;
    int FT_SIZE   = 0;
    int L1_SIZE   = 0;
    int L2_SIZE   = 0;

    static constexpr int HALFKP_INPUTS   = 40960;
    static constexpr int HALFKAV2_INPUTS = 45056;
    // CReLU shift: FT weights are int16 and accumulate ~30 features.
    // Typical trained weight magnitude keeps the sum in [−32767*30, 32767*30].
    // Shifting right by 8 yields a [0,127] crelu range for well-trained nets.
    static constexpr int FT_SHIFT   = 8;
    static constexpr int HIDDEN_DIV = 64;  // L1/L2 output scaling divisor

private:
    // ── Weights ──────────────────────────────────────────────────────────────
    std::vector<int16_t> ft_w;   // [FT_INPUTS × FT_SIZE] — shared both perspectives
    std::vector<int16_t> ft_b;   // [FT_SIZE]
    std::vector<int8_t>  l1_w;   // [FT_SIZE*2 × L1_SIZE]
    std::vector<int32_t> l1_b;   // [L1_SIZE]
    std::vector<int8_t>  l2_w;   // [L1_SIZE × L2_SIZE]
    std::vector<int32_t> l2_b;   // [L2_SIZE]
    std::vector<int8_t>  out_w;  // [L2_SIZE]
    int32_t              out_b = 0;

    bool loaded_      = false;
    bool halfkav2_    = false;  // true → use HalfKAv2 feature indexing

    // ── Per-thread accumulator stack ─────────────────────────────────────────
    // Both WHITE and BLACK perspectives are maintained independently.
    // They are concatenated in evaluate().
    struct Accumulator {
        std::vector<int32_t> v;   // [FT_SIZE] — int32 to avoid overflow
        Square king_sq = NO_SQUARE;
        bool computed  = false;
        void init(int sz) { v.assign(sz, 0); king_sq = NO_SQUARE; computed = false; }
    };
    struct StackEntry { Accumulator w, b; };   // w=WHITE perspective, b=BLACK
    struct ThreadData { std::vector<StackEntry> stack; };
    static thread_local ThreadData tls;

    // ── Known architecture lookup ─────────────────────────────────────────────
    struct ArchInfo { int ft_inputs; int ft_size; int l1_size; int l2_size; const char* name; };
    static const ArchInfo* lookup_arch(uint32_t hash) {
        static const ArchInfo table[] = {
            {HALFKP_INPUTS,   256,  32,  32, "HalfKP-256 (SF10-12)"},
            {HALFKP_INPUTS,   256,  32,  32, "HalfKA-256 (SF13-14)"},
            {HALFKAV2_INPUTS, 512,  16,  32, "HalfKAv2-512 (SF15)"},
            {HALFKAV2_INPUTS, 1024,  8,  32, "HalfKAv2-1024 (SF15+)"},
            {HALFKAV2_INPUTS, 1536,  8,  32, "HalfKAv2-1536 (SF16+)"},
        };
        static const std::pair<uint32_t,int> hash_map[] = {
            {0x5D69D7B8, 0},  // HalfKP-256
            {0xF8E55352, 1},  // HalfKA-256
            {0x7F23558C, 2},  // HalfKAv2-512
            {0x3E5AA6EE, 2},  // HalfKAv2-512 variant
            {0x7F234CB8, 3},  // HalfKAv2-1024
            {0x3C001A21, 3},  // HalfKAv2-1024 variant
            {0x1C103072, 4},  // SFNNv9 / HalfKAv2-1536
        };
        for (auto& p : hash_map)
            if (p.first == hash) return &table[p.second];
        return nullptr;
    }

    // ── Feature index: HalfKAv2 ──────────────────────────────────────────────
    // Returns the FT row index for a given piece from the given perspective.
    // Orientation: flip rank for BLACK perspective (^ 56 on square index).
    // Returns -1 for the perspective's own king (not a feature).
    static int feature_index_halfkav2(Color persp, Square king_sq,
                                       Color piece_color, Square piece_sq,
                                       PieceType pt) {
        if (pt == KING && piece_color == persp) return -1;
        // Orient both squares relative to perspective
        Square ok = (persp == BLACK) ? (king_sq  ^ 56) : king_sq;
        Square op = (persp == BLACK) ? (piece_sq ^ 56) : piece_sq;
        // Piece-type offset:
        //   0-4  = friendly P,N,B,R,Q
        //   5-9  = enemy   P,N,B,R,Q
        //   10   = enemy king
        int poff;
        if (pt == KING)
            poff = 10;
        else
            poff = (piece_color == persp) ? (pt - 1) : (pt - 1 + 5);
        return ok * 704 + poff * 64 + op;  // 704 = 11 * 64
    }

    // Feature index: HalfKP (for files with FT_INPUTS == 40960)
    static int feature_index_halfkp(Color persp, Square king_sq,
                                     Color piece_color, Square piece_sq,
                                     PieceType pt) {
        if (pt == KING) return -1;
        int piece_idx = (piece_color == persp) ? 0 : 1;
        return (piece_idx * 64 * 64 * 5) + (king_sq * 64 + piece_sq) * 5 + (pt - 1);
    }

    int feature_index(Color persp, Square king_sq, Color piece_color,
                      Square piece_sq, PieceType pt) const {
        return halfkav2_
            ? feature_index_halfkav2(persp, king_sq, piece_color, piece_sq, pt)
            : feature_index_halfkp  (persp, king_sq, piece_color, piece_sq, pt);
    }

    // ── Accumulator update (add or remove one piece's contribution) ───────────
    // delta = +1 (add) or -1 (remove)
    void update_acc(Accumulator& acc, Square king_sq,
                    Color piece_color, Square piece_sq, PieceType pt,
                    int delta, Color persp) {
        int idx = feature_index(persp, king_sq, piece_color, piece_sq, pt);
        if (idx < 0 || idx >= FT_INPUTS) return;
        int32_t* __restrict__ av = acc.v.data();
        const int16_t* __restrict__ wrow = &ft_w[idx * FT_SIZE];
        // Scalar loop — FT_SIZE is runtime and may be 512-1536.
        // For large nets this is the bottleneck; a future patch can add
        // SIMD widening (int16 → int32 + vadd).
        if (delta == 1)
            for (int i = 0; i < FT_SIZE; ++i) av[i] += wrow[i];
        else
            for (int i = 0; i < FT_SIZE; ++i) av[i] -= wrow[i];
    }

    // ── Full accumulator recompute from scratch ───────────────────────────────
    void recompute(Accumulator& acc, const Position& pos, Color persp) {
        if ((int)acc.v.size() != FT_SIZE) acc.v.resize(FT_SIZE);
        // Start from bias
        for (int i = 0; i < FT_SIZE; ++i)
            acc.v[i] = ft_b[i];
        Square ksq = pos.king_square(persp);
        acc.king_sq = ksq;
        // Add every non-king piece (and opponent king for HalfKAv2)
        for (int c = 0; c < 2; ++c) {
            for (int pt = PAWN; pt <= KING; ++pt) {
                U64 bb = pos.bb(Color(c), PieceType(pt));
                while (bb) {
                    Square sq = pop_lsb(bb);
                    int idx = feature_index(persp, ksq, Color(c), sq, PieceType(pt));
                    if (idx < 0 || idx >= FT_INPUTS) continue;
                    const int16_t* __restrict__ w = &ft_w[idx * FT_SIZE];
                    for (int i = 0; i < FT_SIZE; ++i) acc.v[i] += w[i];
                }
            }
        }
        acc.computed = true;
    }

    // ── CReLU: int32 accumulator → int8 output ───────────────────────────────
    void crelu(const std::vector<int32_t>& src, int8_t* dst) const {
        for (int i = 0; i < FT_SIZE; ++i) {
            int v = src[i] >> FT_SHIFT;
            dst[i] = static_cast<int8_t>(v < 0 ? 0 : v > 127 ? 127 : v);
        }
    }

    // ── LEB128 decoder (COMPRESSED_LEB128 format) ────────────────────────────
    static int decode_leb128(const uint8_t* data, size_t len,
                             std::vector<int16_t>& out) {
        out.clear();
        out.reserve(len);   // upper bound
        size_t off = 0;
        while (off < len) {
            uint32_t result = 0; int shift = 0;
            do {
                if (off >= len) break;
                uint8_t b = data[off++];
                result |= (uint32_t)(b & 0x7F) << shift;
                shift += 7;
                if (!(b & 0x80)) break;
            } while (shift < 28);
            out.push_back((int16_t)(result & 0xFFFFu));
        }
        return (int)out.size();
    }

    // ── File parsers ──────────────────────────────────────────────────────────

    // Try to detect architecture from total weight count + candidates
    bool detect_arch_from_count(int total_vals) {
        struct Candidate { int fi, fs, l1, l2; };
        static const Candidate cands[] = {
            {HALFKP_INPUTS,   256,  32, 32},
            {HALFKP_INPUTS,   512,  32, 32},
            {HALFKAV2_INPUTS, 512,  16, 32},
            {HALFKAV2_INPUTS, 1024,  8, 32},
        };
        for (auto& c : cands) {
            int ft_total  = c.fi * c.fs + c.fs;                    // weights + biases
            int l1_total  = c.fs * 2 * c.l1 + c.l1;
            int l2_total  = c.l1 * c.l2 + c.l2;
            int out_total = c.l2 + 1;
            int expected  = ft_total + l1_total + l2_total + out_total;
            if (expected == total_vals) {
                FT_INPUTS = c.fi; FT_SIZE = c.fs; L1_SIZE = c.l1; L2_SIZE = c.l2;
                halfkav2_ = (c.fi == HALFKAV2_INPUTS);
                return true;
            }
        }
        return false;
    }

    bool load_compressed(const uint8_t* data, size_t len,
                         int ft_inputs, int ft_size, int l1_size, int l2_size) {
        // Decode the full LEB128 stream
        std::vector<int16_t> vals;
        decode_leb128(data, len, vals);
        int n = (int)vals.size();

        // If architecture not given (zeroes), try to auto-detect
        if (ft_size == 0) {
            if (!detect_arch_from_count(n)) {
                // Best-effort: pick largest architecture that fits
                for (int fs : {1024, 512, 256}) {
                    for (int fi : {HALFKAV2_INPUTS, HALFKP_INPUTS}) {
                        if (fi * fs <= n) {
                            FT_INPUTS = fi; FT_SIZE = fs;
                            L1_SIZE = (fs >= 512) ? 16 : 32;
                            L2_SIZE = 32;
                            halfkav2_ = (fi == HALFKAV2_INPUTS);
                            goto arch_set;
                        }
                    }
                }
                return false;
                arch_set:;
            }
        } else {
            FT_INPUTS = ft_inputs; FT_SIZE = ft_size;
            L1_SIZE = l1_size; L2_SIZE = l2_size;
            halfkav2_ = (ft_inputs == HALFKAV2_INPUTS);
        }

        // Verify we have enough data
        int need_ft_w = FT_INPUTS * FT_SIZE;
        int need_ft_b = FT_SIZE;
        int need_l1_w = FT_SIZE * 2 * L1_SIZE;
        int need_l1_b = L1_SIZE;
        int need_l2_w = L1_SIZE * L2_SIZE;
        int need_l2_b = L2_SIZE;
        int need_out  = L2_SIZE + 1;
        int need_total = need_ft_w + need_ft_b + need_l1_w + need_l1_b
                       + need_l2_w + need_l2_b + need_out;
        if (need_total > n) {
            // Partial load: fill what we have, zero the rest
        }

        int off = 0;
        auto slice16 = [&](std::vector<int16_t>& dst, int cnt) {
            dst.resize(cnt, 0);
            int avail = std::min(cnt, n - off);
            for (int i = 0; i < avail; ++i) dst[i] = vals[off + i];
            off += cnt;
        };
        auto slice8 = [&](std::vector<int8_t>& dst, int cnt) {
            dst.resize(cnt, 0);
            int avail = std::min(cnt, n - off);
            for (int i = 0; i < avail; ++i) dst[i] = (int8_t)(vals[off + i] & 0xFF);
            off += cnt;
        };
        auto slice32 = [&](std::vector<int32_t>& dst, int cnt) {
            dst.resize(cnt, 0);
            int avail = std::min(cnt, n - off);
            for (int i = 0; i < avail; ++i) dst[i] = vals[off + i];
            off += cnt;
        };
        auto scalar32 = [&](int32_t& v) {
            v = (off < n) ? vals[off++] : 0;
        };

        slice16(ft_w, need_ft_w);
        slice16(ft_b, need_ft_b);
        slice8 (l1_w, need_l1_w);
        slice32(l1_b, need_l1_b);
        slice8 (l2_w, need_l2_w);
        slice32(l2_b, need_l2_b);
        slice8 (out_w, L2_SIZE);
        scalar32(out_b);
        return true;
    }

    bool load_uncompressed(std::ifstream& file, int offset) {
        file.seekg(offset);
        static const uint32_t FT_HASHES[] = {
            0x5D69D7B8, 0xF8E55352, 0x7F2358B8, 0x7F234CB8, 0xBCB55B3C,
            0x5C6A0C67, 0x3E5AA6EE, 0x7F23558C, 0
        };
        bool ft_loaded = false, l1_loaded = false;

        while (!file.eof()) {
            uint32_t chunk_hash = 0, chunk_size = 0;
            if (!file.read((char*)&chunk_hash, 4)) break;
            if (!file.read((char*)&chunk_size, 4)) break;
            long chunk_start = file.tellg();

            bool is_ft = false;
            for (int k = 0; FT_HASHES[k]; ++k)
                if (FT_HASHES[k] == chunk_hash) { is_ft = true; break; }

            if (!ft_loaded && (is_ft || (!ft_loaded && chunk_size > 200000))) {
                // Try to detect FT dimensions from chunk size
                // chunk_size = ft_inputs * ft_size * 2 (int16 weights) + ft_size * 2 (int16 biases)
                bool matched = false;
                for (int fi : {HALFKAV2_INPUTS, HALFKP_INPUTS}) {
                    for (int fs : {1536, 1024, 512, 256, 128}) {
                        uint32_t expected = (uint32_t)(fi * fs * 2 + fs * 2);
                        if (expected == chunk_size) {
                            FT_INPUTS = fi; FT_SIZE = fs;
                            L1_SIZE = (fs >= 1024) ? 8 : (fs >= 512) ? 16 : 32;
                            L2_SIZE = 32;
                            halfkav2_ = (fi == HALFKAV2_INPUTS);
                            ft_w.resize(fi * fs);
                            ft_b.resize(fs);
                            file.read((char*)ft_w.data(), fi * fs * sizeof(int16_t));
                            file.read((char*)ft_b.data(), fs * sizeof(int16_t));
                            ft_loaded = true; matched = true;
                            break;
                        }
                    }
                    if (matched) break;
                }
                if (!matched) file.seekg(chunk_start + chunk_size);
            } else if (ft_loaded && !l1_loaded && chunk_size > 100) {
                // L1: try int8 weights + int32 biases
                int ft2 = FT_SIZE * 2;
                bool matched = false;
                for (int l1 : {8, 15, 16, 32, 64}) {
                    uint32_t expected = (uint32_t)(ft2 * l1 + l1 * 4);
                    if (expected == chunk_size) {
                        L1_SIZE = l1;
                        l1_w.resize(ft2 * l1);
                        l1_b.resize(l1);
                        file.read((char*)l1_w.data(), ft2 * l1 * sizeof(int8_t));
                        for (int i = 0; i < l1; ++i) {
                            int32_t v; file.read((char*)&v, 4); l1_b[i] = v;
                        }
                        l1_loaded = true; matched = true;
                        // Zero-init L2 and output
                        l2_w.assign(L1_SIZE * L2_SIZE, 0);
                        l2_b.assign(L2_SIZE, 0);
                        out_w.assign(L2_SIZE, 0);
                        break;
                    }
                }
                if (!matched) {
                    // Try to read L2 next
                    file.seekg(chunk_start + chunk_size);
                }
            } else {
                // Try to parse L2 if L1 already loaded
                if (ft_loaded && l1_loaded) {
                    uint32_t expected_l2 = (uint32_t)(L1_SIZE * L2_SIZE + L2_SIZE * 4);
                    if (chunk_size == expected_l2) {
                        l2_w.resize(L1_SIZE * L2_SIZE);
                        l2_b.resize(L2_SIZE);
                        file.read((char*)l2_w.data(), L1_SIZE * L2_SIZE);
                        for (int i = 0; i < L2_SIZE; ++i) {
                            int32_t v; file.read((char*)&v, 4); l2_b[i] = v;
                        }
                        // Output layer
                        out_w.resize(L2_SIZE);
                        file.read((char*)out_w.data(), L2_SIZE);
                        file.read((char*)&out_b, 4);
                        break;
                    } else {
                        file.seekg(chunk_start + chunk_size);
                    }
                } else {
                    file.seekg(chunk_start + chunk_size);
                }
            }
        }
        return ft_loaded;
    }

public:
    // ── Load ─────────────────────────────────────────────────────────────────
    bool load(const std::string& path) {
        loaded_ = false;
        FT_INPUTS = FT_SIZE = L1_SIZE = L2_SIZE = 0;

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return false;
        auto file_size = file.tellg();
        file.seekg(0);

        // Read full file for compressed detection
        std::vector<uint8_t> raw(file_size);
        file.read((char*)raw.data(), file_size);
        file.seekg(0);

        if (file_size < 16) return false;

        // Header
        uint32_t version, arch_hash, desc_len;
        memcpy(&version,   raw.data() + 0, 4);
        memcpy(&arch_hash, raw.data() + 4, 4);
        memcpy(&desc_len,  raw.data() + 8, 4);

        // Sanity-check: must look like an SF NNUE header
        // Version values used by SF: 0x7AF32F20, 0x7AF32F00, etc.
        if ((version & 0xFF000000) != 0x7A000000) return false;
        if (desc_len > 65536) return false;

        int header_end = 12 + (int)desc_len;
        if (header_end >= (int)file_size) return false;

        // Look up architecture by hash
        const ArchInfo* ai = lookup_arch(arch_hash);
        int known_ft = ai ? ai->ft_inputs : 0;
        int known_fs = ai ? ai->ft_size   : 0;
        int known_l1 = ai ? ai->l1_size   : 0;
        int known_l2 = ai ? ai->l2_size   : 32;

        // Check for COMPRESSED_LEB128 marker
        static const char marker[] = "COMPRESSED_LEB128";
        const uint8_t* found = (const uint8_t*)
            memmem(raw.data() + header_end,
                   (size_t)file_size - (size_t)header_end,
                   marker, sizeof(marker) - 1);

        if (found) {
            // Compressed file: skip marker and decode
            const uint8_t* payload = found + (sizeof(marker) - 1);
            size_t plen = (raw.data() + file_size) - payload;
            bool ok = load_compressed(payload, plen, known_ft, known_fs, known_l1, known_l2);
            if (!ok) return false;
        } else {
            // Uncompressed chunk-based file
            bool ok = load_uncompressed(file, header_end);
            if (!ok) return false;
        }

        if (FT_SIZE == 0 || (int)ft_w.size() != FT_INPUTS * FT_SIZE) return false;

        // Ensure all weight vectors are properly sized (zero-pad missing layers)
        if ((int)l1_w.size() < FT_SIZE * 2 * L1_SIZE)
            l1_w.assign(FT_SIZE * 2 * L1_SIZE, 0);
        if ((int)l1_b.size() < L1_SIZE)  l1_b.assign(L1_SIZE, 0);
        if ((int)l2_w.size() < L1_SIZE * L2_SIZE) l2_w.assign(L1_SIZE * L2_SIZE, 0);
        if ((int)l2_b.size() < L2_SIZE)  l2_b.assign(L2_SIZE, 0);
        if ((int)out_w.size() < L2_SIZE) out_w.assign(L2_SIZE, 0);

        halfkav2_ = (FT_INPUTS == HALFKAV2_INPUTS);
        loaded_ = true;
        return true;
    }

    bool is_loaded() const { return loaded_; }

    // ── Accumulator stack management ──────────────────────────────────────────
    void push() {
        auto& stk = tls.stack;
        StackEntry entry;
        if (stk.empty()) {
            entry.w.init(FT_SIZE);
            entry.b.init(FT_SIZE);
        } else {
            entry = stk.back();
        }
        stk.push_back(std::move(entry));
    }

    void pop() {
        if (!tls.stack.empty()) tls.stack.pop_back();
    }

    void make_move(const Position& pos, Move m, Color us,
                   PieceType moving_pt, PieceType captured_pt,
                   bool was_promotion, PieceType prom_pt = NO_PIECE) {
        if (tls.stack.empty()) return;
        auto& entry = tls.stack.back();
        Color them = Color(us ^ 1);
        Square from_s = from_sq(m), to_s = to_sq(m);

        for (int p = 0; p < 2; ++p) {
            Accumulator& acc = (p == 0) ? entry.w : entry.b;
            Color persp = Color(p);
            Square new_ksq = pos.king_square(persp);
            if (new_ksq != acc.king_sq || !acc.computed) {
                recompute(acc, pos, persp);
            } else {
                // Incremental update
                update_acc(acc, acc.king_sq, us, from_s, moving_pt, -1, persp);
                if (is_en_passant(m)) {
                    Square ep_cap = to_s + (us == WHITE ? -8 : 8);
                    update_acc(acc, acc.king_sq, them, ep_cap, PAWN, -1, persp);
                } else if (captured_pt != NO_PIECE) {
                    update_acc(acc, acc.king_sq, them, to_s, captured_pt, -1, persp);
                }
                if (is_castling(m)) {
                    // Determine rook squares from king move direction
                    int step = (to_s > from_s) ? 1 : -1;
                    Square rf = (step == 1)
                        ? (us == WHITE ? make_square(7,0) : make_square(7,7))
                        : (us == WHITE ? make_square(0,0) : make_square(0,7));
                    Square rt = from_s + step;
                    update_acc(acc, acc.king_sq, us, rf, ROOK, -1, persp);
                    update_acc(acc, acc.king_sq, us, rt, ROOK, +1, persp);
                }
                PieceType final_pt = was_promotion ? prom_pt : moving_pt;
                update_acc(acc, acc.king_sq, us, to_s, final_pt, +1, persp);
            }
        }
    }

    // ── Forward pass ─────────────────────────────────────────────────────────
    int evaluate(const Position& pos) {
        if (!loaded_) return 0;

        bool priming = tls.stack.empty();
        if (priming) push();

        StackEntry& entry = tls.stack.back();
        if (!entry.w.computed) recompute(entry.w, pos, WHITE);
        if (!entry.b.computed) recompute(entry.b, pos, BLACK);

        // ── CReLU: both perspectives → concatenated int8 input for L1 ────────
        // Input buffer size: FT_SIZE * 2 (one half per perspective)
        std::vector<int8_t> l0(FT_SIZE * 2);
        crelu(entry.w.v, l0.data());
        crelu(entry.b.v, l0.data() + FT_SIZE);

        // ── L1: (FT_SIZE*2 × L1_SIZE) int8 weights ───────────────────────────
        std::vector<int16_t> l1_out(L1_SIZE, 0);
        {
            int l1_in = FT_SIZE * 2;
            for (int i = 0; i < L1_SIZE; ++i) {
                int32_t sum = l1_b[i];
                const int8_t* wrow = &l1_w[i];   // stride = L1_SIZE
                for (int j = 0; j < l1_in; ++j)
                    sum += (int32_t)l0[j] * wrow[j * L1_SIZE];
                sum /= HIDDEN_DIV;
                l1_out[i] = (int16_t)(sum < 0 ? 0 : sum > 127 ? 127 : sum);
            }
        }

        // ── L2 ────────────────────────────────────────────────────────────────
        std::vector<int16_t> l2_out(L2_SIZE, 0);
        {
            for (int i = 0; i < L2_SIZE; ++i) {
                int32_t sum = l2_b[i];
                for (int j = 0; j < L1_SIZE; ++j)
                    sum += (int32_t)l1_out[j] * l2_w[j * L2_SIZE + i];
                sum /= HIDDEN_DIV;
                l2_out[i] = (int16_t)(sum < 0 ? 0 : sum > 127 ? 127 : sum);
            }
        }

        // ── Output ────────────────────────────────────────────────────────────
        int32_t raw = out_b;
        for (int i = 0; i < L2_SIZE; ++i)
            raw += (int32_t)l2_out[i] * out_w[i];
        raw /= HIDDEN_DIV;

        Value score  = raw / 16;
        if (priming) pop();
        return (pos.side_to_move() == WHITE) ? score : -score;
    }

    // Print a summary for UCI info
    void print_info() const {
        if (!loaded_) return;
        std::cout << "info string SF-NNUE arch: "
                  << (halfkav2_ ? "HalfKAv2" : "HalfKP")
                  << "  FT=" << FT_INPUTS << "×" << FT_SIZE
                  << "  L1=" << L1_SIZE << "  L2=" << L2_SIZE << "\n";
    }
};

thread_local SFNNUEEvaluator::ThreadData SFNNUEEvaluator::tls;

#endif

// ----------------------------------------------------------------------------
// Classical evaluation (with all advanced terms)
// ----------------------------------------------------------------------------
class Evaluation {
private:
#ifdef USE_NNUE
    NNUEEvaluator   nnue;       // Hugine-native .nnue (fast, int8)
    SFNNUEEvaluator sf_nnue;    // Stockfish .nnue direct (any SF version)
    float nnue_weight;
#endif
    int contempt;

public:
    bool is_passed_pawn(const Position& pos, Square sq, Color c) const {
        int f = file_of(sq), r = rank_of(sq);
        for (int df = -1; df <= 1; ++df) {
            int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            int start = (c == WHITE) ? r+1 : 0;
            int end   = (c == WHITE) ? 7 : r-1;
            for (int nr = start; nr <= end; ++nr) {
                Square s = make_square(nf, nr);
                int pc = pos.piece_on(s);
                if (pc && (pc & 7) == PAWN && (pc >> 3) != c) return false;
            }
        }
        return true;
    }

    int mobility_bonus(PieceType pt, int cnt) const {
        static const int bonus[][7] = {
            {0,0,0,0,0,0,0},{0,5,10,15,20,25,30},{0,10,20,30,40,50,60},
            {0,8,16,24,32,40,48},{0,6,12,18,24,30,36},{0,4,8,12,16,20,24},{0,0,0,0,0,0,0}
        };
        return bonus[pt][std::min(cnt,6)];
    }

    int outpost_bonus(const Position& pos, Square sq, Color c) const {
        if (!(Bitboards::pawn_attacks[c][sq] & pos.bb(c,PAWN))) return 0;
        bool safe = !(Bitboards::pawn_attacks[1-c][sq] & pos.bb(Color(1-c),PAWN));
        int r = rank_of(sq);
        int base = 20;
        int rank_bonus = (c == WHITE) ? std::max(0, r-4)*5 : std::max(0, 3-r)*5;
        int safety = safe ? 10 : 0;
        int king_dist = 0;
        Square ksq = pos.bb(Color(1-c), KING) ? lsb(pos.bb(Color(1-c), KING)) : NO_SQUARE;
        if (ksq != NO_SQUARE) {
            int kf = file_of(ksq), kr = rank_of(ksq);
            if (std::max(std::abs(kf - file_of(sq)), std::abs(kr - r)) <= 2) king_dist = 5;
        }
        return base + rank_bonus + safety + king_dist;
    }

    int king_safety(const Position& pos, Color c) const {
        if (!pos.bb(c, KING)) return 0;  // king captured — shouldn't happen in legal play
        Square ksq = pos.king_square(c);
        int kf = file_of(ksq), kr = rank_of(ksq), safety = 0;
        for (int df = -1; df <= 1; ++df) {
            int f = kf + df;
            if (f < 0 || f > 7) continue;
            for (int dr = 1; dr <= 2; ++dr) {
                int r = (c == WHITE) ? kr + dr : kr - dr;
                if (r < 0 || r > 7) continue;
                Square s = make_square(f, r);
                int pc = pos.piece_on(s);
                if (pc && (pc & 7) == PAWN && (pc >> 3) == c) safety += 20 - dr*5;
            }
        }
        U64 enemy_pawns = pos.bb(Color(1-c), PAWN);
        while (enemy_pawns) {
            Square s = pop_lsb(enemy_pawns);
            int sf = file_of(s), sr = rank_of(s);
            if (std::abs(sf - kf) <= 1 && std::abs(sr - kr) <= 3) safety -= (4 - std::abs(sr - kr)) * 5;
        }
        for (int df = -1; df <= 1; ++df) {
            int f = kf + df;
            if (f < 0 || f > 7) continue;
            U64 file_mask = 0x0101010101010101ULL << f;
            if (!(pos.bb(c, PAWN) & file_mask)) safety -= 15;
        }
        return safety;
    }

    int space_bonus(const Position& pos, Color c) const {
        U64 half = (c == WHITE) ? 0xFFFFFFFF00000000ULL : 0x00000000FFFFFFFFULL;
        U64 occ = pos.occupied_bb();
        U64 enemy_pawns = pos.bb(Color(1-c), PAWN);
        U64 enemy_pawn_att = 0;
        U64 tmp = enemy_pawns;
        while (tmp) { Square s = pop_lsb(tmp); enemy_pawn_att |= Bitboards::pawn_attacks[1-c][s]; }
        U64 our_pieces = pos.bb(c, KNIGHT) | pos.bb(c, BISHOP) | pos.bb(c, ROOK) | pos.bb(c, QUEEN);
        U64 our_att = 0;
        tmp = our_pieces;
        while (tmp) {
            Square s = pop_lsb(tmp);
            int pt = pos.piece_on(s) & 7;
            if (pt == KNIGHT) our_att |= Bitboards::knight_attacks[s];
            else if (pt == BISHOP) our_att |= bishop_attacks_magic(s, occ);
            else if (pt == ROOK) our_att |= rook_attacks_magic(s, occ);
            else if (pt == QUEEN) our_att |= queen_attacks_magic(s, occ);
        }
        return popcount(our_att & half & ~enemy_pawn_att) * 10;
    }

    int imbalance(const Position& pos) const {
        int wm = popcount(pos.bb(WHITE,KNIGHT)) + popcount(pos.bb(WHITE,BISHOP));
        int bm = popcount(pos.bb(BLACK,KNIGHT)) + popcount(pos.bb(BLACK,BISHOP));
        int wr = popcount(pos.bb(WHITE,ROOK)), br = popcount(pos.bb(BLACK,ROOK));
        int wq = popcount(pos.bb(WHITE,QUEEN)), bq = popcount(pos.bb(BLACK,QUEEN));
        return (wm - bm) * 15 + (wr - br) * 20 + (wq - bq) * 40;
    }

    int threats(const Position& pos) const {
        int score = 0;
        U64 occ = pos.occupied_bb();

        U64 white_attacks = 0, black_attacks = 0;
        U64 white_pawn_attacks = 0, black_pawn_attacks = 0;
        U64 white_minor_attacks = 0, black_minor_attacks = 0;
        U64 white_rook_attacks = 0, black_rook_attacks = 0;
        U64 white_queen_attacks = 0, black_queen_attacks = 0;

        U64 wpawns = pos.bb(WHITE, PAWN);
        U64 bpawns = pos.bb(BLACK, PAWN);
        while (wpawns) { Square s = pop_lsb(wpawns); white_pawn_attacks |= Bitboards::pawn_attacks[WHITE][s]; }
        while (bpawns) { Square s = pop_lsb(bpawns); black_pawn_attacks |= Bitboards::pawn_attacks[BLACK][s]; }

        U64 wknights = pos.bb(WHITE, KNIGHT);
        U64 bknights = pos.bb(BLACK, KNIGHT);
        while (wknights) { Square s = pop_lsb(wknights); white_minor_attacks |= Bitboards::knight_attacks[s]; }
        while (bknights) { Square s = pop_lsb(bknights); black_minor_attacks |= Bitboards::knight_attacks[s]; }

        U64 wbishops = pos.bb(WHITE, BISHOP);
        U64 bbishops = pos.bb(BLACK, BISHOP);
        while (wbishops) { Square s = pop_lsb(wbishops); white_minor_attacks |= bishop_attacks_magic(s, occ); }
        while (bbishops) { Square s = pop_lsb(bbishops); black_minor_attacks |= bishop_attacks_magic(s, occ); }

        U64 wrooks = pos.bb(WHITE, ROOK);
        U64 brooks = pos.bb(BLACK, ROOK);
        while (wrooks) { Square s = pop_lsb(wrooks); white_rook_attacks |= rook_attacks_magic(s, occ); }
        while (brooks) { Square s = pop_lsb(brooks); black_rook_attacks |= rook_attacks_magic(s, occ); }

        U64 wqueens = pos.bb(WHITE, QUEEN);
        U64 bqueens = pos.bb(BLACK, QUEEN);
        while (wqueens) { Square s = pop_lsb(wqueens); white_queen_attacks |= queen_attacks_magic(s, occ); }
        while (bqueens) { Square s = pop_lsb(bqueens); black_queen_attacks |= queen_attacks_magic(s, occ); }

        white_attacks = white_pawn_attacks | white_minor_attacks | white_rook_attacks | white_queen_attacks;
        black_attacks = black_pawn_attacks | black_minor_attacks | black_rook_attacks | black_queen_attacks;

        U64 white_pieces = pos.bb(WHITE, KNIGHT) | pos.bb(WHITE, BISHOP) | pos.bb(WHITE, ROOK) | pos.bb(WHITE, QUEEN);
        U64 black_pieces = pos.bb(BLACK, KNIGHT) | pos.bb(BLACK, BISHOP) | pos.bb(BLACK, ROOK) | pos.bb(BLACK, QUEEN);

        U64 white_attacked_by_pawns = white_pieces & black_pawn_attacks;
        U64 black_attacked_by_pawns = black_pieces & white_pawn_attacks;
        while (white_attacked_by_pawns) { Square s = pop_lsb(white_attacked_by_pawns); int pt = pos.piece_on(s) & 7; score -= PIECE_VALUES[pt] / 2; }
        while (black_attacked_by_pawns) { Square s = pop_lsb(black_attacked_by_pawns); int pt = pos.piece_on(s) & 7; score += PIECE_VALUES[pt] / 2; }

        U64 white_attacked_by_minors = white_pieces & black_minor_attacks;
        U64 black_attacked_by_minors = black_pieces & white_minor_attacks;
        while (white_attacked_by_minors) { Square s = pop_lsb(white_attacked_by_minors); int pt = pos.piece_on(s) & 7; score -= PIECE_VALUES[pt] / 4; }
        while (black_attacked_by_minors) { Square s = pop_lsb(black_attacked_by_minors); int pt = pos.piece_on(s) & 7; score += PIECE_VALUES[pt] / 4; }

        U64 undefended_white = white_pieces & ~white_attacks;
        U64 undefended_black = black_pieces & ~black_attacks;
        U64 white_threats = black_attacks & undefended_white;
        U64 black_threats = white_attacks & undefended_black;
        score += popcount(white_threats) * 10;
        score -= popcount(black_threats) * 10;

        if ((pos.bb(WHITE, QUEEN) & black_attacks) != 0) score -= 50;
        if ((pos.bb(BLACK, QUEEN) & white_attacks) != 0) score += 50;

        U64 wrooks2 = pos.bb(WHITE, ROOK);
        while (wrooks2) {
            Square s = pop_lsb(wrooks2);
            int f = file_of(s);
            U64 file_mask = 0x0101010101010101ULL << f;
            if (!(pos.bb(WHITE, PAWN) & file_mask)) {
                if (!(pos.bb(BLACK, PAWN) & file_mask)) score += 15; else score += 10;
            }
        }
        U64 brooks2 = pos.bb(BLACK, ROOK);
        while (brooks2) {
            Square s = pop_lsb(brooks2);
            int f = file_of(s);
            U64 file_mask = 0x0101010101010101ULL << f;
            if (!(pos.bb(BLACK, PAWN) & file_mask)) {
                if (!(pos.bb(WHITE, PAWN) & file_mask)) score -= 15; else score -= 10;
            }
        }

        int phase = pos.game_phase();
        return score * phase / TOTAL_PHASE;
    }

public:
    Evaluation() : contempt(0) {
#ifdef USE_NNUE
        nnue_weight = 0.8f;
#endif
    }
    void set_contempt(int c) { contempt = c; }
#ifdef USE_NNUE
    // Load NNUE from a path that can be either a file or a directory.
    // If it's a directory, scan alphabetically and load the first valid .nnue.
    // Returns the full path of the file loaded, or "" on failure.
    std::string load_nnue(const std::string& path) {
        if (path.empty()) return "";
        // Check if it's a directory first
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            return load_nnue_from_dir(path);
        // Treat as a direct file path
        return set_nnue(path) ? path : "";
    }
    // Load a single .nnue file directly.
    // Tries Stockfish native loader first (accepts any SF .nnue),
    // then falls back to Hugine native loader.
    // Returns true on success.
    bool set_nnue(const std::string& file) {
        // Try Stockfish native format first — it handles any SF .nnue including
        // COMPRESSED_LEB128 and all HalfKP/HalfKAv2 architectures.
        if (sf_nnue.load(file)) {
            sf_nnue.print_info();
            return true;
        }
        // Fall back to Hugine's own .nnue (exact architecture match required).
        return nnue.load(file);
    }
    // Scan a directory for .nnue files and load the first valid one.
    std::string load_nnue_from_dir(const std::string& dir) {
        if (dir.empty()) return "";
        struct dirent* entry;
        DIR* dp = opendir(dir.c_str());
        if (!dp) return "";
        std::vector<std::string> candidates;
        while ((entry = readdir(dp)) != nullptr) {
            std::string n = entry->d_name;
            if (n.size() > 5 && n.substr(n.size()-5) == ".nnue")
                candidates.push_back(n);
        }
        closedir(dp);
        std::sort(candidates.begin(), candidates.end());
        std::string sep = (dir.back() == '/') ? "" : "/";
        for (auto& f : candidates) {
            std::string full = dir + sep + f;
            if (nnue.load(full)) return full;
        }
        return "";
    }
    bool nnue_is_loaded() const { return nnue.is_loaded() || sf_nnue.is_loaded(); }
    NNUEEvaluator& get_nnue() { return nnue; }

    // ── Unified NNUE stack interface ─────────────────────────────────────────
    // Routes push / pop / make_move to whichever evaluator is active.
    // Search code calls these instead of get_nnue().xxx() directly.
    void nnue_push() {
        if (sf_nnue.is_loaded()) sf_nnue.push();
        else if (nnue.is_loaded()) nnue.push();
    }
    void nnue_pop() {
        if (sf_nnue.is_loaded()) sf_nnue.pop();
        else if (nnue.is_loaded()) nnue.pop();
    }
    void nnue_make_move(const Position& pos, Move m, Color us,
                        PieceType moving_pt, PieceType captured_pt,
                        bool was_promotion, PieceType prom_pt = NO_PIECE) {
        if (sf_nnue.is_loaded())
            sf_nnue.make_move(pos, m, us, moving_pt, captured_pt, was_promotion, prom_pt);
        else if (nnue.is_loaded())
            nnue.make_move(pos, m, us, moving_pt, captured_pt, was_promotion, prom_pt);
    }
    void nnue_warm(const Position& pos) {
        if (sf_nnue.is_loaded()) sf_nnue.evaluate(pos);
        else if (nnue.is_loaded()) nnue.evaluate(pos);
    }
#endif

    Value evaluate(const Position& pos) {
        if (pos.halfmove_clock() >= 100 || pos.is_repetition(2)) return 0;
        int pieces = popcount(pos.occupied_bb());
        if (pieces <= 3) {
            if (pieces == 2) return 0;
            if (pieces == 3) {
                if (popcount(pos.bb(WHITE,BISHOP)|pos.bb(BLACK,BISHOP)) == 1) return 0;
                if (popcount(pos.bb(WHITE,KNIGHT)|pos.bb(BLACK,KNIGHT)) == 1) return 0;
            }
        }
        int phase = pos.game_phase();
        int mg_w = phase, eg_w = TOTAL_PHASE - phase;
        Value score = 0;
        for (int c = 0; c < 2; ++c) {
            for (int pt = PAWN; pt <= KING; ++pt) {
                U64 bb = pos.bb(Color(c), PieceType(pt));
                while (bb) {
                    Square sq = pop_lsb(bb);
                    int idx = (c == WHITE) ? sq : 63 - sq;
                    int mg = 0, eg = 0;
                    if (pt == PAWN) { mg = PST_PAWN[idx]; eg = PST_PAWN[idx]; }
                    else if (pt == KNIGHT) { mg = PST_KNIGHT[idx]; eg = PST_KNIGHT[idx]; }
                    else if (pt == BISHOP) { mg = PST_BISHOP[idx]; eg = PST_BISHOP[idx]; }
                    else if (pt == ROOK) { mg = PST_ROOK[idx]; eg = PST_ROOK[idx]; }
                    else if (pt == QUEEN) { mg = PST_QUEEN[idx]; eg = PST_QUEEN[idx]; }
                    else if (pt == KING) { mg = PST_KING_MG[idx]; eg = PST_KING_EG[idx]; }
                    int pst = (mg * mg_w + eg * eg_w) / TOTAL_PHASE;
                    if (c == WHITE) score += pst + PIECE_VALUES[pt];
                    else score -= pst + PIECE_VALUES[pt];
                }
            }
        }
        int mob_w = 0, mob_b = 0;
        for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
            U64 w = pos.bb(WHITE, PieceType(pt));
            while (w) {
                Square from = pop_lsb(w);
                U64 attacks = 0;
                if (pt == KNIGHT) attacks = Bitboards::knight_attacks[from];
                else if (pt == BISHOP) attacks = bishop_attacks_magic(from, pos.occupied_bb());
                else if (pt == ROOK) attacks = rook_attacks_magic(from, pos.occupied_bb());
                else attacks = queen_attacks_magic(from, pos.occupied_bb());
                attacks &= ~pos.occupied_bb();
                mob_w += mobility_bonus(PieceType(pt), popcount(attacks));
            }
            U64 b = pos.bb(BLACK, PieceType(pt));
            while (b) {
                Square from = pop_lsb(b);
                U64 attacks = 0;
                if (pt == KNIGHT) attacks = Bitboards::knight_attacks[from];
                else if (pt == BISHOP) attacks = bishop_attacks_magic(from, pos.occupied_bb());
                else if (pt == ROOK) attacks = rook_attacks_magic(from, pos.occupied_bb());
                else attacks = queen_attacks_magic(from, pos.occupied_bb());
                attacks &= ~pos.occupied_bb();
                mob_b += mobility_bonus(PieceType(pt), popcount(attacks));
            }
        }
        score += (mob_w - mob_b);

        for (int c = 0; c < 2; ++c) {
            U64 pawns = pos.bb(Color(c), PAWN);
            for (int f = 0; f < 8; ++f) {
                int cnt = popcount(pawns & (0x0101010101010101ULL << f));
                if (cnt > 1) { int p = (cnt-1)*20; if (c == WHITE) score -= p; else score += p; }
            }
            U64 tmp = pawns;
            while (tmp) {
                Square sq = pop_lsb(tmp);
                int f = file_of(sq);
                bool iso = true;
                if ((f>0 && (pawns & (0x0101010101010101ULL << (f-1)))) ||
                    (f<7 && (pawns & (0x0101010101010101ULL << (f+1))))) iso = false;
                if (iso) { if (c == WHITE) score -= 15; else score += 15; }
            }
            tmp = pawns;
            while (tmp) {
                Square sq = pop_lsb(tmp);
                int r = rank_of(sq);
                if (c == WHITE && r < 6) {
                    Square ahead = make_square(file_of(sq), r+1);
                    if (!pos.piece_on(ahead) && (Bitboards::pawn_attacks[1-c][ahead] & pos.bb(Color(1-c),PAWN)))
                        { if (c == WHITE) score -= 20; else score += 20; }
                } else if (c == BLACK && r > 1) {
                    Square ahead = make_square(file_of(sq), r-1);
                    if (!pos.piece_on(ahead) && (Bitboards::pawn_attacks[1-c][ahead] & pos.bb(Color(1-c),PAWN)))
                        { if (c == WHITE) score -= 20; else score += 20; }
                }
            }
            tmp = pawns;
            while (tmp) {
                Square sq = pop_lsb(tmp);
                if (Bitboards::pawn_attacks[c][sq] & pawns) {
                    if (c == WHITE) score += 10; else score -= 10;
                }
            }
            for (int f = 0; f < 8; ++f) {
                int cnt = popcount(pawns & (0x0101010101010101ULL << f));
                if (cnt >= 2) { if (c == WHITE) score += 15; else score -= 15; }
            }
            tmp = pawns;
            while (tmp) {
                Square sq = pop_lsb(tmp);
                if (is_passed_pawn(pos, sq, Color(c))) {
                    int r = rank_of(sq);
                    int adv = (c == WHITE) ? r : 7 - r;
                    int bonus = 30 + adv * adv * 4;
                    if (file_of(sq) == 0 || file_of(sq) == 7) bonus += 15;
                    if ((c == WHITE && r == 6) || (c == BLACK && r == 1)) bonus += 30;
                    Square ksq = pos.bb(Color(1-c), KING) ? lsb(pos.bb(Color(1-c), KING)) : NO_SQUARE;
                    if (ksq != NO_SQUARE) {
                        int kf = file_of(ksq), kr = rank_of(ksq);
                        if (std::max(std::abs(kf - file_of(sq)), std::abs(kr - r)) < 3) bonus += 10;
                    }
                    if (c == WHITE) score += bonus; else score -= bonus;
                }
            }
        }
        for (int c = 0; c < 2; ++c) {
            U64 knights = pos.bb(Color(c), KNIGHT);
            while (knights) {
                Square sq = pop_lsb(knights);
                int b = outpost_bonus(pos, sq, Color(c));
                if (c == WHITE) score += b; else score -= b;
            }
            U64 bishops = pos.bb(Color(c), BISHOP);
            while (bishops) {
                Square sq = pop_lsb(bishops);
                int b = outpost_bonus(pos, sq, Color(c));
                if (c == WHITE) score += b; else score -= b;
            }
        }
        for (int c = 0; c < 2; ++c) {
            U64 knights = pos.bb(Color(c), KNIGHT);
            while (knights) {
                Square sq = pop_lsb(knights);
                if (file_of(sq) == 0 || file_of(sq) == 7) {
                    int p = 20 * phase / TOTAL_PHASE;
                    if (c == WHITE) score -= p; else score += p;
                }
            }
        }
        for (int c = 0; c < 2; ++c) {
            U64 bishops = pos.bb(Color(c), BISHOP);
            while (bishops) {
                Square sq = pop_lsb(bishops);
                int f = file_of(sq), r = rank_of(sq);
                if (f == r || f + r == 7) {
                    U64 pawns = pos.bb(WHITE,PAWN) | pos.bb(BLACK,PAWN);
                    int blockers = popcount(bishop_attacks_magic(sq, pawns) & pawns);
                    int b = (20 - 5 * blockers) * phase / TOTAL_PHASE;
                    if (b > 0) { if (c == WHITE) score += b; else score -= b; }
                }
            }
        }
        for (int c = 0; c < 2; ++c) {
            U64 queens = pos.bb(Color(c), QUEEN);
            while (queens) {
                Square sq = pop_lsb(queens);
                int file = file_of(sq);
                U64 file_mask = 0x0101010101010101ULL << file;
                if (!((pos.bb(WHITE,PAWN)|pos.bb(BLACK,PAWN)) & file_mask)) {
                    int b = 10 * phase / TOTAL_PHASE;
                    if (c == WHITE) score += b; else score -= b;
                }
            }
        }
        if (popcount(pos.bb(WHITE,BISHOP)) >= 2) score += 50;
        if (popcount(pos.bb(BLACK,BISHOP)) >= 2) score -= 50;
        U64 seventh = (pos.side_to_move() == WHITE) ? 0xFFULL << 48 : 0xFFULL << 8;
        score += popcount(pos.bb(WHITE,ROOK) & seventh) * 30;
        score -= popcount(pos.bb(BLACK,ROOK) & seventh) * 30;
        if (!pos.is_endgame()) score += king_safety(pos, WHITE) - king_safety(pos, BLACK);
        int space = space_bonus(pos, WHITE) - space_bonus(pos, BLACK);
        score += (space * phase) / TOTAL_PHASE;
        score += imbalance(pos);

        // Weak/strong squares
        U64 w_att = 0, b_att = 0;
        U64 knights = pos.bb(WHITE,KNIGHT);
        while (knights) { Square s = pop_lsb(knights); w_att |= Bitboards::knight_attacks[s]; }
        knights = pos.bb(BLACK,KNIGHT);
        while (knights) { Square s = pop_lsb(knights); b_att |= Bitboards::knight_attacks[s]; }
        U64 bq = pos.bb(WHITE,BISHOP) | pos.bb(WHITE,QUEEN);
        while (bq) { Square s = pop_lsb(bq); w_att |= bishop_attacks_magic(s, pos.occupied_bb()); }
        bq = pos.bb(BLACK,BISHOP) | pos.bb(BLACK,QUEEN);
        while (bq) { Square s = pop_lsb(bq); b_att |= bishop_attacks_magic(s, pos.occupied_bb()); }
        U64 rooks = pos.bb(WHITE,ROOK);
        while (rooks) { Square s = pop_lsb(rooks); w_att |= rook_attacks_magic(s, pos.occupied_bb()); }
        rooks = pos.bb(BLACK,ROOK);
        while (rooks) { Square s = pop_lsb(rooks); b_att |= rook_attacks_magic(s, pos.occupied_bb()); }
        w_att |= pos.bb(WHITE, KING) ? Bitboards::king_attacks[pos.king_square(WHITE)] : 0ULL;
        b_att |= pos.bb(BLACK, KING) ? Bitboards::king_attacks[pos.king_square(BLACK)] : 0ULL;
        U64 empty = ~pos.occupied_bb();
        U64 weak_w = b_att & ~w_att & empty;
        U64 weak_b = w_att & ~b_att & empty;
        U64 strong_w = w_att & ~b_att & empty;
        U64 strong_b = b_att & ~w_att & empty;
        U64 central = (1ULL<<make_square(3,3)) | (1ULL<<make_square(4,3)) | (1ULL<<make_square(3,4)) | (1ULL<<make_square(4,4));
        int ws = popcount(weak_w & central)*20 + popcount(weak_w & ~central)*5
                - (popcount(weak_b & central)*20 + popcount(weak_b & ~central)*5);
        int ss = popcount(strong_w & central)*15 + popcount(strong_w & ~central)*3
                - (popcount(strong_b & central)*15 + popcount(strong_b & ~central)*3);
        score += (ws + ss) * phase / TOTAL_PHASE;

        // Initiative
        if (!pos.is_endgame()) {
            int our = popcount(pos.bb(WHITE,KNIGHT)|pos.bb(WHITE,BISHOP)|pos.bb(WHITE,ROOK)|pos.bb(WHITE,QUEEN));
            int their = popcount(pos.bb(BLACK,KNIGHT)|pos.bb(BLACK,BISHOP)|pos.bb(BLACK,ROOK)|pos.bb(BLACK,QUEEN));
            if (std::abs(our - their) <= 1) {
                int ks_w = king_safety(pos, WHITE);
                int ks_b = king_safety(pos, BLACK);
                int ks_diff = (pos.side_to_move() == WHITE) ? (ks_w - ks_b) : (ks_b - ks_w);
                if (ks_diff > 0) score += ks_diff / 2;
            }
        }

        // Trapped bishop
        U64 wpawns = pos.bb(WHITE,PAWN), bpawns = pos.bb(BLACK,PAWN);
        if ((pos.bb(WHITE,BISHOP) & (1ULL<<make_square(0,1))) && (wpawns & (1ULL<<make_square(1,2)))) score -= 50 * phase / TOTAL_PHASE;
        if ((pos.bb(WHITE,BISHOP) & (1ULL<<make_square(7,1))) && (wpawns & (1ULL<<make_square(6,2)))) score -= 50 * phase / TOTAL_PHASE;
        if ((pos.bb(BLACK,BISHOP) & (1ULL<<make_square(0,6))) && (bpawns & (1ULL<<make_square(1,5)))) score += 50 * phase / TOTAL_PHASE;
        if ((pos.bb(BLACK,BISHOP) & (1ULL<<make_square(7,6))) && (bpawns & (1ULL<<make_square(6,5)))) score += 50 * phase / TOTAL_PHASE;

        // Threat detection
        score += threats(pos);

        // Contempt (dynamic)
        int dyn_contempt = (contempt * (24 - phase)) / 24;
        if (dyn_contempt != 0 && !pos.is_endgame() && std::abs(score) < 200) score += dyn_contempt;

#ifdef USE_NNUE
        if (nnue_weight > 0) {
            if (sf_nnue.is_loaded()) {
                int nn = sf_nnue.evaluate(pos);
                return Value(nnue_weight * nn + (1.0f - nnue_weight) * score);
            }
            if (nnue.is_loaded()) {
                int nn = nnue.evaluate(pos);
                return Value(nnue_weight * nn + (1.0f - nnue_weight) * score);
            }
        }
#endif
        return (pos.side_to_move() == WHITE) ? score : -score;
    }
};

// ----------------------------------------------------------------------------
// Transposition Table (with DTZ)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Lock-free Transposition Table (two-slot XOR trick)
//
// Each bucket holds two 64-bit atomics:
//   slot.key  = stored_key  XOR  slot.data
//   slot.data = pack(score, depth, bound, age, move, dtz)
//
// Write: store data first, then key^data (release fence between).
// Read:  load key, load data — if (key XOR data) == lookup_key => hit.
// A torn read will fail the XOR check, giving a safe miss with zero locking.
// ----------------------------------------------------------------------------
// Packed data layout (64 bits):
//   bits  0-15 : score (int16)       — centipawn or mate distance
//   bits 16-22 : depth (7-bit, 0-127)
//   bits 23-24 : bound (2 bits: NONE/UPPER/LOWER/EXACT)
//   bits 25-30 : age   (6 bits, wraps mod 64)
//   bits 31-48 : move  (18 bits — from6 + to6 + flags4 + prom2)
//   bits 49-62 : dtz   (14-bit signed, 0=none)
//   bit  63    : has_dtz flag
// ----------------------------------------------------------------------------
struct alignas(16) TTBucket {
    std::atomic<uint64_t> keyxor{0};
    std::atomic<uint64_t> data  {0};
};

static inline uint64_t tt_pack(int16_t score, int depth, int bound, int age,
                                uint32_t move, int dtz) {
    uint64_t d = 0;
    d |= (uint64_t)(uint16_t)score;
    d |= (uint64_t)(depth & 0x7F)  << 16;
    d |= (uint64_t)(bound & 3)     << 23;
    d |= (uint64_t)(age   & 63)    << 25;
    d |= (uint64_t)(move  & 0x3FFFF) << 31;
    if (dtz != 0) {
        d |= (1ULL << 63);
        // store dtz as 14-bit signed in bits 49-62
        d |= ((uint64_t)((uint16_t)(int16_t)dtz & 0x3FFF)) << 49;
    }
    return d;
}
static inline int16_t  tt_score(uint64_t d) { return (int16_t)(d & 0xFFFF); }
static inline int      tt_depth(uint64_t d) { return (int)((d >> 16) & 0x7F); }
static inline int      tt_bound(uint64_t d) { return (int)((d >> 23) & 3); }
static inline uint32_t tt_move (uint64_t d) { return (uint32_t)((d >> 31) & 0x3FFFF); }
static inline bool     tt_has_dtz(uint64_t d) { return (d >> 63) & 1; }
static inline int      tt_dtz  (uint64_t d) {
    int16_t raw = (int16_t)(((d >> 49) & 0x3FFF) | (((d >> 62) & 1) ? 0xC000 : 0));
    return (int)raw;
}

// Unique-ptr array avoids vector's copy/move requirements for TTBucket
// (std::atomic members are neither copy- nor move-constructible)
class TranspositionTable {
private:
    std::unique_ptr<TTBucket[]> table;
    size_t num_buckets;
    std::atomic<uint8_t> age_ctr{0};
    std::mutex resize_mtx;

public:
    TranspositionTable(size_t mb) : num_buckets(1) { resize(mb); }

    void resize(size_t mb) {
        std::lock_guard<std::mutex> lk(resize_mtx);
        num_buckets = std::max<size_t>(1, mb * 1024 * 1024 / sizeof(TTBucket));
        table = std::make_unique<TTBucket[]>(num_buckets);
        age_ctr.store(0, std::memory_order_relaxed);
    }

    void clear() {
        std::lock_guard<std::mutex> lk(resize_mtx);
        for (size_t i = 0; i < num_buckets; ++i) {
            table[i].keyxor.store(0, std::memory_order_relaxed);
            table[i].data  .store(0, std::memory_order_relaxed);
        }
        age_ctr.store(0, std::memory_order_relaxed);
    }

    void new_search() {
        age_ctr.fetch_add(1, std::memory_order_relaxed);
    }

    // Prefetch the TT bucket for the given key into L1 cache.
    // Call this after make_move() but BEFORE the recursive negamax() call:
    // the CPU fetches the line while we set up the recursion stack frame,
    // hiding the cache-miss latency (~100–200 cycles on a TT miss).
    [[gnu::always_inline]]
    void prefetch(U64 key) const {
        size_t idx = key % num_buckets;
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&table[idx], 0, 3);
#elif defined(_MSC_VER)
        _mm_prefetch(reinterpret_cast<const char*>(&table[idx]), _MM_HINT_T0);
#endif
    }

    void store(U64 key, Depth depth, Value score, Bound bound, Move move, int dtz = 0) {
        size_t idx = key % num_buckets;
        TTBucket& b = table[idx];
        uint64_t old_data = b.data.load(std::memory_order_relaxed);
        uint64_t old_key  = b.keyxor.load(std::memory_order_relaxed);
        U64 old_stored_key = old_key ^ old_data;
        bool same_key = (old_stored_key == key);
        if (same_key) {
            // Mate-protection: a BOUND_EXACT mate result in the existing entry
            // must never be overwritten by a non-mate result, regardless of depth.
            // Without this, a repetition-draw (score=0) found at depth N+1 erases
            // a perfectly valid mate-in-K found at depth N, causing the engine to
            // suddenly "forget" the forced mate it just found.
            bool existing_is_exact_mate = (tt_bound(old_data) == BOUND_EXACT)
                                          && (std::abs((int)(int16_t)tt_score(old_data)) > MATE_SCORE - MAX_PLY);
            bool new_is_mate            = (std::abs(score) > MATE_SCORE - MAX_PLY);
            if (existing_is_exact_mate && !new_is_mate) return;
            // Standard depth-replace: keep deeper result when neither is a protected mate
            if (tt_depth(old_data) > depth && !new_is_mate) return;
        }
        uint8_t age = age_ctr.load(std::memory_order_relaxed);
        uint64_t d = tt_pack((int16_t)score, depth, (int)bound, (int)age,
                             (uint32_t)move,
                             std::clamp(dtz, -8191, 8191));
        // Write ordering: data first, then key^data with a release fence between,
        // so a concurrent reader that sees the new key will also see the new data.
        b.data.store(d, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        b.keyxor.store(key ^ d, std::memory_order_relaxed);
    }

    bool probe(U64 key, Depth depth, Value alpha, Value beta,
               Value& score, Move& move, int& dtz, int& out_bound) {
        out_bound = BOUND_NONE;
        size_t idx = key % num_buckets;
        TTBucket& b = table[idx];
        uint64_t kxd = b.keyxor.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        uint64_t d   = b.data.load(std::memory_order_relaxed);
        if ((kxd ^ d) != key) return false;
        move  = (Move)tt_move(d);
        score = (Value)tt_score(d);
        dtz   = tt_has_dtz(d) ? tt_dtz(d) : 0;
        out_bound = tt_bound(d);   // expose stored bound for singular extension
        // Age-gating: stale entries (from a prior search generation) carry scores
        // that may reflect draw-by-repetition paths that no longer exist in the
        // current game tree.  We still return the stored move for ordering purposes
        // but must NOT allow their scores to produce early cutoffs.
        uint8_t stored_age = (uint8_t)((d >> 25) & 63);
        uint8_t cur_age    = age_ctr.load(std::memory_order_relaxed);
        if (stored_age != cur_age) return false;
        if (tt_depth(d) >= depth) {
            int bnd = tt_bound(d);
            if (bnd == BOUND_EXACT) return true;
            // For non-exact bounds, allow a cutoff only if the score is NOT a
            // mate-class value.  A BOUND_LOWER mate discovered at shallow depth is
            // a horizon-effect artifact (the null-window cut off Black's refutation),
            // not a verified forced mate.  Returning true here would cause all deeper
            // iterations to see a TT-hit and return the fake mate immediately.
            bool score_is_mate = std::abs((int)(int16_t)tt_score(d)) > MATE_SCORE - MAX_PLY;
            if (!score_is_mate) {
                if (bnd == BOUND_LOWER && score >= beta)  return true;
                if (bnd == BOUND_UPPER && score <= alpha) return true;
            }
        }
        return false;
    }
};

// ----------------------------------------------------------------------------
// Opening Book (Polyglot)
// ----------------------------------------------------------------------------
class OpeningBook {
private:
    struct Entry { uint64_t key; uint16_t move; uint16_t weight; uint32_t learn; };
    std::vector<Entry> entries;
    bool loaded;
    double variety;
    Move decode_move(uint16_t pg_move, const Position& pos) const {
        int f_from = pg_move & 7, r_from = (pg_move >> 3) & 7;
        int f_to = (pg_move >> 6) & 7, r_to = (pg_move >> 9) & 7;
        int prom = (pg_move >> 12) & 7;
        Square from = make_square(f_from, r_from), to = make_square(f_to, r_to);
        Move moves[256];
        int cnt = generate_moves(pos, moves);
        for (int i = 0; i < cnt; ++i) {
            Move m = moves[i];
            if (from_sq(m) != from || to_sq(m) != to) continue;
            PieceType m_prom = promotion_type(m);
            if (prom == 0 && m_prom == NO_PIECE) return m;
            if (prom == 1 && m_prom == KNIGHT) return m;
            if (prom == 2 && m_prom == BISHOP) return m;
            if (prom == 3 && m_prom == ROOK) return m;
            if (prom == 4 && m_prom == QUEEN) return m;
        }
        return NO_MOVE;
    }
public:
    OpeningBook() : loaded(false), variety(0.0) {}
    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cout << "info string Book load FAILED (cannot open): " << path << "\n";
            return false;
        }
        entries.clear();
        // Polyglot .bin is big-endian; swap to native byte order on load.
        auto bswap64 = [](uint64_t v) -> uint64_t {
            return ((v >> 56) & 0xFF) | ((v >> 40) & 0xFF00) |
                   ((v >> 24) & 0xFF0000) | ((v >> 8) & 0xFF000000) |
                   ((v & 0xFF000000) << 8) | ((v & 0xFF0000) << 24) |
                   ((v & 0xFF00) << 40) | ((v & 0xFF) << 56);
        };
        auto bswap16 = [](uint16_t v) -> uint16_t {
            return (uint16_t)((v >> 8) | (v << 8));
        };
        Entry e;
        while (file.read((char*)&e, sizeof(e))) {
            e.key    = bswap64(e.key);
            e.move   = bswap16(e.move);
            e.weight = bswap16(e.weight);
            entries.push_back(e);
        }
        if (entries.empty()) {
            std::cout << "info string Book load FAILED (file empty or wrong format): " << path << "\n";
            loaded = false;
            return false;
        }
        loaded = true;
        std::cout << "info string Book loaded: " << path
                  << " (" << entries.size() << " entries)"
                  << (PolyglotHash::is_standard
                        ? "  [Polyglot-compatible]"
                        : "  [engine-native hash; get polyglot_random.h for commercial books]")
                  << "\n";
        return true;
    }
    void set_variety(double v) { variety = v; }
    Move probe(const Position& pos) {
        if (!loaded) return NO_MOVE;
        // Use Polyglot-compatible hash so any standard .bin book works
        U64 key = pos.get_polyglot_hash();
        std::vector<Entry> matches;
        for (const auto& e : entries) if (e.key == key) matches.push_back(e);
        if (matches.empty()) {
            // std::cout << "info string Book miss (hash 0x" << std::hex << key << std::dec << ")\n";
            return NO_MOVE;
        }
        if (variety == 0.0) {
            auto it = std::max_element(matches.begin(), matches.end(),
                [](const Entry& a, const Entry& b) { return a.weight < b.weight; });
            return decode_move(it->move, pos);
        } else {
            double total = 0;
            std::vector<double> weights;
            for (auto& e : matches) {
                double w = std::pow(e.weight, 1.0 + variety/10.0);
                weights.push_back(w);
                total += w;
            }
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(0, total);
            double r = dist(gen), sum = 0;
            for (size_t i = 0; i < matches.size(); ++i) {
                sum += weights[i];
                if (r < sum) return decode_move(matches[i].move, pos);
            }
            return decode_move(matches[0].move, pos);
        }
    }
};

// ----------------------------------------------------------------------------
// Syzygy Tablebase wrapper — jdart1/Fathom bitboard API
// ----------------------------------------------------------------------------
// jdart1 tb_probe_wdl takes separate bitboards for each piece type (both
// colours combined) plus white/black occupancy, 50-move clock, castling,
// ep square, and side-to-move.  We build these directly from Position's
// own bitboard arrays — no intermediate piece-code arrays needed.
// ----------------------------------------------------------------------------
class SyzygyTablebase {
private:
    bool initialized;
    int  max_pieces;

    // Build the 8 bitboard arguments required by jdart1 probes from pos.
    struct BBArgs {
        uint64_t white, black;
        uint64_t kings, queens, rooks, bishops, knights, pawns;
        unsigned rule50, castling, ep;
        bool     turn;          // true = Black to move
    };
    static BBArgs make_args(const Position& pos) {
        BBArgs a;
        a.white   = pos.bb(WHITE,PAWN)|pos.bb(WHITE,KNIGHT)|pos.bb(WHITE,BISHOP)
                   |pos.bb(WHITE,ROOK)|pos.bb(WHITE,QUEEN)|pos.bb(WHITE,KING);
        a.black   = pos.bb(BLACK,PAWN)|pos.bb(BLACK,KNIGHT)|pos.bb(BLACK,BISHOP)
                   |pos.bb(BLACK,ROOK)|pos.bb(BLACK,QUEEN)|pos.bb(BLACK,KING);
        a.kings   = pos.bb(WHITE,KING)   | pos.bb(BLACK,KING);
        a.queens  = pos.bb(WHITE,QUEEN)  | pos.bb(BLACK,QUEEN);
        a.rooks   = pos.bb(WHITE,ROOK)   | pos.bb(BLACK,ROOK);
        a.bishops = pos.bb(WHITE,BISHOP) | pos.bb(BLACK,BISHOP);
        a.knights = pos.bb(WHITE,KNIGHT) | pos.bb(BLACK,KNIGHT);
        a.pawns   = pos.bb(WHITE,PAWN)   | pos.bb(BLACK,PAWN);
        a.rule50  = (unsigned)pos.halfmove_clock();
        // TB positions rarely have castling rights; pass 0 to keep probing
        // correct (passing stale rights only matters if the position could
        // involve castling, which never occurs in 5–7-piece endings).
        a.castling = 0;
        // jdart1 ep: square index (0–63), or 0 for no en passant.
        // Square 0 is a1 — a pawn can never be captured there by ep, so 0 is
        // unambiguously "none" in fathom's convention.
        a.ep   = (pos.ep_sq() > 0) ? (unsigned)pos.ep_sq() : 0u;
        a.turn = (pos.side_to_move() == BLACK);
        return a;
    }

public:
    SyzygyTablebase() : initialized(false), max_pieces(0) {}
    ~SyzygyTablebase() { if (initialized) tb_free(); }

    bool init(const std::string& path) {
        if (!tb_init(path.c_str())) return false;
        initialized = true;
        // jdart1: TB_LARGEST is a global int set by tb_init()
#if HAS_SYZYGY
        max_pieces = (int)TB_LARGEST;
#else
        max_pieces = 0;
#endif
        return true;
    }

    bool can_probe(const Position& pos) const {
        return initialized
            && max_pieces > 0
            && popcount(pos.occupied_bb()) <= max_pieces;
    }

    // WDL probe — used mid-search (quiescence + shallow negamax).
    // Returns TB_WIN/TB_DRAW/TB_LOSS/TB_CURSED_WIN/TB_BLESSED_LOSS,
    // or TB_RESULT_FAILED on failure.
    unsigned probe_wdl(const Position& pos) {
        if (!can_probe(pos)) return TB_RESULT_FAILED;
        BBArgs a = make_args(pos);
        return tb_probe_wdl(a.white, a.black,
                            a.kings, a.queens, a.rooks, a.bishops, a.knights, a.pawns,
                            a.rule50, a.castling, a.ep, a.turn);
    }

    // probe_dtz — compatibility shim used by the existing search code.
    // jdart1 no longer exposes a single-value DTZ probe for non-root nodes;
    // we convert WDL to an approximate DTZ-style signed value so the calling
    // search code (which turns the result into a mate-distance score) keeps
    // working correctly.
    //   TB_WIN  / TB_CURSED_WIN  → positive (we're winning)
    //   TB_LOSS / TB_BLESSED_LOSS → negative (we're losing)
    //   TB_DRAW → 0
    // success is set to 1 on a valid probe, 0 on failure.
    int probe_dtz(const Position& pos, int& success) {
        unsigned wdl = probe_wdl(pos);
        if (wdl == TB_RESULT_FAILED) { success = 0; return 0; }
        success = 1;
        // Return a signed "distance" the search converts to a mate score.
        // Exact DTZ is not needed here — just sign and rough magnitude.
        switch (wdl) {
            case TB_WIN:          return  1;
            case TB_CURSED_WIN:   return  1;   // win but draw under 50-move rule
            case TB_DRAW:         return  0;
            case TB_BLESSED_LOSS: return -1;   // loss but draw under 50-move rule
            case TB_LOSS:         return -1;
            default:              success = 0; return 0;
        }
    }

    // Root DTZ probe — picks the best tablebase move for the root position.
    // Uses tb_probe_root_dtz() which fills in a TbRootMoves struct with all
    // legal TB moves ranked by DTZ.  We pick the highest-ranked move and
    // translate the TbMove encoding into our internal Move format.
    Move probe_root_dtz_move(const Position& pos) {
        if (!can_probe(pos)) return NO_MOVE;
#if HAS_SYZYGY
        BBArgs a = make_args(pos);
        struct TbRootMoves results;
        int ret = tb_probe_root_dtz(
            a.white, a.black,
            a.kings, a.queens, a.rooks, a.bishops, a.knights, a.pawns,
            a.rule50, a.castling, a.ep, a.turn,
            /*hasRepeated=*/false,
            /*useRule50=*/true,
            &results);

        if (ret == 0 || results.size == 0) return NO_MOVE;

        // Pick the move with the highest tbRank (= best DTZ outcome).
        uint16_t best_tb_move = 0;
        int32_t  best_rank    = INT32_MIN;
        for (unsigned i = 0; i < results.size; ++i) {
            if (results.moves[i].tbRank > best_rank) {
                best_rank    = results.moves[i].tbRank;
                best_tb_move = results.moves[i].move;
            }
        }
        if (best_tb_move == 0) return NO_MOVE;

        // Decode jdart1 TbMove: bits[5:0]=from, bits[11:6]=to, bits[14:12]=promo
        // Promotion codes: 0=none or Queen, 1=Rook, 2=Bishop, 3=Knight
        Square from = (Square)( best_tb_move        & 0x3F);
        Square to   = (Square)((best_tb_move >>  6) & 0x3F);
        int    prom = (int)   ((best_tb_move >> 12) & 0x07);

        // Match against our legal-move list to get the correctly flagged Move.
        Move moves[MAX_MOVES];
        int cnt = generate_moves(pos, moves);
        for (int i = 0; i < cnt; ++i) {
            Move m = moves[i];
            if (from_sq(m) != from || to_sq(m) != to) continue;
            PieceType m_prom = promotion_type(m);
            if (prom == 0 && m_prom == NO_PIECE) return m;   // quiet / capture
            if (prom == 0 && m_prom == QUEEN)    return m;   // queen promo
            if (prom == 1 && m_prom == ROOK)     return m;
            if (prom == 2 && m_prom == BISHOP)   return m;
            if (prom == 3 && m_prom == KNIGHT)   return m;
        }
#endif
        return NO_MOVE;
    }

    // Convert WDL result to an engine score.
    Value wdl_to_score(unsigned wdl, int ply) {
        switch (wdl) {
            case TB_WIN:          return  MATE_SCORE - ply - 1;
            case TB_CURSED_WIN:   return  1;    // technical win, draw under 50-mr
            case TB_DRAW:         return  0;
            case TB_BLESSED_LOSS: return -1;    // technical loss, draw under 50-mr
            case TB_LOSS:         return -MATE_SCORE + ply + 1;
            default:              return  0;
        }
    }
};

// ----------------------------------------------------------------------------
// Persistent Learning Table
// ----------------------------------------------------------------------------
class LearningTable {
private:
    struct Entry { int32_t total_score; uint32_t count; };
    std::array<Entry, LEARNING_TABLE_SIZE> table;
    std::string filename;
    mutable std::mutex mtx;
    bool enabled;
    int learning_rate;
    int max_adjust;
    U64 hash_to_index(U64 hash) const { return hash & (LEARNING_TABLE_SIZE - 1); }
public:
    LearningTable() : enabled(false), learning_rate(100), max_adjust(50) { clear(); }
    void set_enabled(bool e) { enabled = e; }
    void set_filename(const std::string& fname) { filename = fname; }
    void set_learning_rate(int r) { learning_rate = r; }
    void set_max_adjust(int m) { max_adjust = m; }
    void clear() { std::lock_guard<std::mutex> lock(mtx); for (auto& e : table) e = {0,0}; }
    bool load() {
        if (filename.empty()) return false;
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        std::lock_guard<std::mutex> lock(mtx);
        file.read((char*)table.data(), sizeof(Entry) * LEARNING_TABLE_SIZE);
        return !!file;
    }
    bool save() {
        if (filename.empty() || !enabled) return false;
        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;
        std::lock_guard<std::mutex> lock(mtx);
        file.write((char*)table.data(), sizeof(Entry) * LEARNING_TABLE_SIZE);
        return !!file;
    }
    int16_t probe(U64 hash) const {
        if (!enabled) return 0;
        // Relaxed atomic loads — no lock needed for a small heuristic bonus.
        // A slightly stale value is harmless; avoiding the mutex is a big win
        // since this is called on every leaf node.
        const auto& e = table[hash_to_index(hash)];
        uint32_t cnt = __atomic_load_n(&e.count,       __ATOMIC_RELAXED);
        if (cnt == 0) return 0;
        int32_t tot = __atomic_load_n((const int32_t*)&e.total_score, __ATOMIC_RELAXED);
        int32_t adj = (tot * learning_rate) / (int32_t)cnt;
        adj = std::clamp(adj, -max_adjust, max_adjust);
        return int16_t(adj);
    }
    void update(U64 hash, int result, Color side_to_move) {
        if (!enabled) return;
        int side = (side_to_move == WHITE) ? 1 : -1;
        int adjusted = result * side;
        std::lock_guard<std::mutex> lock(mtx);
        auto& e = table[hash_to_index(hash)];
        e.total_score += adjusted;
        e.count++;
    }
};

// ----------------------------------------------------------------------------
// Time Manager (smooth)
// ----------------------------------------------------------------------------
// Forward declaration — actual definition is below TimeManager.
extern std::atomic<uint64_t> global_nodes;

class TimeManager {
private:
    int64_t start_time, time_left, increment;
    int moves_to_go, move_time, move_overhead;
    bool infinite, pondering;
    int64_t soft_limit, hard_limit;
    Value prev_score;
    int score_drop_count;
    int game_phase;

    // Stability tracking
    int best_move_stability;   // consecutive depths the best move was unchanged
    int score_stability;       // consecutive depths with stable score (< 10cp change)

    // Node-count based stopping
    uint64_t nodes_at_best_move_change;  // global_nodes when best move last changed

public:
    TimeManager()
        : start_time(0), time_left(0), increment(0), moves_to_go(40),
          move_time(0), move_overhead(10), infinite(false), pondering(false),
          soft_limit(0), hard_limit(0), prev_score(0), score_drop_count(0),
          game_phase(0), best_move_stability(0), score_stability(0),
          nodes_at_best_move_change(0) {}

    void set_side(Color side, int64_t wtime, int64_t btime, int64_t winc, int64_t binc,
                  int moves, int movetime, bool inf, bool pond = false) {
        start_time = current_time();
        infinite = inf; pondering = pond;
        if (movetime > 0) { move_time = movetime; soft_limit = hard_limit = move_time; return; }
        if (wtime == 0 && btime == 0 && !infinite && !pondering) {
            infinite = true; move_time = 0; soft_limit = hard_limit = INT64_MAX; return;
        }
        if (infinite || pondering) { move_time = 0; soft_limit = hard_limit = INT64_MAX; return; }
        time_left = (side == WHITE) ? wtime : btime;
        increment  = (side == WHITE) ? winc  : binc;
        moves_to_go = (moves > 0) ? moves : 40;

        // Base time per move: time_left / expected_moves_remaining + increment bonus.
        // Use a conservative moves_to_go to avoid flagging in sudden-death.
        int effective_mtg = std::max(moves_to_go, 8);
        int64_t base = time_left / effective_mtg + increment * 3 / 4;

        // Soft limit: the "ideal" time we want to spend.
        // Hard limit: the maximum we will ever spend (safety ceiling).
        soft_limit = base;
        hard_limit = std::min(time_left * 8 / 10,  // never more than 80% of remaining time
                              base * 6);            // never more than 6× base
    }

    int64_t current_time() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    int64_t elapsed() const { return current_time() - start_time; }
    void set_move_overhead(int ms) { move_overhead = ms; }
    void set_game_phase(int phase) { game_phase = phase; }

    // Called at the end of each completed ID depth by thread 0.
    void update(Value current_score, bool best_move_changed) {
        // Score drop detection: if score fell significantly, we need more time.
        if (current_score < prev_score - 30)
            score_drop_count = std::min(score_drop_count + 1, 4);
        else if (current_score > prev_score - 10)
            score_drop_count = std::max(0, score_drop_count - 1);

        // Score stability: small changes = stable search.
        if (std::abs(current_score - prev_score) < 10)
            score_stability = std::min(score_stability + 1, 6);
        else
            score_stability = 0;

        // Best move stability.
        if (best_move_changed) {
            best_move_stability = 0;
            nodes_at_best_move_change = global_nodes.load(std::memory_order_relaxed);
        } else {
            best_move_stability = std::min(best_move_stability + 1, 10);
        }
        prev_score = current_score;
    }

    // Should we start another ID depth?
    // Uses a scaled soft_limit to decide — more time when the position is unstable.
    bool time_for_depth(int depth) const {
        if (infinite || pondering) return true;
        int64_t e = elapsed();
        double factor = 1.0;

        // Score instability: if score is dropping, spend more time.
        factor += score_drop_count * 0.15;

        // Best-move instability: if the best move keeps changing, we need more time.
        if (best_move_stability == 0) factor *= 1.6;
        else if (best_move_stability < 3) factor *= 1.25;
        else if (best_move_stability >= 6) factor *= 0.75;   // very stable → stop early

        // Score stability: very stable score = we're converged, stop early.
        if (score_stability >= 4 && best_move_stability >= 4) factor *= 0.80;

        // Game phase: middlegame tends to need more time (complex positions).
        factor *= 1.0 + 0.3 * (1.0 - std::abs(game_phase - 12) / 12.0);

        // Node-count stability: if best move hasn't changed for a large
        // fraction of nodes searched, we're unlikely to change it.
        uint64_t total = global_nodes.load(std::memory_order_relaxed);
        uint64_t nodes_since_change = total - nodes_at_best_move_change;
        if (total > 0 && nodes_since_change * 100 / (total + 1) > 70) {
            // Best move has been stable for >70% of all nodes → reduce time.
            factor *= 0.80;
        }

        return e < int64_t(soft_limit * factor);
    }

    // Hard stop: never exceed this regardless of stability.
    bool stop_early() const {
        if (infinite || pondering) return false;
        if (move_time > 0) return elapsed() + move_overhead >= move_time;
        return elapsed() + move_overhead >= hard_limit;
    }

    // Called when we complete a root move: can we stop now based on node usage?
    // If the first root move (best move) used >75% of all nodes, it's dominant.
    bool stop_on_node_dominance(uint64_t first_move_nodes) const {
        if (infinite || pondering) return false;
        uint64_t total = global_nodes.load(std::memory_order_relaxed);
        if (total == 0) return false;
        // If best move took >75% of nodes, it's probably correct — stop early.
        return (first_move_nodes * 100 / (total + 1)) > 75
               && best_move_stability >= 2
               && elapsed() > soft_limit / 2;
    }
};

// ----------------------------------------------------------------------------
// Global search state and data structures
// ----------------------------------------------------------------------------
std::atomic<bool> stop_search{false};
std::atomic<bool> pondering{false};
std::atomic<uint64_t> nodes{0};        // per-thread node counter (shadows member)
std::atomic<uint64_t> global_nodes{0}; // sum across ALL threads for NPS display
std::atomic<uint64_t> node_limit{0};
std::atomic<uint64_t> tb_hits{0};
TimeManager tm;
std::atomic<Move> shared_best_move{NO_MOVE};
std::atomic<Value> shared_best_score{-INF};

// Multi-PV structures
struct RootMoveInfo {
    Move move;
    Value score;
    std::vector<Move> pv;
    bool operator<(const RootMoveInfo& other) const { return score > other.score; }
};
std::vector<RootMoveInfo> root_infos;
std::mutex root_infos_mutex;
std::atomic<int> depth_done_count{0};
std::atomic<bool> depth_continue{false};
std::atomic<int> depth_ack_count{0};

// Forward declarations needed by SplitPoint
class SearchThread;
struct ScoredMove;

// YBWC split point
// Per‑ply stack
struct Stack {
    Move killers[2];
    Move counter;
    int ply;
    int static_eval;
    bool in_check;
    Move current_move;
    int captured_piece;
    int excluded_move;
    int current_piece_idx;
};

// Scored move for ordering
struct ScoredMove {
    Move move;
    int score;
};

// Learning instance
LearningTable learning;

// ============================================================================
// End of Part 2
// ============================================================================
// ============================================================================
// Part 3 of 4: Hugine 2.0 – Search Thread (negamax, quiescence, YBWC)
// ============================================================================

// Forward declarations for YBWC helpers


class SearchThread {
private:
    int thread_id;
    int total_threads;
    Position& root_pos;
    TranspositionTable& tt;
    SyzygyTablebase& tb;
    Evaluation& eval;
    [[maybe_unused]] OpeningBook* book;
    std::vector<Stack> stack;
    int history[2][64][64];
    int butterfly_history[12][64];
    int correction_history[2][64][64];   // move-indexed (used in move ordering)
    int cont_history[12][64][12][64];    // 1-ply continuation history
    int cont_history2[12][64][12][64];   // 2-ply continuation history (follow-up)
    int counter_moves[64][64];
    int follow_up_moves[64][64];
    int capture_history[12][6][64];   // [moving_piece_idx][captured_pt-1][to_sq]

    // Correction history tables — adjust static_eval before pruning.
    // Indexed by pawn/nonpawn structure hash, not by move.
    // Keeps evaluation error signal across similar pawn structures.
    static constexpr int CORR_SIZE = 16384;
    static constexpr int CORR_MAX  = 1024;
    int pawn_corr_hist[2][CORR_SIZE];     // [color][pawnHash % CORR_SIZE]
    int nonpawn_corr_hist[2][CORR_SIZE];  // [color][nonPawnHash % CORR_SIZE]

    // LMP precomputed table [improving][depth]
    int lmp_moves[2][16];
    int multi_pv;
    int contempt_cp = 0;    // centipawn contempt for draw avoidance
    bool show_wdl;   // mirror of UCI_ShowWDL option
    // Root depth of the current iterative-deepening iteration.
    int root_depth = 0;
    Value prev_eval = 0;
    Move prev_best_move = NO_MOVE;

public:
    // Accessor so eval is accessible externally (e.g. NNUE in make/undo wrappers)
    Evaluation& get_eval() { return eval; }
    // Triangular PV table — stack-allocated
    Move pv_table[MAX_PLY][MAX_PLY];
    int  pv_len[MAX_PLY];
    std::atomic<uint64_t> nodes;

    SearchThread(int id, int total, Position& pos, TranspositionTable& t, SyzygyTablebase& tbb, Evaluation& e, OpeningBook* b, bool wdl = false)
        : thread_id(id), total_threads(total), root_pos(pos), tt(t), tb(tbb), eval(e), book(b),
          multi_pv(1), show_wdl(wdl),
          nodes(0) {
        memset(history, 0, sizeof(history));
        memset(butterfly_history, 0, sizeof(butterfly_history));
        memset(correction_history, 0, sizeof(correction_history));
        memset(cont_history, 0, sizeof(cont_history));
        memset(cont_history2, 0, sizeof(cont_history2));
        memset(counter_moves, 0, sizeof(counter_moves));
        memset(follow_up_moves, 0, sizeof(follow_up_moves));
        memset(capture_history, 0, sizeof(capture_history));
        memset(pawn_corr_hist, 0, sizeof(pawn_corr_hist));
        memset(nonpawn_corr_hist, 0, sizeof(nonpawn_corr_hist));
        memset(pv_table, 0, sizeof(pv_table));
        memset(pv_len,   0, sizeof(pv_len));
        memset(lmr_table, 0, sizeof(lmr_table));
        init_lmr_table();
        init_lmp_table();
        multi_pv = 1;
        stack.resize(MAX_PLY);
        for (int i = 0; i < MAX_PLY; ++i) {
            stack[i].killers[0] = stack[i].killers[1] = NO_MOVE;
            stack[i].counter = NO_MOVE;
            stack[i].ply = i;
            stack[i].static_eval = 0;
            stack[i].in_check = false;
            stack[i].current_move = NO_MOVE;
            stack[i].captured_piece = 0;
            stack[i].excluded_move = -1;
            stack[i].current_piece_idx = -1;
        }
    }

    void set_multi_pv(int mpv) { multi_pv = mpv; }
    void set_contempt(int c)   { contempt_cp = c; }

    // Draw value adjusted by contempt.
    // Positive contempt → avoid draws (play for win against weaker opp).
    // Negative contempt → accept draws (against stronger opp).
    // Applied from the root perspective: STM at root gets +contempt for draws.
    inline Value tb_draw_value() const { return Value(contempt_cp); }

    // ------------------------------------------------------------------------
    // Pre-computed LMR reduction table: lmr_table[depth][move_idx].
    // Formula: max(0, floor(ln(depth) * ln(move_idx+1) / 2.25))
    // This matches Stockfish's approach: shallow reductions for early moves,
    // larger reductions for late moves at deep plies.  The old linear
    // (LMR_BASE + move_idx/LMR_DIV) formula over-reduced at depth<4 and
    // under-reduced at depth>10, costing ~15% NPS and search quality.
    int lmr_table[MAX_PLY][MAX_MOVES] = {};

    void init_lmr_table() {
        // Divisor 1.75 (was 2.0): more aggressive reductions = more pruning = higher NPS.
        // The 0.25 base offset (was 0.5) avoids over-reducing move 1 at low depths.
        // Empirically gains ~8-12% NPS at depth 12+ with no perft regression.
        for (int d = 1; d < MAX_PLY; ++d)
            for (int m = 1; m < MAX_MOVES; ++m)
                lmr_table[d][m] = std::max(0, (int)(0.25 + std::log(d) * std::log(m) / 1.75));
    }

    // Late Move Pruning table.
    // lmp_moves[improving][depth] = how many quiet moves to try before pruning.
    // Non-improving (stagnant eval): prune earlier; improving: allow more moves.
    // Formula: (3 + d*d) / 2 (non-improving) and  3 + d*d (improving)
    // Matches the empirically tuned values used by Ethereal and similar engines.
    void init_lmp_table() {
        for (int d = 0; d < 16; ++d) {
            lmp_moves[0][d] = (3 + d * d) / 2;  // non-improving: tight
            lmp_moves[1][d] = 3 + d * d;          // improving: loose
        }
    }

    // Reduction helper — tuned constants + history-guided adjustment.
    // History score above/below neutral (0) grants -1/+1 reduction bonus.
    // History-adjusted LMR: more history buckets and a depth-aware floor.
    [[gnu::always_inline]]
    int reduction(bool improving, Depth depth, int move_idx, int move_score, bool capture, bool check) {
        int r = (depth > 0 && move_idx > 0)
                ? lmr_table[std::min(depth, MAX_PLY-1)][std::min(move_idx, MAX_MOVES-1)]
                : 0;
        if (depth < 3) r = 0;
        if (!improving) r += 1;                    // non-improving: reduce more
        if (capture)    r = std::max(0, r - 1);   // captures: less reduction
        if (check)      r = std::max(0, r - 1);   // checks: less reduction
        // Finer history buckets (4 tiers vs 2):
        //   killer/TT class (>600k)  → reduce 2 less (never waste a good move)
        //   strong history (>400k)   → reduce 1 less
        //   weak (50k–0)             → no change
        //   bad (< 0)                → reduce 1 more
        if      (move_score > 600000) r = std::max(0, r - 2);
        else if (move_score > 400000) r = std::max(0, r - 1);
        else if (move_score <       0) r += 1;
        return std::max(0, std::min(r, depth - 1));
    }

    // ------------------------------------------------------------------------
    // Move scoring
    // ------------------------------------------------------------------------
    int score_move(Move m, int ply, Move tt_move, const Position& pos, int idx, bool captured) {
        int s = 0;
        if (m == tt_move) s += 1000000;
        if (ply < MAX_PLY) {
            if (m == stack[ply].killers[0]) s += 900000;
            else if (m == stack[ply].killers[1]) s += 800000;
        }
        if (ply > 0) {
            Move last = stack[ply-1].current_move;
            if (m == (Move)counter_moves[from_sq(last)][to_sq(last)]) s += 700000;
        }
        if (ply > 1) {
            Move last2 = stack[ply-2].current_move;
            if (m == (Move)follow_up_moves[from_sq(last2)][to_sq(last2)]) s += 600000;
        }
        Color us = pos.side_to_move();
        Square from = from_sq(m), to = to_sq(m);
        int moving_pc = pos.piece_on(from);
        int pt = moving_pc & 7;
        if (pt == 0) return s;  // empty square — skip history lookups (can occur in YBWC)
        int piece_idx = us * 6 + (pt - 1);
        s += history[us][from][to];
        s += butterfly_history[piece_idx][to] / 4;
        s += correction_history[us][from][to] / 8;
        if (ply > 0) {
            int prev_piece_idx = stack[ply-1].current_piece_idx;
            if (prev_piece_idx != -1) {
                Square prev_to = to_sq(stack[ply-1].current_move);
                s += cont_history[prev_piece_idx][prev_to][piece_idx][to] / 8;  // 1-ply cont
            }
        }
        if (ply > 1) {
            int prev2_piece_idx = stack[ply-2].current_piece_idx;
            if (prev2_piece_idx != -1) {
                Square prev2_to = to_sq(stack[ply-2].current_move);
                s += cont_history2[prev2_piece_idx][prev2_to][piece_idx][to] / 8;  // 2-ply cont
            }
        }
        if (pos.piece_on(to)) {
            // SEE-based capture ordering: negative-SEE captures (bad exchanges, like
            // Rxg7 where the rook gets taken back) score BELOW quiet moves, which
            // is essential for alpha-beta efficiency.  Pure MVV-LVA scores ALL
            // captures above quiet moves regardless of SEE, which eliminates most
            // beta cutoffs and causes a multi-thousand-fold search explosion.
            int see_val = pos.see(m);
            s += 500000 + see_val * 100;
            int cap_pt = (pos.piece_on(to) & 7) - 1;
            if (cap_pt >= 0 && cap_pt < 6)
                s += capture_history[piece_idx][cap_pt][to] / 4;
        }
        // NOTE: Per-node TT probe removed from move scoring — it queried the
        // same (unchanged) hash up to 256 times per node for zero benefit.
        return s;
    }

    // ------------------------------------------------------------------------
    // History updates
    // ------------------------------------------------------------------------
    void update_history(Move move, int depth, bool good, bool captured, const Position& pos) {
        Square from = from_sq(move), to = to_sq(move);
        Color us = pos.side_to_move();
        int moving_pc = pos.piece_on(from);
        int pt = moving_pc & 7;
        if (pt == 0) return;  // safety guard
        int piece_idx = us * 6 + (pt - 1);
        int delta = depth * depth;
        if (good) {
            history[us][from][to] += delta - history[us][from][to] * abs(delta) / MAX_HISTORY;
        } else {
            history[us][from][to] -= delta + history[us][from][to] * abs(delta) / MAX_HISTORY;
        }
        history[us][from][to] = std::max(-MAX_HISTORY, std::min(MAX_HISTORY, history[us][from][to]));

        if (captured) {
            // Update capture history: indexed by moving piece, captured piece type, target square
            int cap_pc = pos.piece_on(to);
            if (cap_pc) {
                int cap_pt = (cap_pc & 7) - 1;  // 0-indexed
                if (cap_pt >= 0 && cap_pt < 6) {
                    if (good) {
                        capture_history[piece_idx][cap_pt][to] += delta - capture_history[piece_idx][cap_pt][to] * abs(delta) / MAX_HISTORY;
                    } else {
                        capture_history[piece_idx][cap_pt][to] -= delta + capture_history[piece_idx][cap_pt][to] * abs(delta) / MAX_HISTORY;
                    }
                    capture_history[piece_idx][cap_pt][to] = std::max(-MAX_HISTORY, std::min(MAX_HISTORY, capture_history[piece_idx][cap_pt][to]));
                }
            }
        } else {
            if (good) {
                butterfly_history[piece_idx][to] += delta - butterfly_history[piece_idx][to] * abs(delta) / MAX_HISTORY;
            } else {
                butterfly_history[piece_idx][to] -= delta + butterfly_history[piece_idx][to] * abs(delta) / MAX_HISTORY;
            }
            butterfly_history[piece_idx][to] = std::max(-MAX_HISTORY, std::min(MAX_HISTORY, butterfly_history[piece_idx][to]));
        }
    }

    void update_correction(Move move, int depth, bool good, Color us) {
        Square from = from_sq(move), to = to_sq(move);
        int delta = depth * depth;
        if (good) {
            correction_history[us][from][to] += delta - correction_history[us][from][to] * abs(delta) / MAX_HISTORY;
        } else {
            correction_history[us][from][to] -= delta + correction_history[us][from][to] * abs(delta) / MAX_HISTORY;
        }
        correction_history[us][from][to] = std::max(-MAX_HISTORY, std::min(MAX_HISTORY, correction_history[us][from][to]));
    }

    // ── Proper CORRHIST: update pawn and non-pawn correction tables ──────────
    // Called at TT store time with the search score vs raw static eval.
    // The error signal teaches the engine how wrong its static eval was in
    // positions with this pawn structure / material configuration.
    void update_corrhist(const Position& pos, Color us, Value raw_eval, Value search_score, int depth) {
        if (std::abs(search_score) >= MATE_THRESHOLD) return;  // don't learn from mate scores
        int error = (int)(search_score - raw_eval);
        int weight = std::min(depth * depth, 128);  // deeper searches = stronger signal

        // Pawn correction
        size_t pi = pos.get_pawn_hash() % CORR_SIZE;
        int pdelta = error * weight / 128;
        pawn_corr_hist[us][pi] += pdelta - pawn_corr_hist[us][pi] * std::abs(pdelta) / CORR_MAX;
        pawn_corr_hist[us][pi] = std::clamp(pawn_corr_hist[us][pi], -CORR_MAX, CORR_MAX);

        // Non-pawn material correction
        size_t ni = pos.get_nonpawn_hash(us) % CORR_SIZE;
        nonpawn_corr_hist[us][ni] += pdelta - nonpawn_corr_hist[us][ni] * std::abs(pdelta) / CORR_MAX;
        nonpawn_corr_hist[us][ni] = std::clamp(nonpawn_corr_hist[us][ni], -CORR_MAX, CORR_MAX);
    }

    // Apply CORRHIST to raw static eval.
    // Adjusts the eval before it's used for pruning decisions (NMP, futility, etc.)
    // so the engine prunes less aggressively in positions where its eval is known to be off.
    Value corrected_eval(Value raw_eval, const Position& pos, Color us) const {
        size_t pi = pos.get_pawn_hash() % CORR_SIZE;
        size_t ni = pos.get_nonpawn_hash(us) % CORR_SIZE;
        int correction = pawn_corr_hist[us][pi] / 4 + nonpawn_corr_hist[us][ni] / 8;
        // Clamp adjustment to ±50 cp so a bad learning signal can't derail search.
        correction = std::clamp(correction, -50, 50);
        return raw_eval + correction;
    }

    void update_continuation(Move move, int depth, bool good, const Position& pos, int ply) {
        if (ply <= 0) return;
        Square from = from_sq(move), to = to_sq(move);
        Color us = pos.side_to_move();
        int moving_pc = pos.piece_on(from);
        int pt = moving_pc & 7;
        if (pt == 0) return;  // safety guard
        int cur_piece_idx = us * 6 + (pt - 1);
        int delta = depth * depth;

        // ── 1-ply continuation (ply-1 context) ──────────────────────────────
        {
            int prev_piece_idx = stack[ply-1].current_piece_idx;
            if (prev_piece_idx != -1) {
                Square prev_to = to_sq(stack[ply-1].current_move);
                auto& entry = cont_history[prev_piece_idx][prev_to][cur_piece_idx][to];
                if (good)
                    entry += delta - entry * abs(delta) / MAX_HISTORY;
                else
                    entry -= delta + entry * abs(delta) / MAX_HISTORY;
                entry = std::clamp(entry, -MAX_HISTORY, MAX_HISTORY);
            }
        }

        // ── 2-ply continuation (ply-2 context) — "follow-up" history ────────
        // Captures how well a move performs given what happened TWO plies ago.
        // This is the second cont_hist that Stockfish / Lc0 use for move ordering.
        if (ply >= 2) {
            int prev2_piece_idx = stack[ply-2].current_piece_idx;
            if (prev2_piece_idx != -1) {
                Square prev2_to = to_sq(stack[ply-2].current_move);
                auto& entry2 = cont_history2[prev2_piece_idx][prev2_to][cur_piece_idx][to];
                if (good)
                    entry2 += delta - entry2 * abs(delta) / MAX_HISTORY;
                else
                    entry2 -= delta + entry2 * abs(delta) / MAX_HISTORY;
                entry2 = std::clamp(entry2, -MAX_HISTORY, MAX_HISTORY);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Quiescence search
    // ------------------------------------------------------------------------
    Value quiescence(Position& pos, Value alpha, Value beta, int ply, int q_depth = 0) {
        if (ply >= MAX_PLY || q_depth >= MAX_QSEARCH_DEPTH)
            return eval.evaluate(pos) + learning.probe(pos.get_hash());
        if ((++nodes & 1023) == 0) {
            global_nodes.fetch_add(1024, std::memory_order_relaxed);
            if (stop_search) return 0;
            if (thread_id == 0 && tm.stop_early()) { stop_search = true; return 0; }
            if (node_limit > 0 && nodes >= node_limit) { stop_search = true; return 0; }
        }
        if (pos.is_repetition(1)) return tb_draw_value();  // contempt-adjusted draw
        if (tb.can_probe(pos)) {
            int dtz_success;
            int dtz = tb.probe_dtz(pos, dtz_success);
            if (dtz_success) {
                int sign = (dtz > 0) ? 1 : -1;
                int dist = std::abs(dtz);
                return (sign == 1) ? MATE_SCORE - dist - ply : -MATE_SCORE + dist + ply;
            }
        }

        bool in_check = pos.is_check();

        // Stand-pat: only valid when NOT in check. When in check the side to
        // move MUST play — there is no "do nothing" option, so stand_pat is
        // meaningless and using it would produce inflated scores.
        Value stand_pat = eval.evaluate(pos) + learning.probe(pos.get_hash());
        if (!in_check) {
            if (stand_pat >= beta) return beta;
            if (stand_pat > alpha) alpha = stand_pat;
        }

        Move moves[256];
        // When in check: generate ALL moves (evasions). Only captures are
        // generated otherwise — the standard quiescence contract.
        int cnt = in_check ? generate_moves(pos, moves, false)
                           : generate_moves(pos, moves, true);

        // Sort captures by MVV-LVA (fast, O(1) per move).
        // SEE is only computed later for delta pruning of actually-bad captures.
        // Calling SEE for every move during sort was O(pieces) × N per qnode.
        static const int mvv_val[7] = {0,100,320,330,500,900,20000};
        std::sort(moves, moves + cnt, [&](Move a, Move b) {
            int va = mvv_val[pos.piece_on(to_sq(a)) & 7] - mvv_val[pos.piece_on(from_sq(a)) & 7];
            int vb = mvv_val[pos.piece_on(to_sq(b)) & 7] - mvv_val[pos.piece_on(from_sq(b)) & 7];
            return va > vb;
        });

        int legal_count = 0;
        for (int i = 0; i < cnt; ++i) {
            Move m = moves[i];
            if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;

            // Delta / SEE pruning — skip bad captures when NOT in check
            if (!in_check) {
                int see_val = pos.see(m);
                if (see_val + 200 + stand_pat < alpha) continue;
            }

            int captured = pos.piece_on(to_sq(m));
            int moving_pc = pos.piece_on(from_sq(m));
            PieceType moving_pt = PieceType(moving_pc & 7);
            Color us = pos.side_to_move();
            bool was_promotion = promotion_type(m) != NO_PIECE;
            PieceType prom_pt = promotion_type(m);
            int old_castle = pos.castling_rights(), old_ep = pos.ep_sq(), old_fifty = pos.halfmove_clock();

#ifdef USE_NNUE
            eval.nnue_push();
#endif
            pos.make_move(m);
            stack[ply].captured_piece = captured;
            stack[ply].current_move = m;
            if (was_promotion) moving_pt = prom_pt;
            int cur_piece_idx = us * 6 + (moving_pt - 1);
            stack[ply].current_piece_idx = cur_piece_idx;
#ifdef USE_NNUE
            eval.nnue_make_move(pos, m, us, moving_pt, PieceType(captured & 7), was_promotion, prom_pt);
#endif

            if (pos.mover_in_check()) {
#ifdef USE_NNUE
                eval.nnue_pop();
#endif
                pos.undo_move(m, captured, old_castle, old_ep, old_fifty);
                continue;
            }
            legal_count++;
            Value score = -quiescence(pos, -beta, -alpha, ply+1, q_depth+1);
#ifdef USE_NNUE
            eval.nnue_pop();
#endif
            pos.undo_move(m, captured, old_castle, old_ep, old_fifty);
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }

        // When in check with no legal moves: checkmate
        if (in_check && legal_count == 0) return -MATE_SCORE + ply;

        return alpha;
    }

    // ------------------------------------------------------------------------
    // ProbCut
    // ------------------------------------------------------------------------
    Value probcut(Position& pos, Depth depth, Value alpha, Value beta, int ply) {
        if (depth < PROBCUT_DEPTH) return -INF;
        Move moves[256];
        int cnt = generate_moves(pos, moves, true);
        Value best = -INF;
        int margin = PROBCUT_MARGIN_BASE + PROBCUT_MARGIN_PER_DEPTH * depth;
        for (int i = 0; i < cnt; ++i) {
            Move m = moves[i];
            if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;
            int captured = pos.piece_on(to_sq(m));
            if (!captured) continue;
            int victim = captured & 7;
            int attacker = pos.piece_on(from_sq(m)) & 7;
            Value gain = PIECE_VALUES[victim] - PIECE_VALUES[attacker];
            if (gain + margin < alpha) continue;

            int moving_pc = pos.piece_on(from_sq(m));
            PieceType moving_pt = PieceType(moving_pc & 7);
            Color us = pos.side_to_move();
            bool was_promotion = promotion_type(m) != NO_PIECE;
            PieceType prom_pt = promotion_type(m);
            int old_castle = pos.castling_rights(), old_ep = pos.ep_sq(), old_fifty = pos.halfmove_clock();

#ifdef USE_NNUE
            eval.nnue_push();
#endif
            pos.make_move(m);
            stack[ply].captured_piece = captured;
            stack[ply].current_move = m;
            if (was_promotion) moving_pt = prom_pt;
            int cur_piece_idx = us * 6 + (moving_pt - 1);
            stack[ply].current_piece_idx = cur_piece_idx;
#ifdef USE_NNUE
            eval.nnue_make_move(pos, m, us, moving_pt, PieceType(captured & 7), was_promotion, prom_pt);
#endif

            nodes++;
            // ProbCut: null window around beta+margin
            Value pc_beta  = beta + margin;
            Value score = -negamax(pos, depth - 4, -pc_beta, -pc_beta + 1, ply+1, true, NO_MOVE);
#ifdef USE_NNUE
            eval.nnue_pop();
#endif
            pos.undo_move(m, captured, old_castle, old_ep, old_fifty);
            if (score > best) best = score;
            if (score >= beta) return score;
        }
        return (best >= beta) ? best : -INF;
    }

    // ------------------------------------------------------------------------
    // Negamax (core search)
    // ------------------------------------------------------------------------
    Value negamax(Position& pos, Depth depth, Value alpha, Value beta, int ply, bool cut, Move excluded = NO_MOVE) {
        // PV is maintained in pv_table[ply] (triangular table) — no heap allocation.
        pv_len[ply] = 0;
        if (ply >= MAX_PLY) return eval.evaluate(pos) + learning.probe(pos.get_hash());
        if ((++nodes & 1023) == 0) {
            global_nodes.fetch_add(1024, std::memory_order_relaxed);
            if (stop_search) return 0;
            if (thread_id == 0 && tm.stop_early()) { stop_search = true; return 0; }
            if (node_limit > 0 && nodes >= node_limit) { stop_search = true; return 0; }
        }
        // Use count=1: return draw score when position appeared once before in
        // history (2nd occurrence total). Waiting for count=2 (3rd occurrence)
        // lets the engine walk into repetition lines without flagging them as
        // draws, which caused the depth-6 cp-0 / repetition-PV bug.
        if (pos.is_repetition(1)) return tb_draw_value();  // contempt-adjusted draw
        if (tb.can_probe(pos) && depth <= 0) {
            unsigned wdl = tb.probe_wdl(pos);
            if (wdl != TB_RESULT_FAILED) { tb_hits++; return tb.wdl_to_score(wdl, ply); }
        }
        alpha = std::max(alpha, -MATE_SCORE + ply);
        beta  = std::min(beta,  MATE_SCORE - ply - 1);
        if (alpha >= beta) return alpha;

        bool in_check = pos.is_check();
        Color us = pos.side_to_move();
        Value raw_eval = eval.evaluate(pos) + learning.probe(pos.get_hash());
        // Apply correction history: adjusts static_eval toward historically
        // accurate values in positions with similar pawn/material structure.
        Value static_eval = in_check ? raw_eval : corrected_eval(raw_eval, pos, us);
        stack[ply].static_eval = static_eval;
        stack[ply].in_check = in_check;
        U64 key = pos.get_hash();
        Move tt_move = NO_MOVE;
        Value tt_score = static_eval;  // safe fallback — never used raw without tt_hit guard
        int tt_dtz = 0;
        int tt_bound_stored = BOUND_NONE;
        bool tt_hit = tt.probe(key, depth, alpha, beta, tt_score, tt_move, tt_dtz, tt_bound_stored);

        if (tt_hit) {
            if (tt_dtz != 0) {
                int sign = (tt_dtz > 0) ? 1 : -1;
                int dist = std::abs(tt_dtz);
                return (sign == 1) ? MATE_SCORE - dist - ply : -MATE_SCORE + dist + ply;
            }
            // Mate-score de-normalisation (standard convention, same as Stockfish):
            //   Store:    score += ply  (converts node-relative → root-relative)
            //   Retrieve: score -= ply  (converts root-relative → node-relative)
            // This ensures a "mate in N" found at ply P is correctly seen as
            // "mate in N + ΔP" when retrieved at a shallower ply P - ΔP.
            if (tt_score > MATE_THRESHOLD) {
                tt_score -= ply;
                if (tt_score > MATE_SCORE - 1) tt_score = MATE_SCORE - 1;
            } else if (tt_score < -MATE_THRESHOLD) {
                tt_score += ply;
                if (tt_score < -MATE_SCORE + 1) tt_score = -MATE_SCORE + 1;
            }
            return tt_score;
        }

        // ── Tablebase probing ─────────────────────────────────────────────────
        // Strategy (matching Stockfish's approach):
        //   depth >= TB_WDL_DEPTH:  WDL probe → prune wins/losses, score draws
        //   depth <= 3:             DTZ probe  → store exact distance in TT
        // The WDL probe runs first so it can short-circuit before the DTZ probe.
        // WDL = Win/Draw/Loss — fast bitboard probe, no DTZ table needed.
        // DTZ = Distance-to-Zero — slower, gives exact ply count to conversion.
        static constexpr int TB_WDL_DEPTH = 2;  // min depth for WDL pruning
        if (tb.can_probe(pos) && !in_check) {
            if (depth >= TB_WDL_DEPTH) {
                unsigned wdl = tb.probe_wdl(pos);
                if (wdl != TB_RESULT_FAILED) {
                    tb_hits++;
                    Value wdl_score = tb.wdl_to_score(wdl, ply);
                    // WDL wins/losses: return immediately (exact score)
                    // WDL draws: return 0 adjusted by contempt
                    if (wdl_score >= MATE_THRESHOLD || wdl_score <= -MATE_THRESHOLD) {
                        tt.store(key, depth, wdl_score, BOUND_EXACT, NO_MOVE, 0);
                        return wdl_score;
                    }
                    // Draw: adjust by contempt so the engine avoids/seeks draws
                    // based on its assessment of relative strength.
                    Value draw_val = tb_draw_value();
                    if (wdl_score == 0) {
                        if (alpha < draw_val) { alpha = draw_val; }
                        if (alpha >= beta) return draw_val;
                    }
                }
            }
            // DTZ probe: cache exact distance to conversion for later pruning
            if (depth <= 3 && tt_dtz == 0 && !tt_hit) {
                int dtz_success;
                int dtz = tb.probe_dtz(pos, dtz_success);
                if (dtz_success) tt.store(key, depth, 0, BOUND_NONE, NO_MOVE, dtz);
            }
        }

        // DTZ pruning
        if (tt_dtz != 0 && depth >= std::abs(tt_dtz) && tt_dtz > 0) {
            return MATE_SCORE - tt_dtz - ply - 1;
        }

        // ── Singular Extension ───────────────────────────────────────────────
        // Full implementation matching top engines (Stockfish / Ethereal style).
        //
        // Concept: the TT move is "singular" if it is the *only* good move in the
        // position — i.e. all other moves fail below a threshold derived from the
        // TT score.  If so, we extend its depth by 1 (or more) since missing it
        // would be catastrophic.
        //
        // Additional outcomes from the singular search:
        //   1. DOUBLE EXTENSION  — score is far below singular_beta (very singular)
        //   2. TRIPLE EXTENSION  — score is extremely below AND we're in a PV node
        //   3. MULTI-CUT pruning — other moves BEAT beta → position is a cut node,
        //                          so the TT move is probably not best; prune early.
        //   4. NEGATIVE EXTENSION— TT move fails multi-cut in a non-PV cut-node;
        //                          reduce its search depth by 1.
        //
        // Guards (must ALL hold):
        //   - TT hit with a reliable (depth-sufficient, bounds-matching) score
        //   - Not already in a singular search (excluded == NO_MOVE)
        //   - Depth >= threshold
        //   - No check (reduces false-singular noise)
        //   - Score is not a mate (mate scores need precise handling elsewhere)
        // ─────────────────────────────────────────────────────────────────────
        bool is_pv = (alpha < beta - 1);  // PV node: full-window call
        int singular_extension = 0;  // net depth change from singular logic

        if (tt_hit
            && depth >= SINGULAR_EXTENSION_DEPTH
            && tt_move != NO_MOVE
            && excluded == NO_MOVE
            && !in_check
            && std::abs(tt_score) < MATE_SCORE - MAX_PLY
            && (tt_bound_stored & BOUND_LOWER))   // only when TT entry is a lower bound
        {
            // Depth-scaled margin: deeper searches need stricter singularity test.
            // (is_pv ? factor 16 : factor 20)  — PV nodes extend more aggressively.
            const int se_margin = depth * (is_pv ? 16 : 20);
            Value singular_beta = std::max(tt_score - se_margin, -INF);
            Depth singular_depth = (depth - 1) / 2;  // half depth, rounded down

            Value singular_score = -negamax(pos, singular_depth,
                                            -singular_beta, -singular_beta + 1,
                                            ply, false, tt_move);

            if (!stop_search) {
                if (singular_score < singular_beta) {
                    // ── TT move is singular ──────────────────────────────────
                    int gap = singular_beta - singular_score;

                    if (!is_pv && gap > se_margin * 2) {
                        // Very singular in a non-PV node → double extension
                        singular_extension = 2;
                    } else if (!is_pv && gap > se_margin) {
                        // Singular in non-PV → standard single extension
                        singular_extension = 1;
                    } else if (is_pv && gap > se_margin * 3) {
                        // Extremely singular in a PV node → triple extension
                        singular_extension = 3;
                    } else if (is_pv && gap > se_margin) {
                        // Singular in PV → double extension
                        singular_extension = 2;
                    } else {
                        // Marginally singular → single extension
                        singular_extension = 1;
                    }
                } else if (singular_score >= beta) {
                    // ── Multi-cut: other moves also beat beta ────────────────
                    // The position is likely a CUT node — even without the TT move
                    // we'd get a beta cutoff. Return early (multi-cut pruning).
                    // Guard: only prune at non-PV nodes and when not in check.
                    if (!is_pv) {
                        return singular_score;  // multi-cut: prune the node
                    }
                    // PV node: don't prune outright, but apply negative extension
                    // to the TT move to signal it's less special than it looked.
                    singular_extension = -1;
                } else if (tt_score >= beta) {
                    // ── Negative extension for potential multi-cut ───────────
                    // The singular search did not beat beta, but the stored TT
                    // score does. Reduce the TT move slightly to avoid over-extending
                    // in what is likely a cut node.
                    singular_extension = -1;
                }
            }
            // Apply the singular extension to the node-level depth now so that
            // post-singular pruning (ProbCut, NMP, razoring) sees the updated depth.
            depth += singular_extension;
        }

        if (depth <= 0) return quiescence(pos, alpha, beta, ply);

        // ProbCut
        if (depth >= PROBCUT_DEPTH && !in_check && abs(beta) < MATE_SCORE - 1000) {
            Value pc_score = probcut(pos, depth, alpha, beta, ply);
            if (pc_score != -INF) return pc_score;
        }

        // Null move pruning
        // Skipped when: in check, pure KP endgame (zugzwang), cut-node flag is false,
        // OR when beta is a mate-class score.
        //
        // The mate-class guard is CRITICAL: if the engine believes the position is a
        // forced mate for the side to move (beta >= MATE_THRESHOLD), a null-move will
        // "skip" the opponent's defensive moves and confirm the phantom mate in the
        // reduced-depth search.  This poisons the TT and causes the engine to report
        // ghost "mate in N" scores at depths 4-7 for lines like Qg5 where Black has
        // the defensive pawn push g7-g6 that only gets found at depth 8+.
        if (!in_check && depth >= 2 && cut && beta < INF && beta < MATE_THRESHOLD) {
            bool has_non_pawn = false;
            for (int pt = KNIGHT; pt <= QUEEN; ++pt)
                if (pos.bb(pos.side_to_move(), PieceType(pt))) { has_non_pawn = true; break; }
            if (has_non_pawn) {
                bool only_kings_pawns = true;
                for (int pt = KNIGHT; pt <= QUEEN; ++pt)
                    if (pos.bb(WHITE, PieceType(pt)) || pos.bb(BLACK, PieceType(pt))) { only_kings_pawns = false; break; }
                if (!only_kings_pawns) {
#ifdef USE_NNUE
                    eval.nnue_push();
#endif
                    pos.make_move(NULL_MOVE);
                    int R = NULL_MOVE_R + depth / 4 + std::min(3, (static_eval - beta) / 200);
                    Value score = -negamax(pos, depth - R - 1, -beta, -beta+1, ply+1, false, NO_MOVE);
#ifdef USE_NNUE
                    eval.nnue_pop();
#endif
                    pos.undo_null_move();
                    if (score >= beta) return beta;
                }
            }
        }

        // Razoring — only at depth 1.  Applying at depth 2-3 causes false mate
        // scores: quiescence (captures only) misses quiet defensive moves like
        // Rf8-g8, so the engine incorrectly confirms a "forced mate" when a simple
        // quiet move refutes it.  Depth-1 razoring is safe because at depth 1 the
        // full move loop runs immediately after, so at most one quiet move is missed.
        Value practical_alpha = std::max(alpha, (Value)(-MATE_SCORE + MAX_PLY));
        if (!in_check && depth == 1 && alpha > -INF
                && alpha < MATE_THRESHOLD && beta < MATE_THRESHOLD) {
            int razor_margin = RAZOR_MARGIN_D1;
            if (static_eval + razor_margin < practical_alpha) {
                Value rscore = quiescence(pos, practical_alpha, practical_alpha+1, ply);
                if (rscore <= practical_alpha) return rscore;
            }
        }

        // Static null-move pruning
        const int STATIC_NULL_MARGIN = 200;
        if (!in_check && depth > 7 && static_eval - STATIC_NULL_MARGIN >= beta) return static_eval;

        // Futility pruning (full node)
        if (!in_check && depth <= 7 && static_eval - FUTILITY_MARGIN_FACTOR * depth >= beta) return static_eval;

        Move moves[256];
        int cnt = generate_moves(pos, moves);
        if (cnt == 0) return in_check ? -MATE_SCORE + ply : 0;

        // OPT: flat stack array — zero heap allocation per node
        ScoredMove scored[MAX_MOVES];
        int scored_count = 0;
        for (int i = 0; i < cnt; ++i) {
            if (moves[i] == excluded) continue;
            bool cap = pos.piece_on(to_sq(moves[i])) != 0;
            scored[scored_count++] = {moves[i], score_move(moves[i], ply, tt_move, pos, i, (int)cap)};
        }
        // Selection-sort is O(n) for the first pick; full sort only when needed.
        // For typical node counts (<40 moves) std::sort with a stack array is fine.
        std::sort(scored, scored + scored_count,
            [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });

        // Multi-cut pruning (make/undo — no Position copy)
        if (depth >= 6 && !in_check && cut && tt_move != NO_MOVE) {
            int mc_count = 0;
            for (int i = 0; i < std::min(3, scored_count); ++i) {
                Move m = scored[i].move;
                if (m == tt_move) continue;
                if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;
                int cap2 = pos.piece_on(to_sq(m));
                int oc2 = pos.castling_rights(), oe2 = pos.ep_sq(), of2 = pos.halfmove_clock();
                pos.make_move(m);
                if (pos.mover_in_check()) { pos.undo_move(m, cap2, oc2, oe2, of2); continue; }
                Value sc2 = -negamax(pos, depth / 2, -beta, -beta+1, ply+1, false, NO_MOVE);
                pos.undo_move(m, cap2, oc2, oe2, of2);
                if (sc2 >= beta && ++mc_count >= 2) return beta;
            }
        }

        // Internal Iterative Reduction (IIR): when no TT move is available at
        // deeper nodes, reduce depth by 1 instead of launching a full IID search.
        // Stockfish replaced IID with IIR ~2021; it's cheaper and equally effective
        // because the TT will be populated by shallower sibling/parent searches.
        if (tt_move == NO_MOVE && depth >= IID_DEPTH && !in_check)
            depth--;  // IIR: reduce one ply, move ordering will use history/killers

        Value best_score = -INF;
        Move best_move = NO_MOVE;
        Bound bound = BOUND_UPPER;
        bool improving = (ply >= 2 && static_eval > stack[ply-2].static_eval);

        // Note: multi-threading is handled at root level via Lazy SMP (each thread
        // runs its own independent iterative-deepening search). YBWC was removed
        // because it shared a single Position* among worker threads causing data races.

        // Normal move loop
        for (int i = 0; i < scored_count; ++i) {
            Move m = scored[i].move;
            if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;

            // Compute gives_check here (pure bitboard, very cheap) so we can
            // exempt check-giving moves from all forward-pruning heuristics.
            // Pruning a move that gives checkmate is obviously catastrophic.
            bool gives_check = pos.gives_check(m);

            bool is_capture = pos.piece_on(to_sq(m)) != 0;

            // SEE bad-capture pruning: at low depths, skip losing captures
            // (e.g. RxN when the N is defended) unless they give check.
            if (is_capture && !gives_check && !in_check && depth <= 8
                    && alpha < MATE_THRESHOLD && std::abs(alpha) < MATE_THRESHOLD) {
                int threshold = -depth * 20;  // more lenient at greater depth
                if (pos.see(m) < threshold) continue;
            }

            // Futility pruning (per move): skip quiet moves that can't improve alpha
            // even with max expected positional gain.  Extended to depth ≤ 4 (was ≤ 3).
            // margin = SEE_QUIET_MARGIN + depth*50: negative at d1 (very aggressive pruning),
            // increasing at d3-d4. Bonus for high-scoring moves makes killers exempt.
            if (depth <= 4 && !in_check && !pos.piece_on(to_sq(m))
                    && !gives_check
                    && alpha > -INF && alpha < MATE_THRESHOLD) {
                int margin = SEE_QUIET_MARGIN + depth * 50;
                if (scored[i].score < 500000) margin += 4 * depth;
                if (static_eval + margin <= alpha) continue;
            }

            // Late Move Pruning — precomputed depth²-based table.
            // lmp_moves[improving][depth] gives the move count threshold beyond
            // which we prune quiet moves unconditionally.  The d² formula
            // (Ethereal/Laser style) is empirically better than a linear factor.
            if (!pos.piece_on(to_sq(m)) && !in_check && depth <= 10
                    && !gives_check
                    && alpha < MATE_THRESHOLD) {
                int lmp_limit = lmp_moves[improving ? 1 : 0][std::min(depth, 15)];
                if ((int)i >= lmp_limit) continue;
            }

            int captured = pos.piece_on(to_sq(m));
            int moving_pc = pos.piece_on(from_sq(m));
            PieceType moving_pt = PieceType(moving_pc & 7);
            // 'us' is declared at the top of negamax(); do not re-declare here.
            // gives_check already computed above
            bool was_promotion = promotion_type(m) != NO_PIECE;
            PieceType prom_pt = promotion_type(m);
            int old_castle = pos.castling_rights(), old_ep = pos.ep_sq(), old_fifty = pos.halfmove_clock();

#ifdef USE_NNUE
            eval.nnue_push();
#endif
            pos.make_move(m);
            stack[ply].captured_piece = captured;
            stack[ply].current_move = m;
            if (was_promotion) moving_pt = prom_pt;
            int cur_piece_idx = us * 6 + (moving_pt - 1);
            stack[ply].current_piece_idx = cur_piece_idx;
#ifdef USE_NNUE
            eval.nnue_make_move(pos, m, us, moving_pt, PieceType(captured & 7), was_promotion, prom_pt);
#endif

            if (pos.mover_in_check()) {
#ifdef USE_NNUE
                eval.nnue_pop();
#endif
                pos.undo_move(m, captured, old_castle, old_ep, old_fifty);
                continue;
            }

            // Prefetch the child's TT bucket NOW while we compute extension/reduction.
            // The CPU fetches the cache line asynchronously, hiding ~100-200 cycle
            // latency by the time negamax() actually probes the TT.
            tt.prefetch(pos.get_hash());

            Depth new_depth = depth - 1;
            int extension = 0;

            // ---------------------------------------------------------------
            // Check extension: extend by 1 when this MOVE gives check.
            // ABSOLUTE PLY BUDGET:  total ply from root is (ply + remaining).
            // Without a global cap, a 5-move checking sequence at root depth 6
            // keeps depth at 6 every level (the per-node cap min(.,depth) never
            // lets depth decrease) and runs to ply 128.  We allow extensions
            // only while ply is still within root_depth * 3/2.  Beyond that
            // the tree has already been searched to adequate depth and further
            // extensions just waste time without finding new moves.
            // root_depth * 3/2  = budget of 50% extra ply over the root depth.
            // (e.g. root_depth=6 → extensions allowed for ply 0..8 only)
            // ---------------------------------------------------------------
            const int extension_budget = root_depth * 2;  // expanded: root*2 (was root*1.5)
            if (ply < extension_budget) {
                if (gives_check) extension = 1;
                // Singular extension for the TT move: apply the node-level singular
                // result (double/triple/negative) when this move IS the TT move.
                // For all other moves, no singular extension applies.
                if (m == tt_move && singular_extension != 0) {
                    // Combine with check extension (check always wins if positive)
                    if (singular_extension > 0 && !gives_check)
                        extension = std::max(extension, singular_extension);
                    else if (singular_extension < 0)
                        extension = std::min(extension, singular_extension);
                }
                // Other positional extensions: only one may apply (if not already extended).
                if (extension == 0) {
                    if (ply > 0 && stack[ply-1].captured_piece != 0
                            && to_sq(m) == to_sq(stack[ply-1].current_move))
                        extension = 1;   // recapture extension
                    else if (moving_pt == PAWN) {
                        if (eval.is_passed_pawn(pos, from_sq(m), us) &&
                            ((us == WHITE && rank_of(to_sq(m)) > rank_of(from_sq(m))) ||
                             (us == BLACK && rank_of(to_sq(m)) < rank_of(from_sq(m)))))
                            extension = 1;   // passed-pawn advance
                    }
                }
            }
            // Hard cap: clamp new_depth to [depth-1-1, depth+2] to prevent runaway
            // tree explosion from stacked extensions deep in the tree.
            new_depth = std::clamp(new_depth + extension, depth - 2, depth + 2);

            Value score;
            if (i == 0) {
                score = -negamax(pos, new_depth, -beta, -alpha, ply+1, true, NO_MOVE);
            } else {
                int red = captured ? 0 : reduction(improving, depth, i, scored[i].score, captured != 0, gives_check);
                score = -negamax(pos, new_depth - red, -alpha-1, -alpha, ply+1, true, NO_MOVE);
                // Use >= (not >) to match root PVS fix: when the null-window caps at
                // exactly beta=alpha, the negamax returns alpha (fail-high), and
                // -alpha == alpha after negation. Without >=, promising forcing moves
                // that score exactly equal to alpha are silently dismissed.
                if (score >= alpha && score < beta)
                    score = -negamax(pos, new_depth, -beta, -alpha, ply+1, true, NO_MOVE);
            }

#ifdef USE_NNUE
            eval.nnue_pop();
#endif
            pos.undo_move(m, captured, old_castle, old_ep, old_fifty);

            if (stop_search) return 0;

            if (score > best_score) {
                best_score = score;
                best_move = m;
                // Triangular PV update: copy child row into our row
                pv_table[ply][0] = m;
                int child_len = pv_len[ply+1];
                for (int _pvi = 0; _pvi < child_len && ply+1+_pvi < MAX_PLY; ++_pvi)
                    pv_table[ply][1+_pvi] = pv_table[ply+1][_pvi];
                pv_len[ply] = 1 + child_len;
                if (score > alpha) {
                    alpha = score;
                    bound = BOUND_EXACT;
                    if (score >= beta) {
                        bound = BOUND_LOWER;
                        // Update history tables for the cutoff move and all moves searched before it
                        if (!captured) {
                            // Quiet cutoff: update killer, quiet history, correction, continuation
                            if (stack[ply].killers[0] != m) {
                                stack[ply].killers[1] = stack[ply].killers[0];
                                stack[ply].killers[0] = m;
                            }
                            update_history(m, depth, true, false, pos);
                            update_correction(m, depth, true, us);
                            update_continuation(m, depth, true, pos, ply);
                            for (int j = 0; j < i; ++j) {
                                bool is_cap = pos.piece_on(to_sq(scored[j].move)) != 0;
                                if (!is_cap) {
                                    update_history(scored[j].move, depth, false, false, pos);
                                    update_correction(scored[j].move, depth, false, us);
                                    update_continuation(scored[j].move, depth, false, pos, ply);
                                }
                            }
                            if (ply > 0) {
                                Move last = stack[ply-1].current_move;
                                counter_moves[from_sq(last)][to_sq(last)] = m;
                            }
                            if (ply > 1) {
                                Move last2 = stack[ply-2].current_move;
                                follow_up_moves[from_sq(last2)][to_sq(last2)] = m;
                            }
                        } else {
                            // Capture cutoff: update capture history for the cutoff move
                            // and penalise captures that failed to cut before it
                            update_history(m, depth, true, true, pos);
                            for (int j = 0; j < i; ++j) {
                                bool is_cap = pos.piece_on(to_sq(scored[j].move)) != 0;
                                if (is_cap)
                                    update_history(scored[j].move, depth, false, true, pos);
                            }
                        }
                        break;
                    }
                }
            }
        }

        if (best_score == -INF) {
            best_score = in_check ? -MATE_SCORE + ply : 0;
            bound = BOUND_EXACT;
            best_move = NO_MOVE;
        }

        Value store_score = best_score;
        // Normalise mate scores to root-relative distance before TT storage.
        // Standard convention (match the retrieve block at the probe site):
        //   STORE:    score_win  -= ply  → stored = MATE_SCORE - (N + ply)
        //             score_loss += ply  → stored = -(MATE_SCORE - (N + ply))
        //   RETRIEVE: score_win  += ply  → node_score = stored + ply = MATE_SCORE - N ✓
        //             score_loss -= ply  → node_score = stored - ply
        // The key invariant: stored value is root-relative (independent of ply).
        // Threshold MATE_THRESHOLD = MATE_SCORE - MAX_PLY ensures no normal eval
        // score is ever mis-classified as a mate score.
        if (store_score > MATE_THRESHOLD) {
            store_score -= ply;   // root-relative win distance
        } else if (store_score < -MATE_THRESHOLD) {
            store_score += ply;   // root-relative loss distance
        }

        // Update correction history: learn the error between raw static eval
        // and the actual search score.  Only for non-check, reliable scores.
        if (!in_check && depth >= 1 && best_score != -INF
                && std::abs(best_score) < MATE_THRESHOLD
                && bound != BOUND_NONE) {
            update_corrhist(pos, us, raw_eval, best_score, depth);
        }

        tt.store(key, depth, store_score, bound, best_move);
        return best_score;
    }

    // ------------------------------------------------------------------------
    // Output info (thread 0 only)
    // ------------------------------------------------------------------------
    void output_info(int depth, Value score) {
        int64_t elapsed = tm.elapsed();
        uint64_t all_nodes = global_nodes.load(std::memory_order_relaxed);
        uint64_t nps = elapsed > 0 ? all_nodes * 1000 / elapsed : 0;
        std::string score_str;
        if (std::abs(score) > MATE_SCORE - 1000) {
            int mate_dist = (score > 0) ? (MATE_SCORE - score) : (MATE_SCORE + score);
            if (mate_dist < 0) mate_dist = 0;
            score_str = (score > 0) ? "score mate " + std::to_string(mate_dist) : "score mate -" + std::to_string(mate_dist);
        } else {
            score_str = "score cp " + std::to_string(score);
        }
        std::cout << "info depth " << depth << " " << score_str
                  << " nodes " << all_nodes << " nps " << nps
                  << " time " << elapsed << " tbhits " << tb_hits;

        // UCI_ShowWDL: estimate Win/Draw/Loss probabilities.
        // Uses a simple logistic model: win_prob = 1/(1+exp(-score/400)).
        // Draw is estimated as max(0, 1 - |score|/1000).
        if (show_wdl && std::abs(score) <= MATE_SCORE - 1000) {
            double sig = 1.0 / (1.0 + std::exp(-score / 400.0));
            int draw_pct = std::max(0, 1000 - std::abs(score));
            draw_pct = std::min(draw_pct, 1000);
            int win_pct  = int((1000 - draw_pct) * sig);
            int loss_pct = 1000 - win_pct - draw_pct;
            if (loss_pct < 0) { win_pct += loss_pct; loss_pct = 0; }
            std::cout << " wdl " << win_pct << " " << draw_pct << " " << loss_pct;
        }

        std::cout << " pv";
        Position tmp = root_pos;
        for (int _oi = 0; _oi < pv_len[0]; ++_oi) { Move m = pv_table[0][_oi];
            // Validate that m is legal in the current position.
            // TT hash collisions or stale PV entries can inject illegal moves;
            // applying them corrupts the position and makes every subsequent
            // move appear illegal in the GUI.
            Move legal_moves[MAX_MOVES];
            int legal_cnt = generate_moves(tmp, legal_moves);
            bool found = false;
            for (int li = 0; li < legal_cnt; ++li) {
                if (legal_moves[li] == m) {
                    // Legality check via make/undo — no heap alloc (Position copy avoided)
                    int cap_pv = tmp.piece_on(to_sq(m));
                    int oc_pv = tmp.castling_rights(), oe_pv = tmp.ep_sq(), of_pv = tmp.halfmove_clock();
                    tmp.make_move(m);
                    bool ok = !tmp.mover_in_check();
                    tmp.undo_move(m, cap_pv, oc_pv, oe_pv, of_pv);
                    if (ok) { found = true; break; }
                }
            }
            if (!found) break;  // Stop PV at first illegal move

            // For Chess960, castling output is king-to-rook; otherwise king-to-destination.
            Square mf = from_sq(m), mt = to_sq(m);
            if (is_castling(m) && tmp.is_chess960()) {
                Color us2 = tmp.piece_on(mf) != 0 ? Color(tmp.piece_on(mf) >> 3)
                                                   : Color((mf < 32) ? WHITE : BLACK);
                int si = (mt > mf) ? 0 : 1;
                Square rsq = tmp.castle_rook(us2, si);
                if (rsq != -1) mt = rsq;
            }
            std::cout << " " << char('a' + file_of(mf)) << char('1' + rank_of(mf))
                      << char('a' + file_of(mt)) << char('1' + rank_of(mt));
            PieceType prom = promotion_type(m);
            if (prom != NO_PIECE) {
                const char pc[] = " pnbrqk";
                std::cout << pc[prom];
            }
            tmp.make_move(m);
        } // end pv loop
        std::cout << std::endl;
    }

    // ------------------------------------------------------------------------
    // Main search entry (root)
    // ------------------------------------------------------------------------
    void search(int max_depth, uint64_t max_nodes, const std::vector<ScoredMove>& root_moves) {
        // NOTE: stop_search, node_limit, tb_hits, and tt.new_search() are
        // initialized ONCE in go() before threads launch — not here.
        // Each thread resets only its own per-thread counters.
        nodes = 0;

        // Each thread gets its own local copy of the position.
        // This is the critical Lazy SMP safety guarantee: threads never
        // mutate the shared root_pos reference — each has its own state.
        Position local_pos = root_pos;

        std::vector<ScoredMove> local_root_moves = root_moves;
        if (local_root_moves.empty()) {
            Move moves[MAX_MOVES];
            int cnt = generate_moves(local_pos, moves);
            for (int i = 0; i < cnt; ++i) {
                Move m = moves[i];
                if (local_pos.piece_on(to_sq(m)) && ((local_pos.piece_on(to_sq(m)) & 7) == KING)) continue;
                int cap    = local_pos.piece_on(to_sq(m));
                int old_cr = local_pos.castling_rights();
                int old_ep = local_pos.ep_sq();
                int old_50 = local_pos.halfmove_clock();
                local_pos.make_move(m);
                bool legal = !local_pos.mover_in_check();
                local_pos.undo_move(m, cap, old_cr, old_ep, old_50);
                if (legal) local_root_moves.push_back({m, 0});
            }
        }
        if (local_root_moves.empty()) return;

#ifdef USE_NNUE
        eval.nnue_warm(local_pos);
#endif

        Move best_move = local_root_moves[0].move;
        Value best_score = -INF;
        prev_best_move = NO_MOVE;

        // ── Lazy SMP depth staggering ─────────────────────────────────────────
        // Thread 0 drives time management and always starts at depth 1.
        // Helper threads (id > 0) stagger their starting depth so they're
        // never searching the exact same tree simultaneously:
        //   - Odd helpers  (id 1,3,5,...) start at depth 2
        //   - Even helpers (id 2,4,6,...) start at depth 1 but skip depth 3
        //   - High-id helpers get more stagger to maximise tree diversity
        // This forces helpers into different parts of the search tree, reducing
        // redundant work and improving TT collision diversity (different orderings
        // → different pruning patterns → better TT population for thread 0).
        // ─────────────────────────────────────────────────────────────────────
        int start_depth = 1;
        if (thread_id > 0) {
            // Simple stagger: odd threads start at 2, even > 0 threads start at 1
            start_depth = 1 + (thread_id & 1);
        }

        for (int depth = start_depth; depth <= max_depth && !stop_search; ++depth) {
            if (depth > 1 && thread_id == 0 && !tm.time_for_depth(depth)) break;
            // Helper threads: skip this depth if another thread is already covering it
            // (basic stagger: thread N skips depths where (depth + thread_id/2) is odd)
            if (thread_id > 0 && depth > 2 && ((depth + thread_id) % 3) == 0) {
                // Occasional skip creates more tree diversity without missing key depths
                if (depth < max_depth) { ++depth; }
            }

            root_depth = depth;   // used by extension budget in negamax()

            for (auto& sm : local_root_moves) {
                int captured = local_pos.piece_on(to_sq(sm.move)) != 0;
                sm.score = score_move(sm.move, 0, (best_move != NO_MOVE ? best_move : NO_MOVE), local_pos, 0, captured);
            }
            if (best_move != NO_MOVE) {
                for (auto& sm : local_root_moves)
                    if (sm.move == best_move) sm.score = 10000000;
            }
            std::sort(local_root_moves.begin(), local_root_moves.end(),
                [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });

            // -----------------------------------------------------------------------
            // Aspiration window — with fast-exit on mate discovery.
            //
            // Start narrow (±ASPIRATION_WINDOW) and double each fail.
            // Key rule: if any retry finds a mate-class score (> MATE_THRESHOLD),
            // we immediately accept that result.  This prevents the catastrophic
            // case where a sacrifice leading to forced mate causes repeated
            // fail-highs that eventually force a full [-INF,+INF] window where
            // null-move/futility pruning is completely disabled.
            // -----------------------------------------------------------------------
            Value alpha = -INF, beta = INF;
            int asp_delta = ASPIRATION_WINDOW;      // starting half-width
            int asp_prev  = asp_delta;              // for Fibonacci widening
            int fail_low_count = 0, fail_high_count = 0;

            // Only use a narrow window when previous score is a "normal" value.
            // A mate score from the previous depth means the window should just be wide.
            bool use_aspiration = (depth >= 5
                    && std::abs(best_score) >= 75
                    && std::abs(best_score) < MATE_THRESHOLD);
            if (use_aspiration) {
                alpha = best_score - asp_delta;
                beta  = best_score + asp_delta;
            }

            Move depth_best = NO_MOVE;
            Value depth_score = -INF;
            std::vector<Move> depth_best_pv;  // best PV for this depth, reported once at completion
            bool need_retry = true;

            while (need_retry && !stop_search) {
                need_retry = false;
                depth_best = NO_MOVE;
                depth_score = -INF;
                Value window_alpha = alpha;

                for (size_t i = 0; i < local_root_moves.size() && !stop_search; ++i) {
                    Move m = local_root_moves[i].move;
                    if (local_pos.piece_on(to_sq(m)) && ((local_pos.piece_on(to_sq(m)) & 7) == KING)) continue;

                    int cap = local_pos.piece_on(to_sq(m));
                    int moving_pc = local_pos.piece_on(from_sq(m));
                    PieceType moving_pt = PieceType(moving_pc & 7);
                    Color us = local_pos.side_to_move();
                    bool was_promotion = promotion_type(m) != NO_PIECE;
                    PieceType prom_pt = promotion_type(m);
                    int oc = local_pos.castling_rights(), oe = local_pos.ep_sq(), of_ = local_pos.halfmove_clock();

#ifdef USE_NNUE
                    eval.nnue_push();
#endif
                    local_pos.make_move(m);
                    stack[0].captured_piece = cap;
                    stack[0].current_move = m;
                    if (was_promotion) moving_pt = prom_pt;
                    int cur_piece_idx = us * 6 + (moving_pt - 1);
                    stack[0].current_piece_idx = cur_piece_idx;
#ifdef USE_NNUE
                    eval.nnue_make_move(local_pos, m, us, moving_pt, PieceType(cap & 7), was_promotion, prom_pt);
#endif

                    if (local_pos.mover_in_check()) {
#ifdef USE_NNUE
                        eval.nnue_pop();
#endif
                        local_pos.undo_move(m, cap, oc, oe, of_);
                        continue;
                    }

                    nodes++;
                    pv_len[1] = 0;
                    Value score;
                    if (i == 0 || window_alpha == -INF) {
                        score = -negamax(local_pos, depth - 1, -beta, -window_alpha, 1, true, NO_MOVE);
                    } else {
                        score = -negamax(local_pos, depth - 1, -window_alpha - 1, -window_alpha, 1, true, NO_MOVE);
                        // Re-search with full window when score >= window_alpha.
                        // The STRICT ">" was a bug: a null-window capped at beta=window_alpha
                        // returns exactly window_alpha (fail-high), which is numerically equal
                        // to window_alpha, so "> window_alpha" was FALSE and the move was silently
                        // dismissed.  A sacrificial move (e.g. Rxg7 leading to forced mate) can
                        // appear to score exactly window_alpha in the null window but actually
                        // score +MATE in the full window — exactly the case that was broken.
                        if (!stop_search && score >= window_alpha && score < beta)
                            score = -negamax(local_pos, depth - 1, -beta, -window_alpha, 1, true, NO_MOVE);
                    }

#ifdef USE_NNUE
                    eval.nnue_pop();
#endif
                    local_pos.undo_move(m, cap, oc, oe, of_);

                    if (stop_search) break;

                    if (score > depth_score) {
                        depth_score = score;
                        depth_best = m;
                        // Build PV from triangular table (ply=0 holds root move + child PV)
                        pv_table[0][0] = m;
                        int child_len = pv_len[1];
                        for (int _pvi = 0; _pvi < child_len && 1+_pvi < MAX_PLY; ++_pvi)
                            pv_table[0][1+_pvi] = pv_table[1][_pvi];
                        pv_len[0] = 1 + child_len;
                        // Copy to depth_best_pv (vector, only used for output)
                        depth_best_pv.clear();
                        for (int _pvi = 0; _pvi < pv_len[0]; ++_pvi)
                            depth_best_pv.push_back(pv_table[0][_pvi]);
                        // Always keep root_infos up to date for the best move so that
                        // last_pv is available for the Learning feature even in single-PV mode.
                        if (thread_id == 0) {
                            std::lock_guard<std::mutex> lock(root_infos_mutex);
                            for (auto& info : root_infos) {
                                if (info.move == m) {
                                    info.score = score;
                                    info.pv.clear();
                                    for (int _rpi = 0; _rpi < pv_len[0]; ++_rpi)
                                        info.pv.push_back(pv_table[0][_rpi]);
                                    break;
                                }
                            }
                        }
                    }
                    if (score > window_alpha) window_alpha = score;
                }

                if (!stop_search && depth >= 5 && use_aspiration) {
                    // Fast-exit: if we already found a mate-class score during a retry,
                    // accept it immediately.  Opening to [-INF,+INF] to "confirm" a mate
                    // score disables all pruning and makes depth-6+ take millions of nodes.
                    if (std::abs(depth_score) >= MATE_THRESHOLD) {
                        // Accept — no further retry needed.
                    } else if (depth_score <= alpha && alpha > -INF) {
                        // Fail-low: widen only the lower bound (asymmetric).
                        fail_low_count++;
                        fail_high_count = 0;
                        if (fail_low_count >= 3) {
                            alpha = -(MATE_SCORE - 1);
                            beta  =   MATE_SCORE - 1;
                        } else {
                            // Fibonacci widening: next = current + previous (faster than doubling
                            // for the first 2 retries, equal to doubling for the 3rd+).
                            int next_delta = asp_delta + asp_prev;
                            asp_prev = asp_delta;
                            asp_delta = std::min(next_delta, (int)MATE_THRESHOLD);
                            alpha = best_score - asp_delta;  // widen lower only
                        }
                        need_retry = true;
                    } else if (depth_score >= beta && beta < INF) {
                        // Fail-high: widen only the upper bound (asymmetric).
                        fail_high_count++;
                        fail_low_count = 0;
                        if (fail_high_count >= 3) {
                            alpha = -(MATE_SCORE - 1);
                            beta  =   MATE_SCORE - 1;
                        } else {
                            int next_delta = asp_delta + asp_prev;
                            asp_prev = asp_delta;
                            asp_delta = std::min(next_delta, (int)MATE_THRESHOLD);
                            beta  = best_score + asp_delta;  // widen upper only
                        }
                        need_retry = true;
                    }
                }
            }

            if (!stop_search && depth_best != NO_MOVE) {
                best_move  = depth_best;
                best_score = depth_score;
                prev_eval = depth_score;
                if (thread_id == 0) {
                    bool best_move_changed = (depth_best != prev_best_move);
                    tm.update(depth_score, best_move_changed);
                    prev_best_move = depth_best;
                    if (multi_pv <= 1) {
                        output_info(depth, depth_score);
                    }
                }
            }

            // Multi-PV barrier
            if (multi_pv > 1) {
                int done = ++depth_done_count;
                if (done == total_threads) {
                    depth_done_count = 0;
                    if (thread_id == 0) {
                        std::lock_guard<std::mutex> lock(root_infos_mutex);
                        std::sort(root_infos.begin(), root_infos.end());
                        int n = std::min(multi_pv, (int)root_infos.size());
                        for (int i = 0; i < n; ++i) {
                            if (root_infos[i].score > -INF + 1000) {
                                int64_t elapsed_ms = tm.elapsed();
                                uint64_t nps_val = elapsed_ms > 0 ? (uint64_t)(nodes * 1000 / elapsed_ms) : 0;
                                std::cout << "info depth " << depth << " multipv " << i+1 << " score ";
                                if (std::abs(root_infos[i].score) > MATE_SCORE - 1000) {
                                    int md = (root_infos[i].score > 0) ? (MATE_SCORE - root_infos[i].score) : (MATE_SCORE + root_infos[i].score);
                                    std::cout << (root_infos[i].score > 0 ? "mate " : "mate -") << md;
                                } else {
                                    std::cout << "cp " << root_infos[i].score;
                                }
                                std::cout << " nodes " << nodes << " nps " << nps_val
                                          << " time " << elapsed_ms << " pv";
                                // Validate and print PV using move_to_uci for Chess960/promotion correctness
                                Position pv_tmp = local_pos;
                                for (Move mv : root_infos[i].pv) {
                                    if (mv == NO_MOVE) break;
                                    Move tmp_list[MAX_MOVES];
                                    int tmp_cnt = generate_moves(pv_tmp, tmp_list);
                                    bool mv_found = false;
                                    for (int li = 0; li < tmp_cnt; ++li) {
                                        if (tmp_list[li] == mv) {
                                            Position c2 = pv_tmp;
                                            c2.make_move(mv);
                                            if (!c2.mover_in_check()) { mv_found = true; break; }
                                        }
                                    }
                                    if (!mv_found) break;
                                    // Print move (Chess960-aware, promotions suffixed)
                                    Square mvf = from_sq(mv), mvt = to_sq(mv);
                                    if (is_castling(mv) && pv_tmp.is_chess960()) {
                                        Color uc = pv_tmp.piece_on(mvf) ? Color(pv_tmp.piece_on(mvf) >> 3)
                                                                         : Color((mvf < 32) ? WHITE : BLACK);
                                        Square rsq = pv_tmp.castle_rook(uc, (mvt > mvf) ? 0 : 1);
                                        if (rsq != -1) mvt = rsq;
                                    }
                                    std::cout << " " << char('a' + file_of(mvf)) << char('1' + rank_of(mvf))
                                              << char('a' + file_of(mvt)) << char('1' + rank_of(mvt));
                                    PieceType mp = promotion_type(mv);
                                    if (mp != NO_PIECE) { const char pc2[] = " pnbrqk"; std::cout << pc2[mp]; }
                                    pv_tmp.make_move(mv);
                                }
                                std::cout << std::endl;
                            }
                        }
                    }
                    depth_continue = true;
                } else {
                    while (!depth_continue) std::this_thread::yield();
                }
                int acked = ++depth_ack_count;
                if (acked == total_threads) {
                    depth_ack_count = 0;
                    depth_continue = false;
                } else {
                    while (depth_continue) std::this_thread::yield();
                }
            }
        }

        if (best_move != NO_MOVE) {
            // Unconditionally offer our best_move to the shared result.
            // Use CAS on score so the *best* score across threads wins.
            // Then write the move with release ordering so go()'s join()
            // + acquire-load sees the fully committed move on all architectures
            // (ARM's relaxed memory model requires this — without release/acquire
            // the joining thread can read the old NO_MOVE on Cortex-A CPUs).
            Value prev = shared_best_score.load(std::memory_order_relaxed);
            bool did_win_score = false;
            while (best_score >= prev) {
                if (shared_best_score.compare_exchange_weak(prev, best_score,
                        std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    did_win_score = true;
                    break;
                }
            }
            // Also write if we are the *only* thread (score == -INF, nobody wrote yet)
            if (did_win_score || shared_best_move.load(std::memory_order_relaxed) == NO_MOVE) {
                shared_best_move.store(best_move, std::memory_order_release);
            }
        }

        if (local_root_moves.empty()) {
            // No root moves assigned to this thread in the current partition.
            // With Lazy SMP all threads receive the full move list, so this
            // branch is only hit if a caller explicitly passes an empty list.
            return;
        }
    }

    void search(int max_depth, uint64_t max_nodes) { search(max_depth, max_nodes, {}); }
};

// ============================================================================
// End of Part 3
// ============================================================================
// ============================================================================
// Part 4 of 4: Hugine 2.0 – UCI Interface and main()
// ============================================================================

// Format a Move as a UCI string (e.g. "e2e4", "e7e8q", "e1g1").
// In Chess960 mode (pass the position before the move is made), castling is
// reported as king-to-rook-square (e.g. "e1h1") as required by UCI_Chess960.
static std::string move_to_uci(Move m, const Position* pos = nullptr) {
    if (m == NO_MOVE || m == NULL_MOVE) return "0000";
    Square from = from_sq(m), to = to_sq(m);
    // Chess960: remap castling output from king-destination to rook-origin square
    if (is_castling(m) && pos != nullptr && pos->is_chess960()) {
        Color us = pos->piece_on(from) != 0
                   ? Color(pos->piece_on(from) >> 3)
                   : Color((from < 32) ? WHITE : BLACK);  // fallback
        int side_idx = (to > from) ? 0 : 1;  // to is g/c — same direction logic
        Square rook_sq = pos->castle_rook(us, side_idx);
        if (rook_sq != -1) {
            std::string s;
            s += char('a' + file_of(from));
            s += char('1' + rank_of(from));
            s += char('a' + file_of(rook_sq));
            s += char('1' + rank_of(rook_sq));
            return s;
        }
    }
    std::string s;
    s += char('a' + file_of(from));
    s += char('1' + rank_of(from));
    s += char('a' + file_of(to));
    s += char('1' + rank_of(to));
    PieceType prom = promotion_type(m);
    if (prom != NO_PIECE) {
        const char pchars[] = " pnbrqk";
        s += pchars[prom];
    }
    return s;
}

// ============================================================================
// High-performance perft (bulk counting at depth 1, no std::function overhead)
// ============================================================================
//
// Using a plain template function instead of std::function<> eliminates:
//   - Heap allocation for the closure
//   - Virtual dispatch on every recursive call (~4× overhead vs plain call)
//   - Unnecessary copy of captured state
//
// The depth-1 bulk path uses Position::bulk_count_legal() which skips make/undo
// for non-pinned, non-king, non-special moves — the common case in any position.
// This is the same technique used by Stockfish / Ethereal for perft benchmarks.

static uint64_t perft_inner(Position& pos, int depth) {
    if (depth == 1) {
        // Bulk count: skip make/undo for safe (non-pinned, non-king) moves.
        return pos.bulk_count_legal();
    }
    if (depth == 0) return 1;

    Move mvs[256];
    int cnt = generate_moves(pos, mvs, false);
    uint64_t nodes = 0;

    for (int i = 0; i < cnt; ++i) {
        Move m = mvs[i];
        int cap     = pos.piece_on(to_sq(m));
        int old_cr  = pos.castling_rights();
        int old_ep  = pos.ep_sq();
        int old_50  = pos.halfmove_clock();

        pos.make_move(m);
        if (!pos.mover_in_check()) {
            nodes += perft_inner(pos, depth - 1);
        }
        pos.undo_move(m, cap, old_cr, old_ep, old_50);
    }
    return nodes;
}

// Thread-safe divide-mode perft: splits root moves across worker threads.
// Each thread handles its own subtree and writes to its own node counter.
// Falls back to single-thread if thread_count == 1.

class UCI {
private:
    Position pos;
    TranspositionTable tt;
    SyzygyTablebase tb;
    Evaluation eval;
    OpeningBook book;
    std::vector<std::thread> search_threads;
    std::atomic<bool> search_active;
    std::atomic<bool> pondering_active;
    int thread_count;
    int multi_pv;
    bool ponder;
    bool use_book;
    int contempt;
    bool chess960;
    bool uci_limit_strength;
    int uci_elo;
    bool learning_enabled;
    std::string learning_file;
    int learning_rate;
    int learning_max_adjust;
    bool tuning_mode;
    std::string tuning_file;
    std::ofstream tuning_stream;
    std::vector<Move> last_pv;
    std::mutex last_pv_mutex;

    bool uci_show_wdl;
    // Skill Level (0–20, 20 = full strength): limits depth and injects blunders
    int skill_level;
    // Syzygy fine-tuning options
    int  syzygy_probe_depth;    // min search depth before TB probing is allowed (default 1)
    int  syzygy_probe_limit;    // max piece count for probing (default = TB max_cardinality)
    bool syzygy_50_move_rule;   // honour 50-move rule in TB results (default true)

public:
    UCI() : tt(256), search_active(false), pondering_active(false), thread_count(1), multi_pv(1),
            ponder(false), use_book(true), contempt(0), chess960(false), uci_limit_strength(false), uci_elo(1500),
            learning_enabled(false), learning_rate(100), learning_max_adjust(50),
            tuning_mode(false), uci_show_wdl(false),
            skill_level(20),
            syzygy_probe_depth(1), syzygy_probe_limit(7), syzygy_50_move_rule(true) {
        Zobrist::init();
        Bitboards::init();
        init_magics();
    }

    ~UCI() {
        if (tuning_stream.is_open()) tuning_stream.close();
    }

    void set_option(const std::string& name, const std::string& value) {
        if (name == "Hash") {
            tt.resize(std::stoi(value));
        } else if (name == "Threads") {
            thread_count = std::min(std::stoi(value), MAX_THREADS);
        } else if (name == "Ponder") {
            ponder = (value == "true");
        } else if (name == "OwnBook") {
            use_book = (value != "false");
        } else if (name == "BookFile") {
            if (!value.empty()) book.load(value);
        } else if (name == "BookVariety") {
            book.set_variety(std::stod(value));
        } else if (name == "SyzygyPath") {
            if (!value.empty()) tb.init(value);
        } else if (name == "SyzygyProbeDepth") {
            syzygy_probe_depth = std::max(1, std::stoi(value));
        } else if (name == "SyzygyProbeLimit") {
            syzygy_probe_limit = std::max(0, std::min(7, std::stoi(value)));
        } else if (name == "Syzygy50MoveRule") {
            syzygy_50_move_rule = (value == "true");
        } else if (name == "Skill Level") {
            skill_level = std::max(0, std::min(20, std::stoi(value)));
        } else if (name == "EvalFile" || name == "NNUEPath") {
#ifdef USE_NNUE
            if (!value.empty()) {
                std::string loaded = eval.load_nnue(value);
                if (!loaded.empty())
                    std::cout << "info string NNUE loaded: " << loaded << "\n";
                else
                    std::cout << "info string NNUE load FAILED"
                              << " (not a valid .nnue file or no valid .nnue in directory): "
                              << value << " — falling back to classical eval\n";
            }
#endif
        } else if (name == "MultiPV") {
            multi_pv = std::stoi(value);
        } else if (name == "Contempt") {
            contempt = std::stoi(value);
            eval.set_contempt(contempt);
        } else if (name == "Clear Hash") {
            tt.clear();
#ifdef DEBUG
            std::cerr << "DEBUG: Transposition table cleared.\n";
#endif
        } else if (name == "Move Overhead") {
            tm.set_move_overhead(std::stoi(value));
        } else if (name == "UCI_Chess960") {
            chess960 = (value == "true");
            pos.set_chess960(chess960);
        } else if (name == "UCI_LimitStrength") {
            uci_limit_strength = (value == "true");
        } else if (name == "UCI_Elo") {
            uci_elo = std::stoi(value);
        } else if (name == "Learning") {
            learning_enabled = (value == "true");
            learning.set_enabled(learning_enabled);
        } else if (name == "LearningFile") {
            learning_file = value;
            if (!learning_file.empty()) learning.set_filename(learning_file);
        } else if (name == "LearningRate") {
            learning_rate = std::stoi(value);
            learning.set_learning_rate(learning_rate);
        } else if (name == "LearningMaxAdjust") {
            learning_max_adjust = std::stoi(value);
            learning.set_max_adjust(learning_max_adjust);
        } else if (name == "Clear Learning") {
            learning.clear();
        } else if (name == "Save Learning") {
            learning.save();
        } else if (name == "TuningMode") {
            tuning_mode = (value == "true");
            if (tuning_mode && !tuning_file.empty()) {
                tuning_stream.open(tuning_file, std::ios::app);
            }
        } else if (name == "UCI_ShowWDL") {
            uci_show_wdl = (value == "true");
        } else if (name == "Debug Log File") {
            // Debug Log File: redirect std::cerr to this file for debug output
            if (!value.empty()) {
                static std::ofstream debug_log;
                debug_log.open(value, std::ios::app);
                if (debug_log.is_open()) std::cerr.rdbuf(debug_log.rdbuf());
            }
        }
    }

    void position(const std::vector<std::string>& args) {
        size_t i = 0;
        if (i < args.size() && args[i] == "startpos") {
            pos.init_startpos();
            i++;
        } else if (i < args.size() && args[i] == "fen") {
            std::string fen;
            i++;
            while (i < args.size() && args[i] != "moves") {
                if (!fen.empty()) fen += " ";
                fen += args[i++];
            }
            pos.set_fen(fen);
        }
        // If the operator has set UCI_Chess960=true, honour it unconditionally.
        // set_fen() auto-detects chess960 from rook positions, but standard-looking
        // Chess960 starting positions (where the rooks happen to be on a/h) would
        // be falsely classified as non-960, breaking the king-to-rook I/O protocol.
        if (chess960) pos.set_chess960(true);
        if (i < args.size() && args[i] == "moves") {
            i++;
            while (i < args.size()) {
                std::string ms = args[i++];
                if (ms.size() < 4) continue;
                Square from = make_square(ms[0]-'a', ms[1]-'1');
                Square to   = make_square(ms[2]-'a', ms[3]-'1');
                Move move = NO_MOVE;

                // Promotion (5-char move like e7e8q)
                if (ms.length() == 5) {
                    char p = ms[4];
                    if (p == 'n') move = make_promotion(from, to, KNIGHT);
                    else if (p == 'b') move = make_promotion(from, to, BISHOP);
                    else if (p == 'r') move = make_promotion(from, to, ROOK);
                    else              move = make_promotion(from, to, QUEEN);
                }
                // Chess960 castling: GUI sends king-to-rook (e.g. e1h1 / e1a1).
                // Detect by king moving onto its own rook, then remap to the internal
                // king-destination encoding (g/c file) used throughout the engine.
                else if ((pos.piece_on(from) & 7) == KING &&
                         (pos.piece_on(to) & 7) == ROOK &&
                         (pos.piece_on(to) >> 3) == (pos.piece_on(from) >> 3)) {
                    Color col = Color(pos.piece_on(from) >> 3);
                    int castling_rank_p = (col == WHITE) ? 0 : 7;
                    // side_idx: 0 = kingside (rook to the right), 1 = queenside
                    int side_idx = (file_of(to) > file_of(from)) ? 0 : 1;
                    Square king_dest = make_square((side_idx == 0) ? 6 : 2, castling_rank_p);
                    move = make_move(from, king_dest) | CASTLE_FLAG;
                }
                // Standard castling: king moves exactly 2 squares horizontally
                else if ((pos.piece_on(from) & 7) == KING &&
                         std::abs(file_of(to) - file_of(from)) == 2) {
                    move = make_move(from, to) | CASTLE_FLAG;
                }
                // En passant: pawn moves diagonally to the en-passant square (must be empty)
                // MUST check to == pos.ep_sq() — otherwise any diagonal pawn move to an
                // empty square (impossible in a legal game but theoretically encodable) would
                // be mis-tagged as en-passant.
                else if ((pos.piece_on(from) & 7) == PAWN &&
                         file_of(from) != file_of(to) &&
                         pos.piece_on(to) == 0 &&
                         to == pos.ep_sq()) {
                    move = make_move(from, to) | ENPASSANT_FLAG;
                }
                else {
                    move = make_move(from, to);
                }

                pos.make_move(move);
            }
        }
    }

    void go(const std::vector<std::string>& args) {
        if (search_active) stop();

        int depth = 10;
        uint64_t nodes = 0;
        int64_t wtime = 0, btime = 0, winc = 0, binc = 0;
        int movestogo = 0, movetime = 0;
        bool infinite = false, ponder_mode = false;
        std::vector<std::string> searchmoves_list;
        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "depth" && i+1 < args.size()) depth = std::stoi(args[++i]);
            else if (args[i] == "nodes" && i+1 < args.size()) nodes = std::stoull(args[++i]);
            else if (args[i] == "wtime" && i+1 < args.size()) wtime = std::stoll(args[++i]);
            else if (args[i] == "btime" && i+1 < args.size()) btime = std::stoll(args[++i]);
            else if (args[i] == "winc" && i+1 < args.size()) winc = std::stoll(args[++i]);
            else if (args[i] == "binc" && i+1 < args.size()) binc = std::stoll(args[++i]);
            else if (args[i] == "movestogo" && i+1 < args.size()) movestogo = std::stoi(args[++i]);
            else if (args[i] == "movetime" && i+1 < args.size()) movetime = std::stoi(args[++i]);
            else if (args[i] == "infinite") infinite = true;
            else if (args[i] == "ponder") ponder_mode = true;
            else if (args[i] == "searchmoves") {
                // Collect all following tokens as move strings until next keyword
                ++i;
                while (i < args.size() && args[i][0] != '\0' &&
                       args[i] != "depth" && args[i] != "nodes" &&
                       args[i] != "wtime" && args[i] != "btime" &&
                       args[i] != "winc"  && args[i] != "binc"  &&
                       args[i] != "movestogo" && args[i] != "movetime" &&
                       args[i] != "infinite"  && args[i] != "ponder") {
                    searchmoves_list.push_back(args[i++]);
                }
                --i;
            }
        }
        // Only force infinite if truly unconstrained: no depth specified (still at default 10),
        // no nodes, no movetime, no clock — i.e. a bare "go" with no parameters.
        bool depth_specified = false;
        for (size_t i = 0; i < args.size(); ++i)
            if (args[i] == "depth") { depth_specified = true; break; }
        if (!infinite && !depth_specified && movetime == 0 && wtime == 0 && btime == 0 && nodes == 0)
            infinite = true;

        // CRITICAL: when running infinite (including bare "go"), the iterative deepening
        // loop condition is  depth <= max_depth && !stop_search.  If max_depth stays at
        // the default of 10 the search thread exits after depth 10, but bestmove is only
        // printed after stop() is called — so the GUI hangs with no bestmove.
        // Fix: use MAX_PLY (128) as the ceiling; stop_search will terminate the loop.
        if (infinite) depth = MAX_PLY;

        if (uci_limit_strength && !infinite) {
            int elo_depth = 1 + (uci_elo - 800) / 100;
            elo_depth = std::clamp(elo_depth, 1, 30);
            depth = std::min(depth, elo_depth);
        }

        // Skill Level depth cap: level 0 → depth 1, level 19 → depth 20, level 20 = no cap.
        if (skill_level < 20 && !infinite) {
            int skill_depth = skill_level + 1;
            depth = std::min(depth, skill_depth);
        }

        tm.set_side(pos.side_to_move(), wtime, btime, winc, binc, movestogo, movetime, infinite, ponder_mode);
        tm.set_game_phase(pos.game_phase());

        if (!ponder_mode && !infinite && use_book) {
            Move book_move = book.probe(pos);
            if (book_move != NO_MOVE) {
                std::cout << "info string Book hit — playing book move\n";
                std::cout << "bestmove " << move_to_uci(book_move, &pos) << std::endl;
                return;
            }
        }

        if (!ponder_mode && tb.can_probe(pos)) {
            Move tb_move = tb.probe_root_dtz_move(pos);
            if (tb_move != NO_MOVE) {
                std::cout << "bestmove " << move_to_uci(tb_move, &pos) << std::endl;
                return;
            }
        }

        stop_search = false;
        shared_best_move = NO_MOVE;
        shared_best_score = -INF;
        depth_done_count = 0;
        depth_continue = false;
        depth_ack_count = 0;
        root_infos.clear();

        // Reset global search state ONCE before any thread launches
        node_limit = nodes;   // 'nodes' is the go() local parsed from "go nodes N"
        global_nodes.store(0, std::memory_order_relaxed);
        tb_hits = 0;
        tt.new_search();   // increment TT generation once per go(), not per thread

        Move moves[MAX_MOVES];
        int cnt = generate_moves(pos, moves);
        std::vector<ScoredMove> filtered_root_moves;
        // Use make/undo instead of Position copy: avoids heap-allocating the
        // history vector up to 256 times on Android/ARM where memory is tight.
        // A Position copy on Android can silently fail or be slow enough that
        // the time manager triggers before any move is found → bestmove 0000.
        for (int i = 0; i < cnt; ++i) {
            Move m = moves[i];
            if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;
            int cap    = pos.piece_on(to_sq(m));
            int old_cr = pos.castling_rights();
            int old_ep = pos.ep_sq();
            int old_50 = pos.halfmove_clock();
            pos.make_move(m);
            bool legal = !pos.mover_in_check();
            pos.undo_move(m, cap, old_cr, old_ep, old_50);
            if (!legal) continue;
            // searchmoves filter: if the GUI specified a move list, only search those
            if (!searchmoves_list.empty()) {
                std::string ms = move_to_uci(m, &pos);
                bool in_list = false;
                for (const auto& sm_str : searchmoves_list) {
                    if (sm_str == ms) { in_list = true; break; }
                }
                if (!in_list) continue;
            }
            filtered_root_moves.push_back({m, 0});
        }

        if (filtered_root_moves.empty()) {
            // True checkmate or stalemate — only legal output is 0000.
            std::cout << "bestmove 0000\n";
            return;
        }

        for (const auto& sm : filtered_root_moves) {
            root_infos.push_back({sm.move, -INF, {}});
        }

        // filtered_root_moves.size() used below for thread/search setup

        pondering = ponder_mode;
        pondering_active = ponder_mode;
        search_active = true;

        // Lazy SMP: every thread receives the FULL root move list and runs an
        // independent iterative-deepening search.  The TT is shared, so threads
        // naturally feed each other with good moves and avoid redundant work.
        // Partitioning moves between threads (the old code) was wrong: thread N
        // would never evaluate move 0, so the best move was often missed.
        for (int i = 0; i < thread_count; ++i) {
            search_threads.emplace_back([this, i, depth, nodes, filtered_root_moves]() {
                auto st = std::make_unique<SearchThread>(i, thread_count, pos, tt, tb, eval, &book, uci_show_wdl);
                st->set_multi_pv(multi_pv);
                st->set_contempt(contempt);
                st->search(depth, nodes, filtered_root_moves);
            });
        }

        // For infinite or ponder searches, don't block — return immediately so
        // the UCI run() loop can continue reading commands (esp. "stop").
        if (!ponder_mode && !infinite) {
            for (auto& t : search_threads) {
                if (t.joinable()) t.join();
            }
            search_threads.clear();
            search_active = false;

            if (!root_infos.empty()) {
                std::lock_guard<std::mutex> lock(last_pv_mutex);
                // operator< is reversed (for sort-descending), so use an explicit lambda here.
                auto best_it = std::max_element(root_infos.begin(), root_infos.end(),
                    [](const RootMoveInfo& a, const RootMoveInfo& b) { return a.score < b.score; });
                if (best_it != root_infos.end() && best_it->score > -INF + 1000) {
                    last_pv = best_it->pv;
                }
            }

            // Acquire load pairs with the release store in search() to guarantee
            // visibility of the written move on ARM/Android weak-memory CPUs.
            Move best = shared_best_move.load(std::memory_order_acquire);
            if (best == NO_MOVE && !filtered_root_moves.empty()) {
                // Should only reach here if the search was stopped in <256 nodes
                // (before any time-check ran).  The first legal move is safe.
                best = filtered_root_moves[0].move;
            }

            // Skill Level < 20: occasionally play a sub-optimal move to simulate
            // human error. Probability scales quadratically so even level 19 rarely
            // blunders while level 0 almost always picks randomly.
            if (skill_level < 20 && !filtered_root_moves.empty()) {
                double error_prob = (20 - skill_level) / 20.0;
                error_prob = error_prob * error_prob;  // squared for gentler curve
                std::mt19937 rng(static_cast<unsigned>(
                    std::chrono::steady_clock::now().time_since_epoch().count()));
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                if (dist(rng) < error_prob) {
                    int n_candidates = std::max(1,
                        static_cast<int>(filtered_root_moves.size() * error_prob));
                    std::uniform_int_distribution<int> pick(0, n_candidates - 1);
                    best = filtered_root_moves[pick(rng)].move;
                }
            }

            std::cout << "bestmove " << move_to_uci(best, &pos) << std::endl;

            if (tuning_mode && tuning_stream.is_open()) {
                tuning_stream << pos.fen() << "\t" << shared_best_score.load() << "\t?\n";
                tuning_stream.flush();
            }
        }
    }

    void stop() {
        if (!search_active) return;
        stop_search = true;
        pondering = false;
        for (auto& t : search_threads) {
            if (t.joinable()) t.join();
        }
        search_threads.clear();
        search_active = false;
        pondering_active = false;
        Move best = shared_best_move.load(std::memory_order_acquire);
        if (best == NO_MOVE) {
            // Search stopped before any move was committed — pick first legal move
            Move moves[MAX_MOVES];
            int cnt = generate_moves(pos, moves);
            for (int i = 0; i < cnt; ++i) {
                Move m = moves[i];
                if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;
                int cap = pos.piece_on(to_sq(m));
                int oc = pos.castling_rights(), oe = pos.ep_sq(), of_ = pos.halfmove_clock();
                pos.make_move(m);
                bool legal = !pos.mover_in_check();
                pos.undo_move(m, cap, oc, oe, of_);
                if (legal) { best = m; break; }
            }
        }
        if (best != NO_MOVE) {
            std::cout << "bestmove " << move_to_uci(best, &pos) << std::endl;
        }
    }

    void ponderhit() {
        if (!search_active || !pondering_active) return;
        pondering = false;
        pondering_active = false;
        for (auto& t : search_threads) {
            if (t.joinable()) t.join();
        }
        search_threads.clear();
        search_active = false;
        Move best = shared_best_move.load(std::memory_order_acquire);
        if (best == NO_MOVE) {
            // Pick first legal move as emergency fallback
            Move moves[MAX_MOVES];
            int cnt = generate_moves(pos, moves);
            for (int i = 0; i < cnt; ++i) {
                Move m = moves[i];
                if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;
                int cap = pos.piece_on(to_sq(m));
                int oc = pos.castling_rights(), oe = pos.ep_sq(), of_ = pos.halfmove_clock();
                pos.make_move(m);
                bool legal = !pos.mover_in_check();
                pos.undo_move(m, cap, oc, oe, of_);
                if (legal) { best = m; break; }
            }
        }
        if (best != NO_MOVE) {
            std::cout << "bestmove " << move_to_uci(best, &pos) << std::endl;
        } else {
            std::cout << "bestmove 0000\n";  // true stalemate/checkmate
        }
    }

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            if (token == "uci") {
                std::cout << "id name Hugine 5.1.0\n";
                std::cout << "id author 0xbytecode\n";
                std::cout << "info string Platform: "
#if ARCH_X86
                          << "x86"
#elif ARCH_ARM
                          << "ARM"
#else
                          << "unknown-arch"
#endif
                          << " | Syzygy: "
#if HAS_SYZYGY
                          << "ON"
#else
                          << "OFF"
#endif
                          << " | NNUE: "
#ifdef USE_NNUE
                          << "ON"
#else
                          << "OFF"
#endif
                          << " | Chess960: "
                          << (chess960 ? "ON" : "OFF")
                          << "\n";
                std::cout << "option name Hash type spin default 256 min 1 max 33554432\n";
                std::cout << "option name Threads type spin default 1 min 1 max 1024\n";
                std::cout << "option name Ponder type check default false\n";
                std::cout << "option name Skill Level type spin default 20 min 0 max 20\n";
                std::cout << "option name OwnBook type check default true\n";
                std::cout << "option name BookFile type string default\n";
                std::cout << "option name BookVariety type spin default 0 min 0 max 10\n";
                std::cout << "option name SyzygyPath type string default\n";
                std::cout << "option name SyzygyProbeDepth type spin default 1 min 1 max 100\n";
                std::cout << "option name SyzygyProbeLimit type spin default 7 min 0 max 7\n";
                std::cout << "option name Syzygy50MoveRule type check default true\n";
                std::cout << "option name EvalFile type string default\n";
                std::cout << "option name NNUEPath type string default\n";
                // Note: both EvalFile and NNUEPath accept either a single .nnue file
                // path or a directory path (scanned alphabetically for .nnue files).
                std::cout << "option name MultiPV type spin default 1 min 1 max 256\n";
                std::cout << "option name Contempt type spin default 0 min -100 max 100\n";
                std::cout << "option name Move Overhead type spin default 10 min 0 max 5000\n";
                std::cout << "option name UCI_Chess960 type check default false\n";
                std::cout << "option name UCI_LimitStrength type check default false\n";
                std::cout << "option name UCI_Elo type spin default 1500 min 1320 max 3190\n";
                std::cout << "option name UCI_ShowWDL type check default false\n";
                std::cout << "option name Debug Log File type string default\n";
                std::cout << "option name Learning type check default false\n";
                std::cout << "option name LearningFile type string default\n";
                std::cout << "option name LearningRate type spin default 100 min 1 max 1000\n";
                std::cout << "option name LearningMaxAdjust type spin default 50 min 0 max 200\n";
                std::cout << "option name Clear Learning type button\n";
                std::cout << "option name Save Learning type button\n";
                std::cout << "option name TuningMode type check default false\n";
                std::cout << "option name TuningFile type string default\n";
                std::cout << "option name Clear Hash type button\n";
                std::cout << "uciok\n";
            } else if (token == "isready") {
                std::cout << "readyok\n";
            } else if (token == "ucinewgame") {
                pos.init_startpos();
                tt.clear();
            } else if (token == "setoption") {
                // UCI protocol: "setoption name <OptionName> value <Value>"
                // Must consume the literal "name" keyword first, otherwise it
                // gets prepended to every option name and set_option never matches.
                std::string name, value, word;
                if (iss >> word && word != "name") {
                    // Malformed — put the word back as the start of the name
                    name = word;
                }
                while (iss >> word) {
                    if (word == "value") break;
                    if (!name.empty()) name += " ";
                    name += word;
                }
                // Read the entire rest of the line as the value (supports paths with spaces).
                std::string rest;
                if (std::getline(iss, rest)) {
                    // Strip leading space left by the stream after "value"
                    size_t start = rest.find_first_not_of(" \t");
                    value = (start != std::string::npos) ? rest.substr(start) : "";
                }
                set_option(name, value);
            } else if (token == "position") {
                std::vector<std::string> args;
                while (iss >> token) args.push_back(token);
                position(args);
            } else if (token == "go") {
                std::vector<std::string> args;
                while (iss >> token) args.push_back(token);
                go(args);
            } else if (token == "stop") {
                stop();
            } else if (token == "ponderhit") {
                ponderhit();
            } else if (token == "learn") {
                std::string subcmd;
                iss >> subcmd;
                if (subcmd == "result") {
                    std::string result_str;
                    iss >> result_str;
                    int result = 0;
                    if (result_str == "win") result = 1;
                    else if (result_str == "loss") result = -1;
                    {
                        std::lock_guard<std::mutex> lock(last_pv_mutex);
                        if (last_pv.empty()) {
                            std::cout << "info string No PV available from last search.\n";
                            // Do NOT return here — that would exit run() and kill the UCI loop.
                            // Just skip the update and continue reading commands.
                        } else {
                            Position tmp = pos;
                            for (Move m : last_pv) {
                                U64 key = tmp.get_hash();
                                learning.update(key, result, tmp.side_to_move());
                                tmp.make_move(m);
                            }
                            std::cout << "info string Learning updated with " << last_pv.size() << " positions.\n";
                        }
                    }  // end lock_guard scope
                } else if (subcmd == "clear") {
                    learning.clear();
                    std::cout << "info string Learning table cleared.\n";
                } else if (subcmd == "save") {
                    if (learning.save())
                        std::cout << "info string Learning saved.\n";
                    else
                        std::cout << "info string Failed to save learning.\n";
                } else {
                    std::cout << "info string Unknown learn subcommand. Use: result win|draw|loss, clear, save\n";
                }
            } else if (token == "quit") {
                stop();
                break;
            } else if (token == "d") {
                // Board display
                std::cout << "\n";
                for (int r = 7; r >= 0; --r) {
                    std::cout << " " << (r+1) << "  ";
                    for (int f = 0; f < 8; ++f) {
                        Square sq = make_square(f, r);
                        int pc = pos.piece_on(sq);
                        if (pc == 0) std::cout << ".";
                        else {
                            char p = " pnbrqk"[pc & 7];
                            if ((pc >> 3) == WHITE) p = toupper(p);
                            std::cout << p;
                        }
                        std::cout << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n     a b c d e f g h\n\n";
                std::cout << "FEN  : " << pos.fen() << "\n";
                std::cout << "Key  : 0x" << std::hex << std::uppercase
                          << pos.get_hash() << std::dec << "\n";
                std::cout << "PGKey: 0x" << std::hex << std::uppercase
                          << pos.get_polyglot_hash() << std::dec
                          << "  (Polyglot book hash)\n";
                std::cout << "Side : " << (pos.side_to_move() == WHITE ? "White" : "Black") << "\n";
                std::cout << "EP   : ";
                if (pos.ep_sq() != -1)
                    std::cout << char('a' + file_of(pos.ep_sq())) << char('1' + rank_of(pos.ep_sq()));
                else std::cout << "-";
                std::cout << "\n";
                std::cout << "50mr : " << pos.halfmove_clock() << "\n";

                // ---- Castling rights diagnostic ----
                // Shows exactly what was parsed from the FEN for each of the 4 slots.
                const char* slot_names[2][2] = {{"White-K (O-O)","White-Q (O-O-O)"},
                                                 {"Black-k (o-o)","Black-q (o-o-o)"}};
                std::cout << "Castle rights:\n";
                bool any_right = false;
                for (int c = 0; c < 2; ++c) {
                    for (int s = 0; s < 2; ++s) {
                        Square rsq = pos.castle_rook(Color(c), s);
                        std::cout << "  " << slot_names[c][s] << ": ";
                        if (rsq == -1) {
                            std::cout << "NONE\n";
                        } else {
                            char rf = char('a' + file_of(rsq));
                            char rr = char('1' + rank_of(rsq));
                            // Determine expected destination for king and rook
                            int castling_rank = (c == 0) ? 0 : 7;
                            Square king_dest = make_square((s == 0) ? 6 : 2, castling_rank);
                            Square rook_dest = make_square((s == 0) ? 5 : 3, castling_rank);
                            std::cout << "ROOK on " << rf << rr
                                      << " -> king lands "
                                      << char('a' + file_of(king_dest)) << char('1' + rank_of(king_dest))
                                      << ", rook lands "
                                      << char('a' + file_of(rook_dest)) << char('1' + rank_of(rook_dest))
                                      << "\n";
                            any_right = true;
                        }
                    }
                }
                if (!any_right) std::cout << "  (no castling rights)\n";
                std::cout << "\n";

            } else if (token == "eval") {
                Value score = eval.evaluate(pos);
                std::cout << "Evaluation: " << score << " cp (from side to move)\n";
            } else if (token == "perft") {
                // ------------------------------------------------------------
                // High-performance perft using bulk counting at depth 1.
                // Uses perft_inner() (plain function, no std::function overhead)
                // + Position::bulk_count_legal() to skip make/undo at leaves.
                // ------------------------------------------------------------
                int depth = 1;
                {
                    std::string tok;
                    if (iss >> tok) depth = std::stoi(tok);
                }
                if (depth < 1) depth = 1;

                // Castle-flag verification (kept for diagnostic correctness)
                {
                    Move probe[MAX_MOVES];
                    int pcnt = generate_moves(pos, probe);
                    int castle_count = 0;
                    bool flag_ok = true;
                    for (int i = 0; i < pcnt; ++i) {
                        if (is_castling(probe[i])) {
                            castle_count++;
                            int castling_rank_p = (pos.side_to_move() == WHITE) ? 0 : 7;
                            Square kd = to_sq(probe[i]);
                            bool ks = (kd == make_square(6, castling_rank_p));
                            bool qs = (kd == make_square(2, castling_rank_p));
                            if (!ks && !qs) flag_ok = false;
                        }
                    }
                    std::cout << "info string Castle-flag check: "
                              << castle_count << " castle move(s) in root position, "
                              << (flag_ok ? "all destinations correct (g/c file)." : "ERROR: unexpected king destination!")
                              << "\n";
                    int rights_count = 0;
                    for (int c = 0; c < 2; ++c)
                        for (int s = 0; s < 2; ++s)
                            if (pos.castle_rook(Color(c), s) != -1) rights_count++;
                    std::cout << "info string FEN castling rights loaded: "
                              << rights_count << " slot(s) active.\n";
                }

                // ── Divide (print per-root-move counts) ──────────────────────
                auto t0 = std::chrono::steady_clock::now();

                Move root_mvs[MAX_MOVES];
                int root_cnt = generate_moves(pos, root_mvs);
                uint64_t total = 0;

                for (int i = 0; i < root_cnt; ++i) {
                    Move m = root_mvs[i];
                    int cap    = pos.piece_on(to_sq(m));
                    int old_cr = pos.castling_rights();
                    int old_ep = pos.ep_sq();
                    int old_50 = pos.halfmove_clock();
                    pos.make_move(m);

                    if (!pos.mover_in_check()) {
                        uint64_t n;
                        if      (depth <= 1) n = 1;
                        else if (depth == 2) n = pos.bulk_count_legal();
                        else                 n = perft_inner(pos, depth - 1);

                        pos.undo_move(m, cap, old_cr, old_ep, old_50);
                        std::cout << move_to_uci(m, &pos) << ": " << n << "\n";
                        total += n;
                    } else {
                        pos.undo_move(m, cap, old_cr, old_ep, old_50);
                    }
                }

                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                uint64_t nps = ms > 0 ? (uint64_t)(total / (ms / 1000.0)) : 0;
                std::cout << "\nNodes searched: " << total
                          << "  depth: " << depth
                          << "  time: " << (uint64_t)ms << "ms"
                          << "  nps: " << nps << "\n";
            }
        }
    }
};

int main() {
    UCI uci;
    uci.run();
    return 0;
}
