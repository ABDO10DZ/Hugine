/*
 * Hugine 2.0 – UCI chess engine
 * Author: 0xbytecode
 * Features: all 24 advanced features, full Syzygy DTZ, learning, NNUE, YBWC, Chess960, etc.
 * Compile: g++ -O2 -std=c++17 -pthread -march=native hugine.cpp -o hugine
 * Options: -DUSE_NNUE, -DDEBUG, -DNO_SYZYGY
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

// Syzygy availability
#if defined(USE_SYZYGY)
    #define HAS_SYZYGY 1
#elif defined(NO_SYZYGY)
    #define HAS_SYZYGY 0
#elif ARCH_X86 && !OS_ANDROID
    #if defined(__has_include) && __has_include("fathom/src/tbprobe.h")
        #define HAS_SYZYGY 1
    #else
        #define HAS_SYZYGY 0
    #endif
#else
    #define HAS_SYZYGY 0
#endif

#if HAS_SYZYGY
extern "C" {
#include "fathom/src/tbprobe.h"
}
#else
#define TB_RESULT_FAILED 0xFFFFFFFF
#define TB_WIN 2
#define TB_LOSS 0
#define TB_DRAW 1
#define TB_CURSED_WIN 3
#define TB_BLESSED_LOSS (-1)
#define TB_PAWN 1
#define TB_KNIGHT 2
#define TB_BISHOP 3
#define TB_ROOK 4
#define TB_QUEEN 5
#define TB_KING 6
#define TB_SIDEMASK 0x40
#define TB_MAX_MOVES 256
inline bool tb_init(const char*) { return false; }
inline void tb_free() {}
inline int tb_max_cardinality() { return 0; }
inline unsigned tb_probe_wdl(unsigned*,unsigned*,int,int,int,int,int,int,int,int) { return TB_RESULT_FAILED; }
inline unsigned tb_probe_root_dtz(unsigned*,unsigned*,int,int,int,int,int,int,int,int,int*) { return TB_RESULT_FAILED; }
inline unsigned* tb_probe_root(unsigned*,unsigned*,int,int,int,int,int,int,int,int,void*) { return nullptr; }
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
constexpr int MATE_OFFSET = 20000;
constexpr int ASPIRATION_WINDOW = 15;
constexpr int ASPIRATION_WIDEN = 50;
constexpr int RAZOR_MARGIN_D1 = 300;
constexpr int RAZOR_MARGIN_D2 = 400;
constexpr int RAZOR_MARGIN_D3 = 600;
constexpr int FUTILITY_MARGIN_FACTOR = 200;
constexpr int LMR_BASE = 1;
constexpr int LMR_DIV = 2;
constexpr int NULL_MOVE_R = 2;
constexpr int IID_DEPTH = 5;
constexpr int IID_REDUCTION = 2;
constexpr int SEE_QUIET_MARGIN = -80;
constexpr int SINGULAR_EXTENSION_DEPTH = 8;
constexpr int SINGULAR_MARGIN = 50;
constexpr int MAX_THREADS = 64;
constexpr int MAX_HISTORY = 16384;
constexpr int PROBCUT_DEPTH = 5;
constexpr int PROBCUT_MARGIN_BASE = 100;
constexpr int PROBCUT_MARGIN_PER_DEPTH = 20;
constexpr int LMP_BASE = 3;
constexpr int LMP_FACTOR = 2;

constexpr int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 0};
constexpr int PHASE_KNIGHT = 1;
constexpr int PHASE_BISHOP = 1;
constexpr int PHASE_ROOK = 2;
constexpr int PHASE_QUEEN = 4;
constexpr int TOTAL_PHASE = 24;

constexpr size_t LEARNING_TABLE_SIZE = 1 << 20; // 1,048,576 entries
constexpr int LEARNING_MAX_ADJUST = 50;

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

inline int popcount(U64 b) { return __builtin_popcountll(b); }
inline Square lsb(U64 b) { assert(b != 0 && "lsb called on empty bitboard"); return Square(__builtin_ctzll(b)); }
inline Square pop_lsb(U64& b) { Square s = lsb(b); b &= b - 1; return s; }

// Magic bitboards
struct Magic {
    U64 mask;
    U64 magic;
    U64* attacks;
    int shift;
};

extern Magic rook_magics[64];
extern Magic bishop_magics[64];
extern U64 rook_attacks_table[102400];
extern U64 bishop_attacks_table[102400];
// Classical attack functions defined below (no longer using magic lookup tables)
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
    U64* rook_ptr = rook_attacks_table;
    U64* bishop_ptr = bishop_attacks_table;

    for (int sq = 0; sq < 64; ++sq) {
        U64 mask = rook_mask(sq);
        int shift = rook_shifts[sq];
        U64 magic = rook_magic_numbers[sq];
        int num_occ = 1 << (64 - shift);
        rook_magics[sq].mask = mask;
        rook_magics[sq].magic = magic;
        rook_magics[sq].shift = shift;
        rook_magics[sq].attacks = rook_ptr;

        for (int occ_index = 0; occ_index < num_occ; ++occ_index) {
            U64 occ = 0;
            U64 bits = occ_index;
            U64 m = mask;
            while (m) {
                Square bit = pop_lsb(m);
                if (bits & 1) occ |= 1ULL << bit;
                bits >>= 1;
            }
            U64 attacks = 0;
            int f = file_of(sq), r = rank_of(sq);
            for (int rr = r+1; rr < 8; ++rr) {
                Square s2 = make_square(f, rr);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            for (int rr = r-1; rr >= 0; --rr) {
                Square s2 = make_square(f, rr);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            for (int ff = f+1; ff < 8; ++ff) {
                Square s2 = make_square(ff, r);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            for (int ff = f-1; ff >= 0; --ff) {
                Square s2 = make_square(ff, r);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            U64 idx = (occ * magic) >> shift;
            rook_ptr[idx] = attacks;
        }
        rook_ptr += num_occ;

        mask = bishop_mask(sq);
        shift = bishop_shifts[sq];
        magic = bishop_magic_numbers[sq];
        num_occ = 1 << (64 - shift);
        bishop_magics[sq].mask = mask;
        bishop_magics[sq].magic = magic;
        bishop_magics[sq].shift = shift;
        bishop_magics[sq].attacks = bishop_ptr;

        for (int occ_index = 0; occ_index < num_occ; ++occ_index) {
            U64 occ = 0;
            U64 bits = occ_index;
            U64 m = mask;
            while (m) {
                Square bit = pop_lsb(m);
                if (bits & 1) occ |= 1ULL << bit;
                bits >>= 1;
            }
            U64 attacks = 0;
            int f = file_of(sq), r = rank_of(sq);
            for (int i = 1; f+i < 8 && r+i < 8; ++i) {
                Square s2 = make_square(f+i, r+i);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            for (int i = 1; f-i >= 0 && r+i < 8; ++i) {
                Square s2 = make_square(f-i, r+i);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            for (int i = 1; f+i < 8 && r-i >= 0; ++i) {
                Square s2 = make_square(f+i, r-i);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            for (int i = 1; f-i >= 0 && r-i >= 0; ++i) {
                Square s2 = make_square(f-i, r-i);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            U64 idx = (occ * magic) >> shift;
            bishop_ptr[idx] = attacks;
        }
        bishop_ptr += num_occ;
    }
}

// Classical sliding attack generators — provably correct, no magic number collisions possible.
// Magic tables are still initialised (for potential future use) but attacks are computed directly.
U64 rook_attacks_magic(Square s, U64 occ) {
    U64 attacks = 0;
    int f = file_of(s), r = rank_of(s);
    for (int rr=r+1; rr<8; rr++) { U64 b=1ULL<<make_square(f,rr); attacks|=b; if(occ&b) break; }
    for (int rr=r-1; rr>=0; rr--) { U64 b=1ULL<<make_square(f,rr); attacks|=b; if(occ&b) break; }
    for (int ff=f+1; ff<8; ff++) { U64 b=1ULL<<make_square(ff,r); attacks|=b; if(occ&b) break; }
    for (int ff=f-1; ff>=0; ff--) { U64 b=1ULL<<make_square(ff,r); attacks|=b; if(occ&b) break; }
    return attacks;
}

U64 bishop_attacks_magic(Square s, U64 occ) {
    U64 attacks = 0;
    int f = file_of(s), r = rank_of(s);
    for (int i=1; f+i<8&&r+i<8; i++) { U64 b=1ULL<<make_square(f+i,r+i); attacks|=b; if(occ&b) break; }
    for (int i=1; f-i>=0&&r+i<8; i++) { U64 b=1ULL<<make_square(f-i,r+i); attacks|=b; if(occ&b) break; }
    for (int i=1; f+i<8&&r-i>=0; i++) { U64 b=1ULL<<make_square(f+i,r-i); attacks|=b; if(occ&b) break; }
    for (int i=1; f-i>=0&&r-i>=0; i++) { U64 b=1ULL<<make_square(f-i,r-i); attacks|=b; if(occ&b) break; }
    return attacks;
}

U64 queen_attacks_magic(Square s, U64 occ) {
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
    bool is_repetition(int count) const {
        int c = 0;
        for (int i = (int)history.size() - 2; i >= 0 && c < count; i -= 2) {
            if (history[i] == hash) c++;
            if (c >= count) return true;
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
    bool gives_check(Move m) const {
        Position copy = *this;
        copy.make_move(m);
        return copy.is_check();
    }

    void make_move(Move m) {
        if (m == NULL_MOVE) {
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
    static constexpr int FT_INPUTS = 40960;   // 2 * 64 * 64 * 5
    static constexpr int FT_SIZE = 256;
    static constexpr int L1_SIZE = 32;
    static constexpr int L2_SIZE = 32;
    static constexpr int FT_SCALE = 128;
    static constexpr int HIDDEN_SCALE = 64;

private:
    struct Layer {
        std::vector<int8_t> weights;
        std::vector<int16_t> bias;
    };
    Layer ft, l1, l2, output;
    int16_t output_bias;

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

    void add_piece(Accumulator& acc, Square king_sq, Color piece_color, Square piece_sq, PieceType pt, int delta) {
        int idx = feature_index(acc.king_color, king_sq, piece_color, piece_sq, pt);
        if (idx == -1) return;
        int8_t* w = &ft.weights[idx * FT_SIZE];
        for (int i = 0; i < FT_SIZE; ++i) acc.values[i] += delta * w[i] * FT_SCALE;
    }

    void recompute_accumulator(Accumulator& acc, const Position& pos, Color perspective_king_color) {
        Square king_sq = pos.king_square(perspective_king_color);
        acc.king_sq = king_sq;
        acc.king_color = perspective_king_color;
        acc.values.assign(FT_SIZE, 0);
        for (int i = 0; i < FT_SIZE; ++i) acc.values[i] = ft.bias[i];
        for (Color c : {WHITE, BLACK}) {
            for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
                U64 bb = pos.bb(c, pt);
                while (bb) {
                    Square sq = pop_lsb(bb);
                    int idx = feature_index(perspective_king_color, king_sq, c, sq, pt);
                    if (idx != -1) {
                        int8_t* w = &ft.weights[idx * FT_SIZE];
                        for (int i = 0; i < FT_SIZE; ++i) acc.values[i] += w[i] * FT_SCALE;
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
        if (magic != 0x5A5A5A5A || version != 2 || ft_inputs != FT_INPUTS || ft_size != FT_SIZE || l1_size != L1_SIZE || l2_size != L2_SIZE || out_dim != 1)
            return false;
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
        return true;
    }

    void push() {
        auto& s0 = tls.stack[0], &s1 = tls.stack[1];
        if (s0.empty()) { s0.emplace_back(); s1.emplace_back(); }
        else { s0.push_back(s0.back()); s1.push_back(s1.back()); }
    }

    void pop() { tls.stack[0].pop_back(); tls.stack[1].pop_back(); }

    void make_move(const Position& pos, Move m, Color us, PieceType moving_pt, PieceType captured_pt,
                   bool was_promotion, PieceType prom_pt = NO_PIECE) {
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
        auto& acc0 = tls.stack[0].back();
        if (!acc0.computed) recompute_accumulator(acc0, pos, WHITE);
        alignas(32) int16_t l0[FT_SIZE];
        for (int i = 0; i < FT_SIZE; ++i) {
            int16_t x = acc0.values[i];
            l0[i] = std::max(0, std::min(127, (int)x));
        }
        alignas(32) int16_t l1_out[L1_SIZE] = {0};
#ifdef USE_AVX2
        for (int i = 0; i < L1_SIZE; ++i) {
            __m256i sum = _mm256_setzero_si256();
            for (int j = 0; j < FT_SIZE; j += 16) {
                __m256i l0_vec = _mm256_load_si256((__m256i*)&l0[j]);
                __m128i w8 = _mm_load_si128((__m128i*)&l1.weights[j * L1_SIZE + i]);
                __m256i w16 = _mm256_cvtepi8_epi16(w8);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(l0_vec, w16));
            }
            __m128i s_high = _mm256_extracti128_si256(sum, 1);
            __m128i s_low = _mm256_castsi256_si128(sum);
            __m128i s128 = _mm_add_epi32(s_low, s_high);
            s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(1,0,3,2)));
            s128 = _mm_add_epi32(s128, _mm_shufflelo_epi16(s128, _MM_SHUFFLE(1,0,3,2)));
            int32_t total = _mm_cvtsi128_si32(s128) + l1.bias[i];
            total = (total * HIDDEN_SCALE) >> 8;
            l1_out[i] = std::max(0, std::min(127, total));
        }
#else
        for (int i = 0; i < L1_SIZE; ++i) {
            int32_t sum = l1.bias[i];
            for (int j = 0; j < FT_SIZE; ++j) sum += l0[j] * l1.weights[j * L1_SIZE + i];
            sum = (sum * HIDDEN_SCALE) >> 8;
            l1_out[i] = std::max(0, std::min(127, sum));
        }
#endif
        alignas(32) int16_t l2_out[L2_SIZE] = {0};
        for (int i = 0; i < L2_SIZE; ++i) {
            int32_t sum = l2.bias[i];
            for (int j = 0; j < L1_SIZE; ++j) sum += l1_out[j] * l2.weights[j * L2_SIZE + i];
            sum = (sum * HIDDEN_SCALE) >> 8;
            l2_out[i] = std::max(0, std::min(127, sum));
        }
        int32_t out = output_bias;
        for (int i = 0; i < L2_SIZE; ++i) out += l2_out[i] * output.weights[i];
        out = (out * HIDDEN_SCALE) >> 8;
        Value score = out / 16;
        return (pos.side_to_move() == WHITE) ? score : -score;
    }
};

thread_local NNUEEvaluator::ThreadData NNUEEvaluator::tls;
#endif

// ----------------------------------------------------------------------------
// Classical evaluation (with all advanced terms)
// ----------------------------------------------------------------------------
class Evaluation {
private:
#ifdef USE_NNUE
    NNUEEvaluator nnue;
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
    void set_nnue(const std::string& file) { nnue.load(file); }
    NNUEEvaluator& get_nnue() { return nnue; }
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
                    U64 diag = bishop_attacks_magic(sq, 0);
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
            int nn = nnue.evaluate(pos);
            return Value(nnue_weight * nn + (1.0 - nnue_weight) * score);
        }
#endif
        return (pos.side_to_move() == WHITE) ? score : -score;
    }
};

// ----------------------------------------------------------------------------
// Transposition Table (with DTZ)
// ----------------------------------------------------------------------------
struct TTEntry {
    U64 key;
    Depth depth;
    Value score;
    Bound bound;
    Move move;
    int age;
    int dtz;           // DTZ value (0 = unknown, positive = winning distance, negative = losing)
};

class TranspositionTable {
private:
    std::vector<TTEntry> table;
    size_t size;
    int age;
    mutable std::shared_mutex mtx;
public:
    TranspositionTable(size_t mb) : age(0) {
        size = mb * 1024 * 1024 / sizeof(TTEntry);
        table.resize(size);
    }
    void resize(size_t mb) {
        std::unique_lock lock(mtx);
        size = mb * 1024 * 1024 / sizeof(TTEntry);
        table.clear();
        table.resize(size);
        age = 0;
    }
    void clear() {
        std::unique_lock lock(mtx);
        std::fill(table.begin(), table.end(), TTEntry{});
        age = 0;
    }
    void new_search() { age++; }
    void store(U64 key, Depth depth, Value score, Bound bound, Move move, int dtz = 0) {
        std::unique_lock lock(mtx);
        size_t idx = key % size;
        TTEntry& e = table[idx];
        if (e.key == key && e.depth > depth) return;
        e = {key, depth, score, bound, move, age, dtz};
    }
    bool probe(U64 key, Depth depth, Value alpha, Value beta, Value& score, Move& move, int& dtz) {
        std::shared_lock lock(mtx);
        size_t idx = key % size;
        TTEntry& e = table[idx];
        if (e.key != key) return false;
        move = e.move;
        dtz  = e.dtz;
        // Always expose the stored score (even on depth-miss) so the caller can
        // use it for singular extension heuristics. The caller decides whether
        // to trust it based on the return value.
        score = e.score;
        if (e.depth >= depth) {
            if (e.bound == BOUND_EXACT) { return true; }
            if (e.bound == BOUND_LOWER && e.score >= beta)  { return true; }
            if (e.bound == BOUND_UPPER && e.score <= alpha) { return true; }
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
        if (!file) return false;
        entries.clear();
        Entry e;
        while (file.read((char*)&e, sizeof(e))) entries.push_back(e);
        loaded = true;
        return true;
    }
    void set_variety(double v) { variety = v; }
    Move probe(const Position& pos) {
        if (!loaded) return NO_MOVE;
        U64 key = pos.get_hash();
        std::vector<Entry> matches;
        for (const auto& e : entries) if (e.key == key) matches.push_back(e);
        if (matches.empty()) return NO_MOVE;
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
// Syzygy Tablebase wrapper (with full DTZ support)
// ----------------------------------------------------------------------------
class SyzygyTablebase {
private:
    bool initialized;
    int max_pieces;
public:
    SyzygyTablebase() : initialized(false), max_pieces(0) {}
    ~SyzygyTablebase() { if (initialized) tb_free(); }
    bool init(const std::string& path) {
        if (!tb_init(path.c_str())) return false;
        initialized = true;
        max_pieces = tb_max_cardinality();
        return true;
    }
    bool can_probe(const Position& pos) const {
        return initialized && popcount(pos.occupied_bb()) <= max_pieces;
    }
    int probe_wdl(const Position& pos) {
        if (!can_probe(pos)) return TB_RESULT_FAILED;
        unsigned pieces[32], squares[32];
        int cnt = 0;
        for (Square sq = 0; sq < 64; ++sq) {
            int pc = pos.piece_on(sq);
            if (pc == 0) continue;
            Color c = Color(pc >> 3);
            PieceType pt = PieceType(pc & 7);
            int code;
            switch (pt) {
                case PAWN:   code = (c == WHITE) ? TB_PAWN : TB_PAWN | TB_SIDEMASK; break;
                case KNIGHT: code = (c == WHITE) ? TB_KNIGHT : TB_KNIGHT | TB_SIDEMASK; break;
                case BISHOP: code = (c == WHITE) ? TB_BISHOP : TB_BISHOP | TB_SIDEMASK; break;
                case ROOK:   code = (c == WHITE) ? TB_ROOK : TB_ROOK | TB_SIDEMASK; break;
                case QUEEN:  code = (c == WHITE) ? TB_QUEEN : TB_QUEEN | TB_SIDEMASK; break;
                case KING:   code = (c == WHITE) ? TB_KING : TB_KING | TB_SIDEMASK; break;
                default: continue;
            }
            pieces[cnt] = code;
            squares[cnt] = sq;
            cnt++;
        }
        for (int i = 0; i < cnt; ++i)
            for (int j = i+1; j < cnt; ++j)
                if (pieces[j] < pieces[i]) { std::swap(pieces[i], pieces[j]); std::swap(squares[i], squares[j]); }
        return tb_probe_wdl(pieces, squares, cnt,
            (pos.castle_rook(WHITE,0) != -1) ? 1 : 0,
            (pos.castle_rook(WHITE,1) != -1) ? 1 : 0,
            (pos.castle_rook(BLACK,0) != -1) ? 1 : 0,
            (pos.castle_rook(BLACK,1) != -1) ? 1 : 0,
            pos.ep_sq() != -1 ? file_of(pos.ep_sq()) : 0,
            pos.halfmove_clock(),
            (pos.side_to_move() == WHITE) ? 0 : 1);
    }
    int probe_dtz(const Position& pos, int& success) {
        if (!can_probe(pos)) { success = 0; return 0; }
        unsigned pieces[32], squares[32];
        int cnt = 0;
        for (Square sq = 0; sq < 64; ++sq) {
            int pc = pos.piece_on(sq);
            if (pc == 0) continue;
            Color c = Color(pc >> 3);
            PieceType pt = PieceType(pc & 7);
            int code;
            switch (pt) {
                case PAWN:   code = (c == WHITE) ? TB_PAWN : TB_PAWN | TB_SIDEMASK; break;
                case KNIGHT: code = (c == WHITE) ? TB_KNIGHT : TB_KNIGHT | TB_SIDEMASK; break;
                case BISHOP: code = (c == WHITE) ? TB_BISHOP : TB_BISHOP | TB_SIDEMASK; break;
                case ROOK:   code = (c == WHITE) ? TB_ROOK : TB_ROOK | TB_SIDEMASK; break;
                case QUEEN:  code = (c == WHITE) ? TB_QUEEN : TB_QUEEN | TB_SIDEMASK; break;
                case KING:   code = (c == WHITE) ? TB_KING : TB_KING | TB_SIDEMASK; break;
                default: continue;
            }
            pieces[cnt] = code;
            squares[cnt] = sq;
            cnt++;
        }
        for (int i = 0; i < cnt; ++i)
            for (int j = i+1; j < cnt; ++j)
                if (pieces[j] < pieces[i]) { std::swap(pieces[i], pieces[j]); std::swap(squares[i], squares[j]); }

        unsigned res = tb_probe_root_dtz(pieces, squares, cnt,
            (pos.castle_rook(WHITE,0) != -1) ? 1 : 0,
            (pos.castle_rook(WHITE,1) != -1) ? 1 : 0,
            (pos.castle_rook(BLACK,0) != -1) ? 1 : 0,
            (pos.castle_rook(BLACK,1) != -1) ? 1 : 0,
            pos.ep_sq() != -1 ? file_of(pos.ep_sq()) : 0,
            pos.halfmove_clock(),
            (pos.side_to_move() == WHITE) ? 0 : 1,
            &success);
        return success ? (res & 0xFFFF) : 0;
    }
    Move probe_root_dtz_move(const Position& pos) {
        if (!can_probe(pos)) return NO_MOVE;
        unsigned pieces[32], squares[32];
        int cnt = 0;
        for (Square sq = 0; sq < 64; ++sq) {
            int pc = pos.piece_on(sq);
            if (pc == 0) continue;
            Color c = Color(pc >> 3);
            PieceType pt = PieceType(pc & 7);
            int code;
            switch (pt) {
                case PAWN:   code = (c == WHITE) ? TB_PAWN : TB_PAWN | TB_SIDEMASK; break;
                case KNIGHT: code = (c == WHITE) ? TB_KNIGHT : TB_KNIGHT | TB_SIDEMASK; break;
                case BISHOP: code = (c == WHITE) ? TB_BISHOP : TB_BISHOP | TB_SIDEMASK; break;
                case ROOK:   code = (c == WHITE) ? TB_ROOK : TB_ROOK | TB_SIDEMASK; break;
                case QUEEN:  code = (c == WHITE) ? TB_QUEEN : TB_QUEEN | TB_SIDEMASK; break;
                case KING:   code = (c == WHITE) ? TB_KING : TB_KING | TB_SIDEMASK; break;
                default: continue;
            }
            pieces[cnt] = code;
            squares[cnt] = sq;
            cnt++;
        }
        for (int i = 0; i < cnt; ++i)
            for (int j = i+1; j < cnt; ++j)
                if (pieces[j] < pieces[i]) { std::swap(pieces[i], pieces[j]); std::swap(squares[i], squares[j]); }

        int success;
        unsigned res = tb_probe_root_dtz(pieces, squares, cnt,
            (pos.castle_rook(WHITE,0) != -1) ? 1 : 0,
            (pos.castle_rook(WHITE,1) != -1) ? 1 : 0,
            (pos.castle_rook(BLACK,0) != -1) ? 1 : 0,
            (pos.castle_rook(BLACK,1) != -1) ? 1 : 0,
            pos.ep_sq() != -1 ? file_of(pos.ep_sq()) : 0,
            pos.halfmove_clock(),
            (pos.side_to_move() == WHITE) ? 0 : 1,
            &success);
        if (!success) return NO_MOVE;
        unsigned pg_move = (res >> 8) & 0xFFFF;
        int f_from = pg_move & 7, r_from = (pg_move >> 3) & 7;
        int f_to = (pg_move >> 6) & 7, r_to = (pg_move >> 9) & 7;
        int prom = (pg_move >> 12) & 7;
        Square from = make_square(f_from, r_from), to = make_square(f_to, r_to);
        Move moves[256];
        int cnt_moves = generate_moves(pos, moves);
        for (int i = 0; i < cnt_moves; ++i) {
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
    Value wdl_to_score(int wdl, int ply) {
        switch (wdl) {
            case TB_WIN: return MATE_SCORE - ply - 1;
            case TB_LOSS: return -MATE_SCORE + ply + 1;
            case TB_DRAW: return 0;
            case TB_CURSED_WIN: return 1;
            case TB_BLESSED_LOSS: return -1;
            default: return 0;
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
        std::lock_guard<std::mutex> lock(mtx);
        const auto& e = table[hash_to_index(hash)];
        if (e.count == 0) return 0;
        int32_t adj = (e.total_score * learning_rate) / e.count;
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
class TimeManager {
private:
    int64_t start_time, time_left, increment;
    int moves_to_go, move_time, move_overhead;
    bool infinite, pondering;
    int64_t soft_limit, hard_limit;
    Value prev_score;
    int score_drop_count, best_move_stability_count, game_phase;
public:
    TimeManager() : start_time(0), time_left(0), increment(0), moves_to_go(40),
                    move_time(0), move_overhead(100), infinite(false), pondering(false),
                    soft_limit(0), hard_limit(0), prev_score(0), score_drop_count(0),
                    best_move_stability_count(0), game_phase(0) {}
    void set_side(Color side, int64_t wtime, int64_t btime, int64_t winc, int64_t binc,
                  int moves, int movetime, bool inf, bool pond = false) {
        start_time = current_time();
        infinite = inf; pondering = pond;
        if (movetime > 0) { move_time = movetime; soft_limit = hard_limit = move_time; return; }
        if (infinite || pondering) { move_time = 0; soft_limit = hard_limit = INT64_MAX; return; }
        time_left = (side == WHITE) ? wtime : btime;
        increment = (side == WHITE) ? winc : binc;
        moves_to_go = (moves > 0) ? moves : 40;
        int64_t base = time_left / std::max(moves_to_go, 5) + increment / 2;
        soft_limit = base;
        hard_limit = std::min(time_left / 2, base * 5);
    }
    int64_t current_time() const { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    int64_t elapsed() const { return current_time() - start_time; }
    void set_move_overhead(int ms) { move_overhead = ms; }
    void set_game_phase(int phase) { game_phase = phase; }
    void scale_time(double factor) {
        factor = std::clamp(factor, 0.2, 1.5);
        soft_limit = int64_t(soft_limit * factor);
        hard_limit = int64_t(hard_limit * factor);
    }
    void update(Value current_score, bool best_move_changed) {
        if (current_score < prev_score - 50) score_drop_count++; else score_drop_count = std::max(0, score_drop_count-1);
        best_move_stability_count = best_move_changed ? 0 : best_move_stability_count + 1;
        prev_score = current_score;
    }
    bool time_for_depth(int depth) const {
        if (infinite || pondering) return true;
        int64_t e = elapsed();
        double factor = 1.0;
        if (best_move_stability_count < 3) factor *= 1.5;
        if (score_drop_count > 2) factor *= 1.3;
        factor *= 1.0 + 0.5 * (1.0 - std::abs(game_phase - 12) / 12.0);
        return e < soft_limit * factor;
    }
    bool stop_early() const {
        if (infinite || pondering) return false;
        if (move_time > 0) return elapsed() + move_overhead >= move_time;
        return elapsed() + move_overhead >= hard_limit;
    }
};

// ----------------------------------------------------------------------------
// Global search state and data structures
// ----------------------------------------------------------------------------
std::atomic<bool> stop_search{false};
std::atomic<bool> pondering{false};
std::atomic<uint64_t> nodes{0};
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
struct SplitPoint {
    Position* pos;
    SearchThread* master;
    std::vector<ScoredMove> moves;
    std::atomic<int> next_move;
    int depth, ply;
    Value alpha, beta;
    bool cut;
    std::mutex mtx;
    std::condition_variable cv;
    int workers;
    bool finished;
    Value best_score;
    Move best_move;
    std::vector<Move> pv;
    SplitPoint() : pos(nullptr), master(nullptr), next_move(0), depth(0), ply(0),
                   alpha(-INF), beta(INF), cut(false), workers(0), finished(false),
                   best_score(-INF), best_move(NO_MOVE) {}
};
std::vector<SplitPoint*> active_splits;
std::mutex splits_mutex;
std::condition_variable splits_cv;
std::atomic<bool> threads_idle{false};

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
void help_search(SearchThread* thread);
void help_at_split(SearchThread* thread, SplitPoint* sp);

class SearchThread {
private:
    Position& root_pos;
    TranspositionTable& tt;
    SyzygyTablebase& tb;
    Evaluation& eval;
    OpeningBook* book;
    std::vector<Stack> stack;
    int history[2][64][64];
    int butterfly_history[12][64];
    int correction_history[2][64][64];
    int cont_history[12][64][12][64];
    int counter_moves[64][64];
    int follow_up_moves[64][64];
    int capture_history[12][6][64];   // [moving_piece_idx][captured_pt-1][to_sq]
    int thread_id;
    int total_threads;
    int multi_pv;
    std::vector<Move> pv[MAX_PLY];
    Value prev_eval = 0;
    Move prev_best_move = NO_MOVE;

public:
    bool idle;
    SplitPoint* current_split;
    std::atomic<uint64_t> nodes;

    SearchThread(int id, int total, Position& pos, TranspositionTable& t, SyzygyTablebase& tbb, Evaluation& e, OpeningBook* b)
        : thread_id(id), total_threads(total), root_pos(pos), tt(t), tb(tbb), eval(e), book(b), idle(false), current_split(nullptr), nodes(0) {
        memset(history, 0, sizeof(history));
        memset(butterfly_history, 0, sizeof(butterfly_history));
        memset(correction_history, 0, sizeof(correction_history));
        memset(cont_history, 0, sizeof(cont_history));
        memset(counter_moves, 0, sizeof(counter_moves));
        memset(follow_up_moves, 0, sizeof(follow_up_moves));
        memset(capture_history, 0, sizeof(capture_history));
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

    // ------------------------------------------------------------------------
    // Reduction helper
    // ------------------------------------------------------------------------
    int reduction(bool improving, Depth depth, int move_idx, int move_score, bool capture, bool check) {
        int r = LMR_BASE + (move_idx / LMR_DIV);
        if (depth < 3) r = 0;
        if (!improving) r += 1;
        if (capture) r -= 1;
        if (check) r -= 1;
        if (move_score < 200000) r += 1;
        return std::max(0, std::min(r, depth - 2));
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
            if (m == counter_moves[from_sq(last)][to_sq(last)]) s += 700000;
        }
        if (ply > 1) {
            Move last2 = stack[ply-2].current_move;
            if (m == follow_up_moves[from_sq(last2)][to_sq(last2)]) s += 600000;
        }
        Color us = pos.side_to_move();
        Square from = from_sq(m), to = to_sq(m);
        int moving_pc = pos.piece_on(from);
        int pt = moving_pc & 7;
        int piece_idx = us * 6 + (pt - 1);
        s += history[us][from][to];
        s += butterfly_history[piece_idx][to] / 4;
        s += correction_history[us][from][to] / 8;
        if (ply > 0) {
            int prev_piece_idx = stack[ply-1].current_piece_idx;
            if (prev_piece_idx != -1) {
                Square prev_to = to_sq(stack[ply-1].current_move);
                s += cont_history[prev_piece_idx][prev_to][piece_idx][to] / 8;
            }
        }
        if (pos.piece_on(to)) {
            int cap_pt = (pos.piece_on(to) & 7) - 1;  // 0-indexed captured piece type
            if (cap_pt >= 0 && cap_pt < 6)
                s += capture_history[piece_idx][cap_pt][to] / 4;
            int see_val = pos.see(m);
            s += 500000 + see_val * 100;
        }
        if (pos.gives_check(m)) s += 400000;

        // DTZ bonus for early moves or captures
        if (ply < 5 && (idx < 3 || captured)) {
            Position child = pos;
            child.make_move(m);
            U64 child_key = child.get_hash();
            Value dummy_score;
            Move dummy_move;
            int child_dtz = 0;
            if (tt.probe(child_key, 0, -INF, INF, dummy_score, dummy_move, child_dtz) && child_dtz != 0) {
                if (child_dtz > 0) s += 5000 - child_dtz * 100;
                else s += 3000 + child_dtz * 100;
            }
        }
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

    void update_continuation(Move move, int depth, bool good, const Position& pos, int ply) {
        if (ply <= 0) return;
        Square from = from_sq(move), to = to_sq(move);
        Color us = pos.side_to_move();
        int moving_pc = pos.piece_on(from);
        int pt = moving_pc & 7;
        int cur_piece_idx = us * 6 + (pt - 1);
        int prev_piece_idx = stack[ply-1].current_piece_idx;
        if (prev_piece_idx == -1) return;
        Square prev_to = to_sq(stack[ply-1].current_move);
        int delta = depth * depth;
        if (good) {
            cont_history[prev_piece_idx][prev_to][cur_piece_idx][to] += delta - cont_history[prev_piece_idx][prev_to][cur_piece_idx][to] * abs(delta) / MAX_HISTORY;
        } else {
            cont_history[prev_piece_idx][prev_to][cur_piece_idx][to] -= delta + cont_history[prev_piece_idx][prev_to][cur_piece_idx][to] * abs(delta) / MAX_HISTORY;
        }
        int& val = cont_history[prev_piece_idx][prev_to][cur_piece_idx][to];
        val = std::max(-MAX_HISTORY, std::min(MAX_HISTORY, val));
    }

    // ------------------------------------------------------------------------
    // Quiescence search
    // ------------------------------------------------------------------------
    Value quiescence(Position& pos, Value alpha, Value beta, int ply, int q_depth = 0) {
        if (ply >= MAX_PLY || q_depth >= MAX_QSEARCH_DEPTH)
            return eval.evaluate(pos) + learning.probe(pos.get_hash());
        nodes++;
        if (nodes.load() % 256 == 0) {
            if (stop_search) return 0;
            if (tm.stop_early()) { stop_search = true; return 0; }
        }
        if (node_limit > 0 && nodes >= node_limit) { stop_search = true; return 0; }
        if (pos.is_repetition(2)) return 0;
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

        // Sort: captures by SEE descending; when in check sort all by rough priority
        std::sort(moves, moves + cnt, [&](Move a, Move b) {
            return pos.see(a) > pos.see(b);
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
            eval.get_nnue().push();
#endif
            pos.make_move(m);
            stack[ply].captured_piece = captured;
            stack[ply].current_move = m;
            if (was_promotion) moving_pt = prom_pt;
            int cur_piece_idx = us * 6 + (moving_pt - 1);
            stack[ply].current_piece_idx = cur_piece_idx;
#ifdef USE_NNUE
            eval.get_nnue().make_move(pos, m, us, moving_pt, PieceType(captured & 7), was_promotion, prom_pt);
#endif

            if (pos.mover_in_check()) {
#ifdef USE_NNUE
                eval.get_nnue().pop();
#endif
                pos.undo_move(m, captured, old_castle, old_ep, old_fifty);
                continue;
            }
            legal_count++;
            Value score = -quiescence(pos, -beta, -alpha, ply+1, q_depth+1);
#ifdef USE_NNUE
            eval.get_nnue().pop();
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
            eval.get_nnue().push();
#endif
            pos.make_move(m);
            stack[ply].captured_piece = captured;
            stack[ply].current_move = m;
            if (was_promotion) moving_pt = prom_pt;
            int cur_piece_idx = us * 6 + (moving_pt - 1);
            stack[ply].current_piece_idx = cur_piece_idx;
#ifdef USE_NNUE
            eval.get_nnue().make_move(pos, m, us, moving_pt, PieceType(captured & 7), was_promotion, prom_pt);
#endif

            nodes++;
            std::vector<Move> dummy;
            Value score = -negamax(pos, depth - 4, -alpha - margin, -alpha + margin, ply+1, true, dummy, NO_MOVE);
#ifdef USE_NNUE
            eval.get_nnue().pop();
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
    Value negamax(Position& pos, Depth depth, Value alpha, Value beta, int ply, bool cut, std::vector<Move>& pv_line, Move excluded = NO_MOVE) {
        pv_line.clear();
        if (ply >= MAX_PLY) return eval.evaluate(pos) + learning.probe(pos.get_hash());
        if (nodes.load() % 256 == 0) {
            if (stop_search) return 0;
            if (tm.stop_early()) { stop_search = true; return 0; }
        }
        if (node_limit > 0 && nodes >= node_limit) { stop_search = true; return 0; }
        if (pos.is_repetition(2)) return 0;
        if (tb.can_probe(pos) && depth <= 0) {
            int wdl = tb.probe_wdl(pos);
            if (wdl != TB_RESULT_FAILED) { tb_hits++; return tb.wdl_to_score(wdl, ply); }
        }
        alpha = std::max(alpha, -MATE_SCORE + ply);
        beta  = std::min(beta,  MATE_SCORE - ply - 1);
        if (alpha >= beta) return alpha;

        bool in_check = pos.is_check();
        Value static_eval = eval.evaluate(pos) + learning.probe(pos.get_hash());
        stack[ply].static_eval = static_eval;
        stack[ply].in_check = in_check;
        U64 key = pos.get_hash();
        Move tt_move = NO_MOVE;
        Value tt_score = static_eval;  // safe fallback — never used raw without tt_hit guard
        int tt_dtz = 0;
        bool tt_hit = tt.probe(key, depth, alpha, beta, tt_score, tt_move, tt_dtz);

        if (tt_hit) {
            if (tt_dtz != 0) {
                int sign = (tt_dtz > 0) ? 1 : -1;
                int dist = std::abs(tt_dtz);
                return (sign == 1) ? MATE_SCORE - dist - ply : -MATE_SCORE + dist + ply;
            } else if (tt_score > MATE_OFFSET) {
                // stored as (MATE_SCORE - dist_from_root) which is MATE_SCORE - (N - ply_at_store + ply_at_store)
                // Standard: stored = score + ply_at_store, retrieve = stored - ply_current
                tt_score -= ply;
                if (tt_score > MATE_SCORE - 1) tt_score = MATE_SCORE - 1;
            } else if (tt_score < -MATE_OFFSET) {
                tt_score += ply;
                if (tt_score < -MATE_SCORE + 1) tt_score = -MATE_SCORE + 1;
            }
            return tt_score;
        }

        if (tb.can_probe(pos) && depth <= 3 && tt_dtz == 0 && !tt_hit) {
            int dtz_success;
            int dtz = tb.probe_dtz(pos, dtz_success);
            if (dtz_success) tt.store(key, depth, 0, BOUND_NONE, NO_MOVE, dtz);
        }

        // DTZ pruning
        if (tt_dtz != 0 && depth >= std::abs(tt_dtz) && tt_dtz > 0) {
            return MATE_SCORE - tt_dtz - ply - 1;
        }

        // Singular extension — only when we have a real TT entry with a reliable score
        // (tt_hit guarantees tt_score is set from a depth-sufficient, bounds-matching entry).
        if (tt_hit && depth >= SINGULAR_EXTENSION_DEPTH && tt_move != NO_MOVE &&
            excluded == NO_MOVE && !in_check && std::abs(tt_score) < MATE_SCORE - MAX_PLY) {
            Value singular_beta = std::max(tt_score - SINGULAR_MARGIN, -INF);
            Depth singular_depth = depth / 2;
            std::vector<Move> dummy;
            Value singular_score = -negamax(pos, singular_depth, -singular_beta, -singular_beta + 1, ply, false, dummy, tt_move);
            if (singular_score <= singular_beta) depth++;
        }

        if (depth <= 0) return quiescence(pos, alpha, beta, ply);

        // ProbCut
        if (depth >= PROBCUT_DEPTH && !in_check && abs(beta) < MATE_SCORE - 1000) {
            Value pc_score = probcut(pos, depth, alpha, beta, ply);
            if (pc_score != -INF) return pc_score;
        }

        // Null move pruning
        if (!in_check && depth >= 2 && cut) {
            bool has_non_pawn = false;
            for (int pt = KNIGHT; pt <= QUEEN; ++pt)
                if (pos.bb(pos.side_to_move(), PieceType(pt))) { has_non_pawn = true; break; }
            if (has_non_pawn) {
                bool only_kings_pawns = true;
                for (int pt = KNIGHT; pt <= QUEEN; ++pt)
                    if (pos.bb(WHITE, PieceType(pt)) || pos.bb(BLACK, PieceType(pt))) { only_kings_pawns = false; break; }
                if (!only_kings_pawns) {
#ifdef USE_NNUE
                    eval.get_nnue().push();
#endif
                    pos.make_move(NULL_MOVE);
                    int R = NULL_MOVE_R + depth / 6;
                    Value score = -negamax(pos, depth - R - 1, -beta, -beta+1, ply+1, false, pv_line, NO_MOVE);
#ifdef USE_NNUE
                    eval.get_nnue().pop();
#endif
                    pos.undo_null_move();
                    if (score >= beta) return beta;
                }
            }
        }

        // Razoring
        if (!in_check && depth <= 6) {
            int razor_margin;
            if (depth <= 1) razor_margin = RAZOR_MARGIN_D1;
            else if (depth == 2) razor_margin = RAZOR_MARGIN_D2;
            else if (depth == 3) razor_margin = RAZOR_MARGIN_D3;
            else razor_margin = RAZOR_MARGIN_D3 + 50 * (depth - 3);
            if (static_eval + razor_margin < alpha) {
                if (depth <= 3) {
                    Value rscore = quiescence(pos, alpha, alpha+1, ply);
                    if (rscore <= alpha) return rscore;
                } else {
                    std::vector<Move> dummy;
                    Value rscore = -negamax(pos, depth - 4, -alpha-1, -alpha, ply, false, dummy, NO_MOVE);
                    if (rscore <= alpha) return rscore;
                }
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

        std::vector<ScoredMove> scored;
        for (int i = 0; i < cnt; ++i) {
            if (moves[i] == excluded) continue;
            int captured = pos.piece_on(to_sq(moves[i])) != 0;
            scored.push_back({moves[i], score_move(moves[i], ply, tt_move, pos, i, captured)});
        }
        std::sort(scored.begin(), scored.end(),
            [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });

        // Multi-cut pruning
        if (depth >= 6 && !in_check && cut && tt_move != NO_MOVE) {
            int mc_count = 0;
            for (int i = 0; i < std::min(3, (int)scored.size()); ++i) {
                Move m = scored[i].move;
                if (m == tt_move) continue;
                if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;
                int captured = pos.piece_on(to_sq(m));
                int old_castle = pos.castling_rights(), old_ep = pos.ep_sq(), old_fifty = pos.halfmove_clock();
                Position pos2 = pos;
                pos2.make_move(m);
                if (pos2.mover_in_check()) continue;
                std::vector<Move> dummy;
                Value score = -negamax(pos2, depth / 2, -beta, -beta+1, ply+1, false, dummy, NO_MOVE);
                if (score >= beta && ++mc_count >= 2) return beta;
            }
        }

        if (tt_move == NO_MOVE && depth >= IID_DEPTH) {
            std::vector<Move> dummy;
            negamax(pos, depth - IID_REDUCTION, alpha, beta, ply, false, dummy, NO_MOVE);
        }

        Value best_score = -INF;
        Move best_move = NO_MOVE;
        Bound bound = BOUND_UPPER;
        bool improving = (ply >= 2 && static_eval > stack[ply-2].static_eval);

        // YBWC split attempt
        if (total_threads > 1 && depth >= 6 && scored.size() > 5 && !idle) {
            SplitPoint* sp = new SplitPoint;
            sp->pos = &pos;
            sp->master = this;
            sp->moves = scored;
            sp->depth = depth;
            sp->ply = ply;
            sp->alpha = alpha;
            sp->beta = beta;
            sp->cut = cut;
            sp->next_move = 0;
            sp->workers = 0;
            sp->finished = false;
            sp->best_score = -INF;
            sp->best_move = NO_MOVE;
            {
                std::lock_guard<std::mutex> lock(splits_mutex);
                active_splits.push_back(sp);
                splits_cv.notify_all();
            }
            help_at_split(this, sp);
            {
                std::unique_lock<std::mutex> lock(sp->mtx);
                sp->cv.wait(lock, [sp]{ return sp->finished; });
            }
            {
                std::lock_guard<std::mutex> lock(splits_mutex);
                active_splits.erase(std::remove(active_splits.begin(), active_splits.end(), sp), active_splits.end());
            }
            pv_line = sp->pv;
            best_score = sp->best_score;
            best_move = sp->best_move;
            delete sp;
            if (best_score != -INF) {
                if (best_score >= beta) bound = BOUND_LOWER;
                else if (best_score > alpha) bound = BOUND_EXACT;
                Value store = best_score;
                if (store > MATE_SCORE - MAX_PLY) {
                    store += ply; // push toward root
                } else if (store < -MATE_SCORE + MAX_PLY) {
                    store -= ply; // push toward root
                }
                tt.store(key, depth, store, bound, best_move);
                return best_score;
            }
        }

        // Normal move loop
        for (size_t i = 0; i < scored.size(); ++i) {
            Move m = scored[i].move;
            if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;

            // Futility pruning (per move)
            if (depth <= 3 && !in_check && !pos.piece_on(to_sq(m))) {
                int margin = SEE_QUIET_MARGIN + depth * 50;
                if (scored[i].score < 500000) margin += 4 * depth;
                if (static_eval + margin <= alpha) continue;
            }

            // Late move pruning
            if (!pos.piece_on(to_sq(m)) && !in_check && depth <= 7 && i >= (size_t)(LMP_BASE + depth * LMP_FACTOR)) {
                if (!improving) continue;
                if (i >= (size_t)(LMP_BASE + depth * LMP_FACTOR * 2)) continue;
            }

            int captured = pos.piece_on(to_sq(m));
            int moving_pc = pos.piece_on(from_sq(m));
            PieceType moving_pt = PieceType(moving_pc & 7);
            Color us = pos.side_to_move();
            bool gives_check = pos.gives_check(m);
            bool was_promotion = promotion_type(m) != NO_PIECE;
            PieceType prom_pt = promotion_type(m);
            int old_castle = pos.castling_rights(), old_ep = pos.ep_sq(), old_fifty = pos.halfmove_clock();

#ifdef USE_NNUE
            eval.get_nnue().push();
#endif
            pos.make_move(m);
            stack[ply].captured_piece = captured;
            stack[ply].current_move = m;
            if (was_promotion) moving_pt = prom_pt;
            int cur_piece_idx = us * 6 + (moving_pt - 1);
            stack[ply].current_piece_idx = cur_piece_idx;
#ifdef USE_NNUE
            eval.get_nnue().make_move(pos, m, us, moving_pt, PieceType(captured & 7), was_promotion, prom_pt);
#endif

            if (pos.mover_in_check()) {
#ifdef USE_NNUE
                eval.get_nnue().pop();
#endif
                pos.undo_move(m, captured, old_castle, old_ep, old_fifty);
                continue;
            }

            nodes++;
            Depth new_depth = depth - 1;
            if (in_check) new_depth++;
            int extension = 0;
            if (ply > 0 && stack[ply-1].captured_piece != 0 && to_sq(m) == to_sq(stack[ply-1].current_move))
                extension = 1;
            else if (!extension && moving_pt == PAWN) {
                if (eval.is_passed_pawn(pos, from_sq(m), us) &&
                    ((us == WHITE && rank_of(to_sq(m)) > rank_of(from_sq(m))) ||
                     (us == BLACK && rank_of(to_sq(m)) < rank_of(from_sq(m)))))
                    extension = 1;
            }
            if (gives_check) extension++;
            new_depth += extension;
            new_depth = std::min(new_depth, depth + 2);

            std::vector<Move> child_pv;
            Value score;
            if (i == 0) {
                score = -negamax(pos, new_depth, -beta, -alpha, ply+1, true, child_pv, NO_MOVE);
            } else {
                int red = captured ? 0 : reduction(improving, depth, i, scored[i].score, captured != 0, gives_check);
                score = -negamax(pos, new_depth - red, -alpha-1, -alpha, ply+1, true, child_pv, NO_MOVE);
                if (score > alpha && score < beta)
                    score = -negamax(pos, new_depth, -beta, -alpha, ply+1, true, child_pv, NO_MOVE);
            }

#ifdef USE_NNUE
            eval.get_nnue().pop();
#endif
            pos.undo_move(m, captured, old_castle, old_ep, old_fifty);

            if (stop_search) return 0;

            if (score > best_score) {
                best_score = score;
                best_move = m;
                pv_line = child_pv;
                pv_line.insert(pv_line.begin(), m);
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
                            for (size_t j = 0; j < i; ++j) {
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
                            for (size_t j = 0; j < i; ++j) {
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
        if (store_score > MATE_SCORE - MAX_PLY) {
            store_score += ply;
        } else if (store_score < -MATE_SCORE + MAX_PLY) {
            store_score -= ply;
        }
        tt.store(key, depth, store_score, bound, best_move);
        return best_score;
    }

    // ------------------------------------------------------------------------
    // Output info (thread 0 only)
    // ------------------------------------------------------------------------
    void output_info(int depth, Value score, const std::vector<Move>& pv) {
        int64_t elapsed = tm.elapsed();
        uint64_t nps = elapsed > 0 ? nodes * 1000 / elapsed : 0;
        std::string score_str;
        if (std::abs(score) > MATE_SCORE - 1000) {
            int mate_dist = (score > 0) ? (MATE_SCORE - score) : (MATE_SCORE + score);
            if (mate_dist < 0) mate_dist = 0;
            score_str = (score > 0) ? "mate " + std::to_string(mate_dist) : "mate -" + std::to_string(mate_dist);
        } else {
            score_str = "cp " + std::to_string(score);
        }
        std::cout << "info depth " << depth << " " << score_str
                  << " nodes " << nodes << " nps " << nps
                  << " time " << elapsed << " tbhits " << tb_hits << " pv";
        Position tmp = root_pos;
        for (Move m : pv) {
            // Validate that m is legal in the current position.
            // TT hash collisions or stale PV entries can inject illegal moves;
            // applying them corrupts the position and makes every subsequent
            // move appear illegal in the GUI.
            Move legal_moves[MAX_MOVES];
            int legal_cnt = generate_moves(tmp, legal_moves);
            bool found = false;
            for (int li = 0; li < legal_cnt; ++li) {
                if (legal_moves[li] == m) {
                    // Final check: make the move and confirm the mover is not
                    // left in check (pseudo-legal generator may include some).
                    Position check_tmp = tmp;
                    check_tmp.make_move(m);
                    if (!check_tmp.mover_in_check()) { found = true; break; }
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
        }
        std::cout << std::endl;
    }

    // ------------------------------------------------------------------------
    // Main search entry (root)
    // ------------------------------------------------------------------------
    void search(int max_depth, uint64_t max_nodes, const std::vector<ScoredMove>& root_moves) {
        stop_search = false;
        nodes = 0;
        node_limit = max_nodes;
        tb_hits = 0;
        tt.new_search();

        std::vector<ScoredMove> local_root_moves = root_moves;
        if (local_root_moves.empty()) {
            Move moves[MAX_MOVES];
            int cnt = generate_moves(root_pos, moves);
            for (int i = 0; i < cnt; ++i) {
                Move m = moves[i];
                if (root_pos.piece_on(to_sq(m)) && ((root_pos.piece_on(to_sq(m)) & 7) == KING)) continue;
                Position tmp = root_pos;
                tmp.make_move(m);
                if (!tmp.mover_in_check()) local_root_moves.push_back({m, 0});
            }
        }
        if (local_root_moves.empty()) return;

#ifdef USE_NNUE
        eval.get_nnue().evaluate(root_pos);
#endif

        Move best_move = local_root_moves[0].move;
        Value best_score = -INF;
        prev_best_move = NO_MOVE;
        idle = false;

        for (int depth = 1; depth <= max_depth && !stop_search; ++depth) {
            if (depth > 1 && !tm.time_for_depth(depth)) break;

            for (auto& sm : local_root_moves) {
                int captured = root_pos.piece_on(to_sq(sm.move)) != 0;
                sm.score = score_move(sm.move, 0, (best_move != NO_MOVE ? best_move : NO_MOVE), root_pos, 0, captured);
            }
            if (best_move != NO_MOVE) {
                for (auto& sm : local_root_moves)
                    if (sm.move == best_move) sm.score = 10000000;
            }
            std::sort(local_root_moves.begin(), local_root_moves.end(),
                [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });

            Value alpha = -INF, beta = INF;
            if (depth >= 5) {
                alpha = best_score - ASPIRATION_WINDOW;
                beta  = best_score + ASPIRATION_WINDOW;
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
                    if (root_pos.piece_on(to_sq(m)) && ((root_pos.piece_on(to_sq(m)) & 7) == KING)) continue;

                    int cap = root_pos.piece_on(to_sq(m));
                    int moving_pc = root_pos.piece_on(from_sq(m));
                    PieceType moving_pt = PieceType(moving_pc & 7);
                    Color us = root_pos.side_to_move();
                    bool was_promotion = promotion_type(m) != NO_PIECE;
                    PieceType prom_pt = promotion_type(m);
                    int oc = root_pos.castling_rights(), oe = root_pos.ep_sq(), of_ = root_pos.halfmove_clock();
                    Position pos2 = root_pos;

#ifdef USE_NNUE
                    eval.get_nnue().push();
#endif
                    pos2.make_move(m);
                    stack[0].captured_piece = cap;
                    stack[0].current_move = m;
                    if (was_promotion) moving_pt = prom_pt;
                    int cur_piece_idx = us * 6 + (moving_pt - 1);
                    stack[0].current_piece_idx = cur_piece_idx;
#ifdef USE_NNUE
                    eval.get_nnue().make_move(pos2, m, us, moving_pt, PieceType(cap & 7), was_promotion, prom_pt);
#endif

                    if (pos2.mover_in_check()) {
#ifdef USE_NNUE
                        eval.get_nnue().pop();
#endif
                        continue;
                    }

                    nodes++;
                    std::vector<Move> pv_line;
                    Value score;
                    if (i == 0 || window_alpha == -INF) {
                        score = -negamax(pos2, depth - 1, -beta, -window_alpha, 1, true, pv_line, NO_MOVE);
                    } else {
                        score = -negamax(pos2, depth - 1, -window_alpha - 1, -window_alpha, 1, true, pv_line, NO_MOVE);
                        if (!stop_search && score > window_alpha && score < beta)
                            score = -negamax(pos2, depth - 1, -beta, -window_alpha, 1, true, pv_line, NO_MOVE);
                    }

#ifdef USE_NNUE
                    eval.get_nnue().pop();
#endif

                    if (stop_search) break;

                    if (score > depth_score) {
                        depth_score = score;
                        depth_best = m;
                        pv_line.insert(pv_line.begin(), m);
                        depth_best_pv = pv_line;  // save PV for end-of-depth reporting
                        if (thread_id == 0 && multi_pv > 1) {
                            std::lock_guard<std::mutex> lock(root_infos_mutex);
                            for (auto& info : root_infos) {
                                if (info.move == m) {
                                    info.score = score;
                                    info.pv = pv_line;
                                    break;
                                }
                            }
                        }
                    }
                    if (score > window_alpha) window_alpha = score;
                }

                if (!stop_search && depth >= 5) {
                    if (depth_score <= alpha && alpha > -INF) {
                        alpha = std::max(Value(-INF), alpha - ASPIRATION_WIDEN);
                        need_retry = true;
                    } else if (depth_score >= beta && beta < INF) {
                        beta = std::min(Value(INF), beta + ASPIRATION_WIDEN);
                        need_retry = true;
                    }
                }
            }

            if (!stop_search && depth_best != NO_MOVE) {
                best_move = depth_best;
                best_score = depth_score;
                prev_eval = best_score;
                if (thread_id == 0) {
                    bool best_move_changed = (depth_best != prev_best_move);
                    tm.update(best_score, best_move_changed);
                    prev_best_move = depth_best;
                    // Emit a single info line per depth, AFTER the full depth
                    // completes — not inside the move loop where nodes/score
                    // are mid-computation and the node count is misleading.
                    if (multi_pv <= 1) {
                        output_info(depth, best_score, depth_best_pv);
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
                                Position pv_tmp = root_pos;
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
            Value prev = shared_best_score.load(std::memory_order_relaxed);
            while (best_score > prev && !shared_best_score.compare_exchange_weak(prev, best_score, std::memory_order_relaxed)) {}
            if (best_score >= shared_best_score.load(std::memory_order_relaxed))
                shared_best_move.store(best_move, std::memory_order_relaxed);
        }

        if (local_root_moves.empty()) {
            idle = true;
            help_search(this);
        }
    }

    void search(int max_depth, uint64_t max_nodes) { search(max_depth, max_nodes, {}); }
};

// ----------------------------------------------------------------------------
// Global helper functions for work stealing
// ----------------------------------------------------------------------------
void help_search(SearchThread* thread) {
    while (!stop_search) {
        SplitPoint* sp = nullptr;
        {
            std::unique_lock lock(splits_mutex);
            splits_cv.wait(lock, []{ return !active_splits.empty() || stop_search; });
            if (stop_search) return;
            for (auto* s : active_splits) {
                if (s->next_move < (int)s->moves.size()) {
                    sp = s;
                    break;
                }
            }
        }
        if (sp) help_at_split(thread, sp);
    }
}

void help_at_split(SearchThread* thread, SplitPoint* sp) {
    sp->workers++;
    while (true) {
        int idx = sp->next_move.fetch_add(1);
        if (idx >= (int)sp->moves.size()) break;
        Move m = sp->moves[idx].move;
        if (sp->pos->piece_on(to_sq(m)) && ((sp->pos->piece_on(to_sq(m)) & 7) == KING)) continue;

        int captured = sp->pos->piece_on(to_sq(m));
        int moving_pc = sp->pos->piece_on(from_sq(m));
        PieceType moving_pt = PieceType(moving_pc & 7);
        Color us = sp->pos->side_to_move();
        bool was_promotion = promotion_type(m) != NO_PIECE;
        PieceType prom_pt = promotion_type(m);
        int old_castle = sp->pos->castling_rights(), old_ep = sp->pos->ep_sq(), old_fifty = sp->pos->halfmove_clock();

        Position pos2 = *sp->pos;
#ifdef USE_NNUE
        thread->eval.get_nnue().push();
#endif
        pos2.make_move(m);
#ifdef USE_NNUE
        thread->eval.get_nnue().make_move(pos2, m, us, moving_pt, PieceType(captured & 7), was_promotion, prom_pt);
#endif
        if (pos2.mover_in_check()) {
#ifdef USE_NNUE
            thread->eval.get_nnue().pop();
#endif
            continue;
        }

        thread->nodes++;
        Depth new_depth = sp->depth - 1;
        if (sp->pos->is_check()) new_depth++;
        std::vector<Move> child_pv;
        Value score = -thread->negamax(pos2, new_depth, -sp->beta, -sp->alpha, sp->ply + 1, sp->cut, child_pv, NO_MOVE);

#ifdef USE_NNUE
        thread->eval.get_nnue().pop();
#endif

        {
            std::lock_guard<std::mutex> lock(sp->mtx);
            if (score > sp->best_score) {
                sp->best_score = score;
                sp->best_move = m;
                sp->pv = child_pv;
                sp->pv.insert(sp->pv.begin(), m);
                if (score > sp->alpha) sp->alpha = score;
            }
        }
    }
    sp->workers--;
    if (sp->workers == 0) {
        std::lock_guard<std::mutex> lock(sp->mtx);
        sp->finished = true;
        sp->cv.notify_one();
    }
}

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

public:
    UCI() : tt(256), search_active(false), pondering_active(false), thread_count(1), multi_pv(1),
            ponder(false), contempt(0), chess960(false), uci_limit_strength(false), uci_elo(1500),
            learning_enabled(false), learning_rate(100), learning_max_adjust(50),
            tuning_mode(false) {
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
            if (value == "false") book = OpeningBook();
        } else if (name == "BookFile") {
            if (!value.empty()) book.load(value);
        } else if (name == "BookVariety") {
            book.set_variety(std::stod(value));
        } else if (name == "SyzygyPath") {
            if (!value.empty()) tb.init(value);
        } else if (name == "EvalFile") {
#ifdef USE_NNUE
            eval.set_nnue(value);
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
        } else if (name == "TuningFile") {
            tuning_file = value;
            if (tuning_mode && !tuning_file.empty()) {
                tuning_stream.open(tuning_file, std::ios::app);
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
        }
        if (!infinite && movetime == 0 && wtime == 0 && btime == 0) infinite = true;

        if (uci_limit_strength && !infinite) {
            int elo_depth = 1 + (uci_elo - 800) / 100;
            elo_depth = std::clamp(elo_depth, 1, 30);
            depth = std::min(depth, elo_depth);
        }

        tm.set_side(pos.side_to_move(), wtime, btime, winc, binc, movestogo, movetime, infinite, ponder_mode);
        tm.set_game_phase(pos.game_phase());

        if (!ponder_mode && !infinite) {
            Move book_move = book.probe(pos);
            if (book_move != NO_MOVE) {
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

        Move moves[MAX_MOVES];
        int cnt = generate_moves(pos, moves);
        std::vector<ScoredMove> filtered_root_moves;
        for (int i = 0; i < cnt; ++i) {
            Move m = moves[i];
            if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING)) continue;
            Position tmp = pos;
            tmp.make_move(m);
            if (!tmp.mover_in_check()) filtered_root_moves.push_back({m, 0});
        }

        if (filtered_root_moves.empty()) {
            std::cout << "bestmove 0000\n";
            return;
        }

        double time_factor = 0.5 + filtered_root_moves.size() / 64.0;
        time_factor = std::clamp(time_factor, 0.2, 1.5);
        tm.scale_time(time_factor);

        for (const auto& sm : filtered_root_moves) {
            root_infos.push_back({sm.move, -INF, {}});
        }

        int total_moves = filtered_root_moves.size();
        int moves_per_thread = (total_moves + thread_count - 1) / thread_count;

        pondering = ponder_mode;
        pondering_active = ponder_mode;
        search_active = true;
        threads_idle = false;

        for (int i = 0; i < thread_count; ++i) {
            int start = i * moves_per_thread;
            int end = std::min(start + moves_per_thread, total_moves);
            if (start >= total_moves) break;

            std::vector<ScoredMove> thread_moves(
                filtered_root_moves.begin() + start,
                filtered_root_moves.begin() + end
            );

            search_threads.emplace_back([this, i, depth, nodes, thread_moves]() {
                auto st = std::make_unique<SearchThread>(i, thread_count, pos, tt, tb, eval, &book);
                st->set_multi_pv(multi_pv);
                st->search(depth, nodes, thread_moves);
            });
        }

        if (!ponder_mode) {
            for (auto& t : search_threads) {
                if (t.joinable()) t.join();
            }
            search_threads.clear();
            search_active = false;

            if (!root_infos.empty()) {
                std::lock_guard<std::mutex> lock(last_pv_mutex);
                auto best_it = std::max_element(root_infos.begin(), root_infos.end());
                if (best_it != root_infos.end() && best_it->score > -INF + 1000) {
                    last_pv = best_it->pv;
                }
            }

            Move best = shared_best_move.load();
            if (best == NO_MOVE && !filtered_root_moves.empty()) {
                best = filtered_root_moves[0].move;
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
        Move best = shared_best_move.load();
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
        Move best = shared_best_move.load();
        if (best == NO_MOVE) {
            Move moves[MAX_MOVES];
            int cnt = generate_moves(pos, moves);
            if (cnt > 0) best = moves[0];
        }
        if (best != NO_MOVE) {
            std::cout << "bestmove " << move_to_uci(best, &pos) << std::endl;
        } else {
            std::cout << "bestmove 0000\n";
        }
    }

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            if (token == "uci") {
                std::cout << "id name Hugine 2.0\n";
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
                std::cout << "option name Hash type spin default 256 min 1 max 8192\n";
                std::cout << "option name Threads type spin default 1 min 1 max 64\n";
                std::cout << "option name Ponder type check default false\n";
                std::cout << "option name OwnBook type check default true\n";
                std::cout << "option name BookFile type string default\n";
                std::cout << "option name BookVariety type spin default 0 min 0 max 10\n";
                std::cout << "option name SyzygyPath type string default\n";
                std::cout << "option name EvalFile type string default\n";
                std::cout << "option name MultiPV type spin default 1 min 1 max 5\n";
                std::cout << "option name Contempt type spin default 0 min -100 max 100\n";
                std::cout << "option name Move Overhead type spin default 100 min 0 max 5000\n";
                std::cout << "option name UCI_Chess960 type check default false\n";
                std::cout << "option name UCI_LimitStrength type check default false\n";
                std::cout << "option name UCI_Elo type spin default 1500 min 800 max 3000\n";
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
                iss >> value;
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
                            return;
                        }
                        Position tmp = pos;
                        for (Move m : last_pv) {
                            U64 key = tmp.get_hash();
                            learning.update(key, result, tmp.side_to_move());
                            tmp.make_move(m);
                        }
                        std::cout << "info string Learning updated with " << last_pv.size() << " positions.\n";
                    }
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
                // Perft – divide at depth 1, timing, correct undo state
                //
                // captured passed to undo_move is always board[to] BEFORE the
                // move (i.e. 0 for castling/ep since those squares are empty).
                // undo_move's own is_castling / is_en_passant branches handle
                // rook and captured-pawn restoration without using that value.
                // ------------------------------------------------------------
                int depth = 1;
                {
                    std::string tok;
                    if (iss >> tok) depth = std::stoi(tok);
                }
                if (depth < 1) depth = 1;

                // ---- Verify castle flag encoding round-trip ----
                // Every castling move generated must have CASTLE_FLAG set and
                // its to-square must be the fixed king destination (g/c file).
                // We print a one-time confirmation at the start of each perft.
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

                    // Also confirm castling rights were parsed from FEN
                    int rights_count = 0;
                    for (int c = 0; c < 2; ++c)
                        for (int s = 0; s < 2; ++s)
                            if (pos.castle_rook(Color(c), s) != -1) rights_count++;
                    std::cout << "info string FEN castling rights loaded: "
                              << rights_count << " slot(s) active.\n";
                }

                // ---- Recursive perft with make/undo ----
                // Returns the number of leaf nodes at depth d from position p.
                // Legal-move filter: make the move, check mover_in_check(), undo.
                std::function<uint64_t(Position&, int)> perft_r =
                    [&](Position& p, int d) -> uint64_t {
                    Move mvs[MAX_MOVES];
                    int cnt = generate_moves(p, mvs);

                    // Bulk-count at depth 1: just count legal moves, no recursion
                    if (d == 1) {
                        uint64_t legal = 0;
                        for (int i = 0; i < cnt; ++i) {
                            // captured = piece currently on the target square
                            // (0 for ep and castling since those squares are empty)
                            int cap = p.piece_on(to_sq(mvs[i]));
                            int old_cr = p.castling_rights();
                            int old_ep = p.ep_sq();
                            int old_50 = p.halfmove_clock();
                            p.make_move(mvs[i]);
                            if (!p.mover_in_check()) ++legal;
                            p.undo_move(mvs[i], cap, old_cr, old_ep, old_50);
                        }
                        return legal;
                    }

                    uint64_t nodes = 0;
                    for (int i = 0; i < cnt; ++i) {
                        int cap = p.piece_on(to_sq(mvs[i]));
                        int old_cr = p.castling_rights();
                        int old_ep = p.ep_sq();
                        int old_50 = p.halfmove_clock();
                        p.make_move(mvs[i]);
                        if (!p.mover_in_check())
                            nodes += perft_r(p, d - 1);
                        p.undo_move(mvs[i], cap, old_cr, old_ep, old_50);
                    }
                    return nodes;
                };

                // ---- Divide: report each root move's subtree count ----
                auto t0 = std::chrono::steady_clock::now();

                Move root_mvs[MAX_MOVES];
                int root_cnt = generate_moves(pos, root_mvs);
                uint64_t total = 0;
                for (int i = 0; i < root_cnt; ++i) {
                    int cap = pos.piece_on(to_sq(root_mvs[i]));
                    int old_cr = pos.castling_rights();
                    int old_ep = pos.ep_sq();
                    int old_50 = pos.halfmove_clock();
                    pos.make_move(root_mvs[i]);
                    if (!pos.mover_in_check()) {
                        uint64_t n = (depth <= 1) ? 1 : perft_r(pos, depth - 1);
                        pos.undo_move(root_mvs[i], cap, old_cr, old_ep, old_50);
                        std::cout << move_to_uci(root_mvs[i], &pos) << ": " << n << "\n";
                        total += n;
                    } else {
                        pos.undo_move(root_mvs[i], cap, old_cr, old_ep, old_50);
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
