/*
 * hugine_final.cpp  –  Fully corrected version with optional DEBUG instrumentation
 *
 * Compile normally:          g++ -O2 -std=c++17 -pthread hugine_final.cpp -o hugine
 * Compile with debug output: g++ -O2 -std=c++17 -pthread -DDEBUG hugine_final.cpp -o hugine
 *
 * Debug output goes to stderr, does not interfere with UCI.
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

// ============================================================================
// PLATFORM / ARCHITECTURE DETECTION
// ============================================================================
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

// ============================================================================
// SYZYGY (Fathom) availability – auto‑enabled only on x86 with fathom/ present
// ============================================================================
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

// ============================================================================
// Fathom include or stubs
// ============================================================================
#if HAS_SYZYGY
  #if ARCH_ARM || OS_ANDROID
    #error "Fathom/Syzygy cannot be compiled on ARM/Android (uses x86 intrinsics). Add -DNO_SYZYGY."
  #endif
extern "C" {
#include "fathom/src/tbprobe.h"
}
#else
// Stubs so the rest of the code compiles unchanged
#define TB_RESULT_FAILED  0xFFFFFFFF
#define TB_WIN            2
#define TB_LOSS           0
#define TB_DRAW           1
#define TB_CURSED_WIN     3
#define TB_BLESSED_LOSS   (-1)
#define TB_PAWN           1
#define TB_KNIGHT         2
#define TB_BISHOP         3
#define TB_ROOK           4
#define TB_QUEEN          5
#define TB_KING           6
#define TB_SIDEMASK       0x40
#define TB_MAX_MOVES      256
inline bool     tb_init(const char*)                                                               { return false; }
inline void     tb_free()                                                                          {}
inline int      tb_max_cardinality()                                                               { return 0; }
inline unsigned tb_probe_wdl(unsigned*,unsigned*,int,int,int,int,int,int,int,int)                 { return TB_RESULT_FAILED; }
inline unsigned* tb_probe_root(unsigned*,unsigned*,int,int,int,int,int,int,int,int,void*)         { return nullptr; }
#endif // HAS_SYZYGY

// ============================================================================
// NNUE availability (must be explicitly enabled with -DUSE_NNUE)
// ============================================================================
// (USE_NNUE is handled inline with #ifdef later)

using U64 = uint64_t;
using Move = uint32_t;
using Square = int;
using Value = int;
using Depth = int;

enum Color { WHITE, BLACK };
enum PieceType { NO_PIECE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum Bound { BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT };

constexpr Square NO_SQUARE = -1;

// --- FIX: NULL_MOVE sentinel (outside 12-bit move encoding) ---
constexpr Move NO_MOVE = 0;
constexpr Move NULL_MOVE = 0xFFFFFFFFu;          // sentinel, cannot be a legal move

constexpr int MAX_PLY = 128;
constexpr int MAX_MOVES = 256;
constexpr Value MATE_SCORE = 32000;
constexpr Value INF = 32001;
constexpr int MATE_OFFSET = 20000;               // offset for TT mate storage
constexpr int ASPIRATION_WINDOW = 15;
constexpr int ASPIRATION_WIDEN = 50;
constexpr int RAZOR_MARGIN_D1 = 300;
constexpr int RAZOR_MARGIN_D2 = 400;
constexpr int RAZOR_MARGIN_D3 = 600;
constexpr int FUTILITY_MARGIN_FACTOR = 200;
constexpr int FUTILITY_MARGINS[8] = {0, 100, 250, 400, 550, 700, 850, 1000};
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
constexpr int HISTORY_BONUS = 32;
constexpr int HISTORY_MALUS = 32;
constexpr int PROBCUT_DEPTH = 5;
constexpr int PROBCUT_MARGIN = 100;

constexpr int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 0};
constexpr int PHASE_KNIGHT = 1;
constexpr int PHASE_BISHOP = 1;
constexpr int PHASE_ROOK = 2;
constexpr int PHASE_QUEEN = 4;
constexpr int TOTAL_PHASE = 24;

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
inline Square lsb(U64 b) { return b ? Square(__builtin_ctzll(b)) : 0; }
inline Square pop_lsb(U64& b) { Square s = lsb(b); b &= b - 1; return s; }

// ============================================================================
// Magic bitboard declarations
// ============================================================================
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
extern U64 rook_attacks_magic(Square s, U64 occ);
extern U64 bishop_attacks_magic(Square s, U64 occ);
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

// --- FIX: Freshly generated collision‑free magic numbers ---
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
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    52, 53, 53, 53, 53, 53, 53, 52
};

const int bishop_shifts[64] = {
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
};

U64 rook_mask(Square s) {
    U64 mask = 0;
    int f = file_of(s), r = rank_of(s);
    for (int rr = r + 1; rr < 7; ++rr) mask |= 1ULL << make_square(f, rr);
    for (int rr = r - 1; rr > 0; --rr) mask |= 1ULL << make_square(f, rr);
    for (int ff = f + 1; ff < 7; ++ff) mask |= 1ULL << make_square(ff, r);
    for (int ff = f - 1; ff > 0; --ff) mask |= 1ULL << make_square(ff, r);
    return mask;
}

U64 bishop_mask(Square s) {
    U64 mask = 0;
    int f = file_of(s), r = rank_of(s);
    for (int i = 1; f + i < 7 && r + i < 7; ++i) mask |= 1ULL << make_square(f + i, r + i);
    for (int i = 1; f - i > 0 && r + i < 7; ++i) mask |= 1ULL << make_square(f - i, r + i);
    for (int i = 1; f + i < 7 && r - i > 0; ++i) mask |= 1ULL << make_square(f + i, r - i);
    for (int i = 1; f - i > 0 && r - i > 0; ++i) mask |= 1ULL << make_square(f - i, r - i);
    return mask;
}

void init_magics() {
    U64* rook_ptr = rook_attacks_table;
    U64* bishop_ptr = bishop_attacks_table;

    for (int sq = 0; sq < 64; ++sq) {
        // Rooks
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
            // up
            for (int rr = r + 1; rr < 8; ++rr) {
                Square s2 = make_square(f, rr);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            // down
            for (int rr = r - 1; rr >= 0; --rr) {
                Square s2 = make_square(f, rr);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            // right
            for (int ff = f + 1; ff < 8; ++ff) {
                Square s2 = make_square(ff, r);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            // left
            for (int ff = f - 1; ff >= 0; --ff) {
                Square s2 = make_square(ff, r);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            // Store at magic‑hashed index (FIX)
            U64 idx = (occ * magic) >> shift;
            rook_ptr[idx] = attacks;
        }
        rook_ptr += num_occ;

        // Bishops
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
            // NE
            for (int i = 1; f + i < 8 && r + i < 8; ++i) {
                Square s2 = make_square(f + i, r + i);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            // NW
            for (int i = 1; f - i >= 0 && r + i < 8; ++i) {
                Square s2 = make_square(f - i, r + i);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            // SE
            for (int i = 1; f + i < 8 && r - i >= 0; ++i) {
                Square s2 = make_square(f + i, r - i);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            // SW
            for (int i = 1; f - i >= 0 && r - i >= 0; ++i) {
                Square s2 = make_square(f - i, r - i);
                attacks |= 1ULL << s2;
                if (occ & (1ULL << s2)) break;
            }
            U64 idx = (occ * magic) >> shift;
            bishop_ptr[idx] = attacks;
        }
        bishop_ptr += num_occ;
    }
}

U64 rook_attacks_magic(Square s, U64 occ) {
    Magic& m = rook_magics[s];
    occ &= m.mask;
    occ *= m.magic;
    occ >>= m.shift;
    return m.attacks[occ];
}

U64 bishop_attacks_magic(Square s, U64 occ) {
    Magic& m = bishop_magics[s];
    occ &= m.mask;
    occ *= m.magic;
    occ >>= m.shift;
    return m.attacks[occ];
}

U64 queen_attacks_magic(Square s, U64 occ) {
    return rook_attacks_magic(s, occ) | bishop_attacks_magic(s, occ);
}

// ============================================================================
// Position class
// ============================================================================
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
    int castle_rights;
    U64 hash;
    std::vector<U64> history;

public:
    Position() { clear(); }
    void clear() {
        memset(_pieces,  0, sizeof(_pieces));
        memset(board,    0, sizeof(board));
        memset(&occupied, 0, sizeof(occupied));
        side         = WHITE;
        fifty        = 0;
        ply          = 0;
        game_ply     = 0;
        ep_square    = -1;
        castle_rights = 0;
        hash         = 0;
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
        castle_rights = 1 | 2 | 4 | 8;
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
        castle_rights = 0;
        if (castle.find('K') != std::string::npos) castle_rights |= 1;
        if (castle.find('Q') != std::string::npos) castle_rights |= 2;
        if (castle.find('k') != std::string::npos) castle_rights |= 4;
        if (castle.find('q') != std::string::npos) castle_rights |= 8;
        ep_square = -1;
        if (ep != "-") {
            int f = ep[0] - 'a';
            int r = ep[1] - '1';
            ep_square = make_square(f, r);
        }
        fifty = hmvc;
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
                    char p = ' ';
                    switch (pt) {
                        case PAWN:   p = 'p'; break;
                        case KNIGHT: p = 'n'; break;
                        case BISHOP: p = 'b'; break;
                        case ROOK:   p = 'r'; break;
                        case QUEEN:  p = 'q'; break;
                        case KING:   p = 'k'; break;
                        default: break;
                    }
                    if (c == WHITE) p = toupper(p);
                    fen += p;
                }
            }
            if (empty > 0) fen += std::to_string(empty);
            if (r > 0) fen += '/';
        }
        fen += (side == WHITE) ? " w " : " b ";
        std::string castle_str;
        if (castle_rights & 1) castle_str += 'K';
        if (castle_rights & 2) castle_str += 'Q';
        if (castle_rights & 4) castle_str += 'k';
        if (castle_rights & 8) castle_str += 'q';
        if (castle_str.empty()) castle_str = '-';
        fen += castle_str + " ";
        if (ep_square != -1)
            fen += char('a' + file_of(ep_square)) + std::to_string(rank_of(ep_square) + 1);
        else
            fen += '-';
        fen += " " + std::to_string(fifty) + " " + std::to_string(game_ply);
        return fen;
    }

    static U64 zobrist_pieces[2][7][64];
    static U64 zobrist_side;
    static U64 zobrist_castle[16];
    static U64 zobrist_ep[64];
    static bool zobrist_initialized;

    static void init_zobrist() {
        if (zobrist_initialized) return;
        std::mt19937_64 rng(0xDEADBEEF);
        for (int c = 0; c < 2; ++c)
            for (int pt = 0; pt < 7; ++pt)
                for (int sq = 0; sq < 64; ++sq)
                    zobrist_pieces[c][pt][sq] = rng();
        zobrist_side = rng();
        for (int i = 0; i < 16; ++i) zobrist_castle[i] = rng();
        for (int i = 0; i < 64; ++i) zobrist_ep[i] = rng();
        zobrist_initialized = true;
    }

    void compute_hash() {
        U64 h = 0;
        for (int c = 0; c < 2; ++c)
            for (int pt = PAWN; pt <= KING; ++pt) {
                U64 bb = _pieces[c][pt];
                while (bb) {
                    Square sq = pop_lsb(bb);
                    h ^= zobrist_pieces[c][pt][sq];
                }
            }
        if (side == BLACK) h ^= zobrist_side;
        h ^= zobrist_castle[castle_rights & 15];
        if (ep_square != -1)
            h ^= zobrist_ep[ep_square];
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

    // attacks_to using current occupancy
    U64 attacks_to(Square s) const {
        return attacks_to(s, occupied);
    }

    // attacks_to using custom occupancy (for SEE)
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

    // --- FIX: SEE with proper occupancy updates for X‑rays ---
    Value see(Move m) const {
        if (m == NULL_MOVE) return 0;
        Square from = from_sq(m), to = to_sq(m);
        int victim_type = board[to] & 7;
        if (victim_type == 0) return 0;

        Value gain[32];
        int d = 0;
        gain[d] = PIECE_VALUES[victim_type];
        U64 occ = occupied;
        Color stm = Color(side ^ 1);
        occ ^= (1ULL << from);
        occ ^= (1ULL << to);

        while (true) {
            int best_att = 0;
            Square best_sq = -1;
            for (int pt = PAWN; pt <= KING; ++pt) {
                U64 attackers = _pieces[stm][pt] & occ & attacks_to(to, occ); // use local occ
                if (attackers) {
                    best_att = pt;
                    best_sq = lsb(attackers);
                    break;
                }
            }
            if (best_sq == -1) break;
            d++;
            gain[d] = PIECE_VALUES[best_att] - gain[d-1];
            occ ^= (1ULL << best_sq);
            stm = Color(stm ^ 1);
        }
        while (d > 0) {
            gain[d-1] = -std::max(-gain[d-1], gain[d]);
            d--;
        }
        return gain[0];
    }

    bool gives_check(Move m) const {
        if (m == NULL_MOVE) return false;
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
            if (to == make_square(6,0)) {
                _pieces[us][ROOK] ^= 1ULL << make_square(7,0);
                _pieces[us][ROOK] |= 1ULL << make_square(5,0);
                board[make_square(7,0)] = 0;
                board[make_square(5,0)] = (us << 3) | ROOK;
            } else if (to == make_square(2,0)) {
                _pieces[us][ROOK] ^= 1ULL << make_square(0,0);
                _pieces[us][ROOK] |= 1ULL << make_square(3,0);
                board[make_square(0,0)] = 0;
                board[make_square(3,0)] = (us << 3) | ROOK;
            } else if (to == make_square(6,7)) {
                _pieces[us][ROOK] ^= 1ULL << make_square(7,7);
                _pieces[us][ROOK] |= 1ULL << make_square(5,7);
                board[make_square(7,7)] = 0;
                board[make_square(5,7)] = (us << 3) | ROOK;
            } else if (to == make_square(2,7)) {
                _pieces[us][ROOK] ^= 1ULL << make_square(0,7);
                _pieces[us][ROOK] |= 1ULL << make_square(3,7);
                board[make_square(0,7)] = 0;
                board[make_square(3,7)] = (us << 3) | ROOK;
            }
        } else if (is_en_passant(m)) {
            Square ep_cap = to + ((us == WHITE) ? -8 : 8);
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
        if (captured && !is_en_passant(m)) {
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
            if (us == WHITE) castle_rights &= ~(1 | 2);
            else castle_rights &= ~(4 | 8);
        }
        if (from == make_square(0,0)) castle_rights &= ~2;
        if (from == make_square(7,0)) castle_rights &= ~1;
        if (from == make_square(0,7)) castle_rights &= ~8;
        if (from == make_square(7,7)) castle_rights &= ~4;
        if (pt == PAWN && abs(to - from) == 16) {
            ep_square = (us == WHITE) ? from + 8 : from - 8;
        } else {
            ep_square = -1;
        }
        if (captured || pt == PAWN)
            fifty = 0;
        else
            fifty++;
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
        if (m == NULL_MOVE) {
            undo_null_move();
            return;
        }
        side = Color(side ^ 1);
        Square from = from_sq(m), to = to_sq(m);
        int pc = board[to];
        int pt = pc & 7;
        Color us = side;
        _pieces[us][pt] ^= 1ULL << to;
        board[to] = captured;
        _pieces[us][pt] |= 1ULL << from;
        board[from] = (us << 3) | pt;
        if (captured && !is_en_passant(m)) {
            int cap_pt = captured & 7;
            Color them = Color(us ^ 1);
            _pieces[them][cap_pt] |= 1ULL << to;
        }
        if (is_castling(m)) {
            if (to == make_square(6,0)) {
                _pieces[us][ROOK] ^= 1ULL << make_square(5,0);
                _pieces[us][ROOK] |= 1ULL << make_square(7,0);
                board[make_square(5,0)] = 0;
                board[make_square(7,0)] = (us << 3) | ROOK;
            } else if (to == make_square(2,0)) {
                _pieces[us][ROOK] ^= 1ULL << make_square(3,0);
                _pieces[us][ROOK] |= 1ULL << make_square(0,0);
                board[make_square(3,0)] = 0;
                board[make_square(0,0)] = (us << 3) | ROOK;
            } else if (to == make_square(6,7)) {
                _pieces[us][ROOK] ^= 1ULL << make_square(5,7);
                _pieces[us][ROOK] |= 1ULL << make_square(7,7);
                board[make_square(5,7)] = 0;
                board[make_square(7,7)] = (us << 3) | ROOK;
            } else if (to == make_square(2,7)) {
                _pieces[us][ROOK] ^= 1ULL << make_square(3,7);
                _pieces[us][ROOK] |= 1ULL << make_square(0,7);
                board[make_square(3,7)] = 0;
                board[make_square(0,7)] = (us << 3) | ROOK;
            }
        } else if (is_en_passant(m)) {
            Square ep_cap = to + ((us == WHITE) ? -8 : 8);
            _pieces[us^1][PAWN] |= 1ULL << ep_cap;
            board[ep_cap] = ((us^1) << 3) | PAWN;
        }
        if (promotion_type(m)) {
            _pieces[us][promotion_type(m)] ^= 1ULL << from;
            _pieces[us][PAWN] |= 1ULL << from;
            board[from] = (us << 3) | PAWN;
        }
        castle_rights = old_castle;
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
    int castling_rights() const { return castle_rights; }

    bool mover_in_check() const {
        Color prev_mover = Color(side ^ 1);
        if (!_pieces[prev_mover][KING]) return false;
        Square ksq = lsb(_pieces[prev_mover][KING]);
        return is_attacked(ksq, side);
    }
};

U64 Position::zobrist_pieces[2][7][64];
U64 Position::zobrist_side;
U64 Position::zobrist_castle[16];
U64 Position::zobrist_ep[64];
bool Position::zobrist_initialized = false;

// ============================================================================
// Move generation – fixed: no king captures, correct separation of captures/non‑captures
// ============================================================================
int generate_moves(const Position& pos, Move* moves, bool captures_only = false) {
    int count = 0;
    Color us = pos.side_to_move();
    Color them = Color(us ^ 1);

    // Opponent pieces excluding their king (can never capture the king)
    U64 their_pieces_no_king = pos.bb(them, PAWN) | pos.bb(them, KNIGHT) | pos.bb(them, BISHOP) |
                                pos.bb(them, ROOK) | pos.bb(them, QUEEN);
    U64 empty = ~pos.occupied_bb();

    // Knights
    U64 knights = pos.bb(us, KNIGHT);
    while (knights) {
        Square from = pop_lsb(knights);
        U64 attacks = Bitboards::knight_attacks[from];
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
            U64 noncaps = attacks & empty;
            while (noncaps) {
                Square to = pop_lsb(noncaps);
                moves[count++] = make_move(from, to);
            }
        }
    }

    // Bishops
    U64 bishops = pos.bb(us, BISHOP);
    while (bishops) {
        Square from = pop_lsb(bishops);
        U64 attacks = bishop_attacks_magic(from, pos.occupied_bb());
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
            U64 noncaps = attacks & empty;
            while (noncaps) {
                Square to = pop_lsb(noncaps);
                moves[count++] = make_move(from, to);
            }
        }
    }

    // Rooks
    U64 rooks = pos.bb(us, ROOK);
    while (rooks) {
        Square from = pop_lsb(rooks);
        U64 attacks = rook_attacks_magic(from, pos.occupied_bb());
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
            U64 noncaps = attacks & empty;
            while (noncaps) {
                Square to = pop_lsb(noncaps);
                moves[count++] = make_move(from, to);
            }
        }
    }

    // Queens
    U64 queens = pos.bb(us, QUEEN);
    while (queens) {
        Square from = pop_lsb(queens);
        U64 attacks = queen_attacks_magic(from, pos.occupied_bb());
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
            U64 noncaps = attacks & empty;
            while (noncaps) {
                Square to = pop_lsb(noncaps);
                moves[count++] = make_move(from, to);
            }
        }
    }

    // King – cannot capture the opponent king, already excluded
    if (pos.bb(us, KING)) {
        Square from = lsb(pos.bb(us, KING));
        U64 attacks = Bitboards::king_attacks[from];
        if (captures_only) {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
        } else {
            U64 caps = attacks & their_pieces_no_king;
            while (caps) {
                Square to = pop_lsb(caps);
                moves[count++] = make_move(from, to);
            }
            U64 noncaps = attacks & empty;
            while (noncaps) {
                Square to = pop_lsb(noncaps);
                moves[count++] = make_move(from, to);
            }
        }
    }

    // Pawns
    U64 pawns = pos.bb(us, PAWN);
    int forward = (us == WHITE) ? 8 : -8;
    U64 promo_rank = (us == WHITE) ? 0xFF00000000000000ULL : 0xFFULL; // rank 7 for white, rank 0 for black

    while (pawns) {
        Square from = pop_lsb(pawns);
        Square to = from + forward;

        // Non‑capture pawn pushes
        if (!captures_only && to >= 0 && to < 64 && !pos.piece_on(to)) {
            if (promo_rank & (1ULL << to)) {
                moves[count++] = make_promotion(from, to, QUEEN);
                moves[count++] = make_promotion(from, to, ROOK);
                moves[count++] = make_promotion(from, to, BISHOP);
                moves[count++] = make_promotion(from, to, KNIGHT);
            } else {
                moves[count++] = make_move(from, to);
                // Double push
                if ((us == WHITE && rank_of(from) == 1) || (us == BLACK && rank_of(from) == 6)) {
                    Square to2 = from + 2*forward;
                    if (!pos.piece_on(to2))
                        moves[count++] = make_move(from, to2);
                }
            }
        }

        // Pawn captures
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

        // En passant
        if (pos.ep_sq() != -1) {
            U64 ep_attacks = Bitboards::pawn_attacks[us][from] & (1ULL << pos.ep_sq());
            if (ep_attacks) {
                moves[count++] = make_move(from, pos.ep_sq()) | ENPASSANT_FLAG;
            }
        }
    }

    // Castling (only if not captures_only)
    if (!captures_only && !pos.is_check()) {
        if (us == WHITE) {
            if ((pos.castling_rights() & 1) &&
                !(pos.occupied_bb() & (1ULL << make_square(5,0))) &&
                !(pos.occupied_bb() & (1ULL << make_square(6,0))) &&
                !pos.is_attacked(make_square(4,0), BLACK) &&
                !pos.is_attacked(make_square(5,0), BLACK) &&
                !pos.is_attacked(make_square(6,0), BLACK))
                moves[count++] = make_move(make_square(4,0), make_square(6,0)) | CASTLE_FLAG;
            if ((pos.castling_rights() & 2) &&
                !(pos.occupied_bb() & (1ULL << make_square(3,0))) &&
                !(pos.occupied_bb() & (1ULL << make_square(2,0))) &&
                !(pos.occupied_bb() & (1ULL << make_square(1,0))) &&
                !pos.is_attacked(make_square(4,0), BLACK) &&
                !pos.is_attacked(make_square(3,0), BLACK) &&
                !pos.is_attacked(make_square(2,0), BLACK))
                moves[count++] = make_move(make_square(4,0), make_square(2,0)) | CASTLE_FLAG;
        } else {
            if ((pos.castling_rights() & 4) &&
                !(pos.occupied_bb() & (1ULL << make_square(5,7))) &&
                !(pos.occupied_bb() & (1ULL << make_square(6,7))) &&
                !pos.is_attacked(make_square(4,7), WHITE) &&
                !pos.is_attacked(make_square(5,7), WHITE) &&
                !pos.is_attacked(make_square(6,7), WHITE))
                moves[count++] = make_move(make_square(4,7), make_square(6,7)) | CASTLE_FLAG;
            if ((pos.castling_rights() & 8) &&
                !(pos.occupied_bb() & (1ULL << make_square(3,7))) &&
                !(pos.occupied_bb() & (1ULL << make_square(2,7))) &&
                !(pos.occupied_bb() & (1ULL << make_square(1,7))) &&
                !pos.is_attacked(make_square(4,7), WHITE) &&
                !pos.is_attacked(make_square(3,7), WHITE) &&
                !pos.is_attacked(make_square(2,7), WHITE))
                moves[count++] = make_move(make_square(4,7), make_square(2,7)) | CASTLE_FLAG;
        }
    }
    return count;
}

// ============================================================================
// Piece‑square tables
// ============================================================================
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

// ============================================================================
// NNUE evaluator (optional)
// ============================================================================
#ifdef USE_NNUE
class NNUEEvaluator {
private:
    static constexpr int FT_SIZE = 256;
    static constexpr int L1_SIZE = 32;
    static constexpr int L2_SIZE = 32;
    static constexpr int INPUT_SIZE = 768;
    struct Layer {
        std::vector<int16_t> weights;
        std::vector<int16_t> bias;
    };
    Layer ft, l1, l2, output;
    int16_t output_bias;
    struct Accumulator {
        std::vector<int16_t> values;
        bool computed;
    } acc[2];
    int piece_index(PieceType pt, Color c) const {
        return (pt - 1) * 2 + (c == WHITE ? 0 : 1);
    }
    void refresh_accumulator(const Position& pos, int perspective) {
        auto& a = acc[perspective];
        a.values.assign(FT_SIZE, 0);
        for (int i = 0; i < FT_SIZE; ++i) a.values[i] = ft.bias[i];
        for (Square sq = 0; sq < 64; ++sq) {
            int pc = pos.piece_on(sq);
            if (pc == 0) continue;
            Color c = Color(pc >> 3);
            PieceType pt = PieceType(pc & 7);
            int idx = piece_index(pt, c) * 64 + sq;
            for (int i = 0; i < FT_SIZE; ++i)
                a.values[i] += ft.weights[idx * FT_SIZE + i];
        }
        a.computed = true;
    }
    std::vector<int16_t> activate(const std::vector<int16_t>& v) const {
        std::vector<int16_t> out(FT_SIZE);
        for (int i = 0; i < FT_SIZE; ++i) {
            int16_t x = v[i];
            out[i] = std::max(0, std::min(127, (int)x));
        }
        return out;
    }
    int16_t forward(const std::vector<int16_t>& input) const {
        std::vector<int16_t> l1_out(L1_SIZE, 0);
        for (int i = 0; i < L1_SIZE; ++i) {
            int32_t sum = l1.bias[i];
            for (int j = 0; j < FT_SIZE; ++j)
                sum += input[j] * l1.weights[j * L1_SIZE + i];
            l1_out[i] = std::max(0, std::min(127, (int)sum));
        }
        std::vector<int16_t> l2_out(L2_SIZE, 0);
        for (int i = 0; i < L2_SIZE; ++i) {
            int32_t sum = l2.bias[i];
            for (int j = 0; j < L1_SIZE; ++j)
                sum += l1_out[j] * l2.weights[j * L2_SIZE + i];
            l2_out[i] = std::max(0, std::min(127, (int)sum));
        }
        int32_t out = output_bias;
        for (int i = 0; i < L2_SIZE; ++i)
            out += l2_out[i] * output.weights[i];
        return out;
    }
public:
    NNUEEvaluator() {
        ft.weights.resize(INPUT_SIZE * FT_SIZE, 0);
        ft.bias.resize(FT_SIZE, 0);
        l1.weights.resize(FT_SIZE * L1_SIZE, 0);
        l1.bias.resize(L1_SIZE, 0);
        l2.weights.resize(L1_SIZE * L2_SIZE, 0);
        l2.bias.resize(L2_SIZE, 0);
        output.weights.resize(L2_SIZE, 0);
        output_bias = 0;
        acc[0].computed = acc[1].computed = false;
    }
    bool load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        uint32_t magic, version, ft_dim, l1_dim, l2_dim, out_dim;
        file.read((char*)&magic, sizeof(magic));
        file.read((char*)&version, sizeof(version));
        file.read((char*)&ft_dim, sizeof(ft_dim));
        file.read((char*)&l1_dim, sizeof(l1_dim));
        file.read((char*)&l2_dim, sizeof(l2_dim));
        file.read((char*)&out_dim, sizeof(out_dim));
        if (magic != 0x5A5A5A5A || version != 1 || ft_dim != FT_SIZE || l1_dim != L1_SIZE || l2_dim != L2_SIZE || out_dim != 1)
            return false;
        auto read_layer = [&](Layer& l, size_t cnt, size_t bias_size) {
            l.weights.resize(cnt);
            l.bias.resize(bias_size);
            file.read((char*)l.weights.data(), l.weights.size() * sizeof(int16_t));
            file.read((char*)l.bias.data(), l.bias.size() * sizeof(int16_t));
        };
        read_layer(ft, INPUT_SIZE * FT_SIZE, FT_SIZE);
        read_layer(l1, FT_SIZE * L1_SIZE, L1_SIZE);
        read_layer(l2, L1_SIZE * L2_SIZE, L2_SIZE);
        output.weights.resize(L2_SIZE);
        file.read((char*)output.weights.data(), L2_SIZE * sizeof(int16_t));
        file.read((char*)&output_bias, sizeof(int16_t));
        acc[0].computed = acc[1].computed = false;
        return true;
    }
    int evaluate(const Position& pos) {
        if (!acc[0].computed) refresh_accumulator(pos, 0);
        auto l0 = activate(acc[0].values);
        int16_t out = forward(l0);
        return out / 10;
    }
};
#endif

// ============================================================================
// Classical evaluation
// ============================================================================
class Evaluation {
private:
#ifdef USE_NNUE
    NNUEEvaluator nnue;
    float nnue_weight;
#endif
    bool is_passed_pawn(const Position& pos, Square sq, Color c) const {
        int f = file_of(sq), r = rank_of(sq);
        for (int df = -1; df <= 1; ++df) {
            int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            int start = (c == WHITE) ? r+1 : 0;
            int end   = (c == WHITE) ? 7 : r-1;
            for (int nr = start; nr <= end; ++nr) {
                Square s = make_square(nf, nr);
                if (pos.piece_on(s) && ((pos.piece_on(s) & 7) == PAWN) && ((pos.piece_on(s) >> 3) != c))
                    return false;
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
        if (Bitboards::pawn_attacks[1-c][sq] & pos.bb(Color(1-c),PAWN)) return 0;
        int r = rank_of(sq);
        if (c == WHITE && r >= 4) return 30;
        if (c == BLACK && r <= 3) return 30;
        return 0;
    }
    int king_safety(const Position& pos, Color c) const {
        Square ksq = pos.bb(c,KING) ? lsb(pos.bb(c,KING)) : NO_SQUARE;
        if (ksq == NO_SQUARE) return 0;
        int kf = file_of(ksq), kr = rank_of(ksq), shield = 0;
        for (int df = -1; df <= 1; ++df) {
            int f = kf + df;
            if (f < 0 || f > 7) continue;
            for (int dr = 1; dr <= 2; ++dr) {
                int r = (c == WHITE) ? kr + dr : kr - dr;
                if (r < 0 || r > 7) continue;
                Square s = make_square(f, r);
                if (pos.piece_on(s) && ((pos.piece_on(s) & 7) == PAWN) && ((pos.piece_on(s) >> 3) == c))
                    shield += 15;
            }
        }
        U64 enemy_pawns = pos.bb(Color(1-c), PAWN);
        while (enemy_pawns) {
            Square sq = pop_lsb(enemy_pawns);
            if (abs(file_of(sq) - kf) <= 1 && abs(rank_of(sq) - kr) <= 3)
                shield -= 5;
        }
        return shield;
    }
    int space_bonus(const Position& pos, Color c) const {
        int space = 0;
        U64 pawns = pos.bb(c,PAWN);
        while (pawns) {
            Square sq = pop_lsb(pawns);
            int f = file_of(sq), r = rank_of(sq);
            if (f >= 2 && f <= 5) {
                if (c == WHITE && r >= 4) space += (r - 3);
                if (c == BLACK && r <= 3) space += (4 - r);
            }
        }
        return space * 5;
    }
    int imbalance(const Position& pos) const {
        int wm = popcount(pos.bb(WHITE,KNIGHT)) + popcount(pos.bb(WHITE,BISHOP));
        int bm = popcount(pos.bb(BLACK,KNIGHT)) + popcount(pos.bb(BLACK,BISHOP));
        int wr = popcount(pos.bb(WHITE,ROOK)), br = popcount(pos.bb(BLACK,ROOK));
        int wq = popcount(pos.bb(WHITE,QUEEN)), bq = popcount(pos.bb(BLACK,QUEEN));
        return (wm - bm) * 15 + (wr - br) * 20 + (wq - bq) * 40;
    }
public:
    Evaluation() {
#ifdef USE_NNUE
        nnue_weight = 0.8f;
#endif
    }
#ifdef USE_NNUE
    void set_nnue(const std::string& filename) { nnue.load(filename); }
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
                U64 pieces_bb = pos.bb(Color(c), PieceType(pt));
                while (pieces_bb) {
                    Square sq = pop_lsb(pieces_bb);
                    int idx = (c == WHITE) ? sq : 63 - sq;
                    int mg = 0, eg = 0;
                    switch (pt) {
                        case PAWN:   mg = PST_PAWN[idx]; eg = PST_PAWN[idx]; break;
                        case KNIGHT: mg = PST_KNIGHT[idx]; eg = PST_KNIGHT[idx]; break;
                        case BISHOP: mg = PST_BISHOP[idx]; eg = PST_BISHOP[idx]; break;
                        case ROOK:   mg = PST_ROOK[idx]; eg = PST_ROOK[idx]; break;
                        case QUEEN:  mg = PST_QUEEN[idx]; eg = PST_QUEEN[idx]; break;
                        case KING:   mg = PST_KING_MG[idx]; eg = PST_KING_EG[idx]; break;
                        default: break;
                    }
                    int pst = (mg * mg_w + eg * eg_w) / TOTAL_PHASE;
                    if (c == WHITE) score += pst + PIECE_VALUES[pt];
                    else score -= pst + PIECE_VALUES[pt];
                }
            }
        }
        int mob_white = 0, mob_black = 0;
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
                mob_white += mobility_bonus(PieceType(pt), popcount(attacks));
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
                mob_black += mobility_bonus(PieceType(pt), popcount(attacks));
            }
        }
        score += (mob_white - mob_black);
        for (int c = 0; c < 2; ++c) {
            U64 pawns = pos.bb(Color(c), PAWN);
            for (int f = 0; f < 8; ++f) {
                int cnt = popcount(pawns & (0x0101010101010101ULL << f));
                if (cnt > 1) {
                    int penalty = (cnt - 1) * 20;
                    if (c == WHITE) score -= penalty; else score += penalty;
                }
            }
            U64 tmp = pawns;
            while (tmp) {
                Square sq = pop_lsb(tmp);
                int f = file_of(sq);
                bool isolated = true;
                if ((f > 0 && (pawns & (0x0101010101010101ULL << (f-1)))) ||
                    (f < 7 && (pawns & (0x0101010101010101ULL << (f+1)))))
                    isolated = false;
                if (isolated) {
                    if (c == WHITE) score -= 15; else score += 15;
                }
            }
            tmp = pawns;
            while (tmp) {
                Square sq = pop_lsb(tmp);
                if (is_passed_pawn(pos, sq, Color(c))) {
                    int r = rank_of(sq);
                    int adv = (c == WHITE) ? r : 7 - r;
                    int bonus = 30 + adv * adv * 4;
                    Square ksq = pos.bb(Color(1-c), KING) ? lsb(pos.bb(Color(1-c), KING)) : NO_SQUARE;
                    if (ksq != NO_SQUARE) {
                        int kf = file_of(ksq), kr = rank_of(ksq);
                        int dist = std::max(std::abs(kf - file_of(sq)), std::abs(kr - r));
                        if (dist < 3) bonus += 10;
                    }
                    if (c == WHITE) score += bonus; else score -= bonus;
                }
            }
        }
        for (int c = 0; c < 2; ++c) {
            U64 knights = pos.bb(Color(c), KNIGHT);
            while (knights) {
                Square sq = pop_lsb(knights);
                int bonus = outpost_bonus(pos, sq, Color(c));
                if (c == WHITE) score += bonus; else score -= bonus;
            }
            U64 bishops = pos.bb(Color(c), BISHOP);
            while (bishops) {
                Square sq = pop_lsb(bishops);
                int bonus = outpost_bonus(pos, sq, Color(c));
                if (c == WHITE) score += bonus; else score -= bonus;
            }
        }
        if (popcount(pos.bb(WHITE,BISHOP)) >= 2) score += 50;
        if (popcount(pos.bb(BLACK,BISHOP)) >= 2) score -= 50;
        U64 seventh = (pos.side_to_move() == WHITE) ? 0xFFULL << 48 : 0xFFULL << 8;
        score += popcount(pos.bb(WHITE,ROOK) & seventh) * 30;
        score -= popcount(pos.bb(BLACK,ROOK) & seventh) * 30;
        if (!pos.is_endgame()) score += king_safety(pos, WHITE) - king_safety(pos, BLACK);
        score += space_bonus(pos, WHITE) - space_bonus(pos, BLACK);
        score += imbalance(pos);
#ifdef USE_NNUE
        if (nnue_weight > 0) {
            int nn = nnue.evaluate(pos);
            if (pos.side_to_move() == BLACK) nn = -nn;
            score = Value(nnue_weight * nn + (1.0 - nnue_weight) * score);
        }
#endif
        return (pos.side_to_move() == WHITE) ? score : -score;
    }
};

// ============================================================================
// Transposition Table
// ============================================================================
struct TTEntry {
    U64 key;
    Depth depth;
    Value score;
    Bound bound;
    Move move;
    int age;
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
    void store(U64 key, Depth depth, Value score, Bound bound, Move move) {
        std::unique_lock lock(mtx);
        size_t idx = key % size;
        TTEntry& e = table[idx];
        if (e.key == key && e.depth > depth) return;
        e = {key, depth, score, bound, move, age};
    }
    bool probe(U64 key, Depth depth, Value alpha, Value beta, Value& score, Move& move) {
        std::shared_lock lock(mtx);
        size_t idx = key % size;
        TTEntry& e = table[idx];
        if (e.key != key) return false;
        move = e.move;
        if (e.depth >= depth) {
            if (e.bound == BOUND_EXACT) {
                score = e.score;
#ifdef DEBUG
                fprintf(stderr, "DBG_TT_HIT key=%llx depth=%d entry_depth=%d bound=EXACT score=%d move=%d\n",
                        (unsigned long long)key, depth, e.depth, e.score, e.move);
#endif
                return true;
            }
            if (e.bound == BOUND_LOWER && e.score >= beta) {
#ifdef DEBUG
                fprintf(stderr, "DBG_TT_CUTOFF LOWER key=%llx depth=%d entry_depth=%d score=%d move=%d\n",
                        (unsigned long long)key, depth, e.depth, e.score, e.move);
#endif
                score = e.score; return true;
            }
            if (e.bound == BOUND_UPPER && e.score <= alpha) {
#ifdef DEBUG
                fprintf(stderr, "DBG_TT_CUTOFF UPPER key=%llx depth=%d entry_depth=%d score=%d move=%d\n",
                        (unsigned long long)key, depth, e.depth, e.score, e.move);
#endif
                score = e.score; return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Opening Book
// ============================================================================
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
                double w = std::pow(e.weight, 1.0 + variety / 10.0);
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

// ============================================================================
// Syzygy Tablebase wrapper
// ============================================================================
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
            (pos.castling_rights() & 1) ? 1 : 0,
            (pos.castling_rights() & 2) ? 1 : 0,
            (pos.castling_rights() & 4) ? 1 : 0,
            (pos.castling_rights() & 8) ? 1 : 0,
            pos.ep_sq() != -1 ? file_of(pos.ep_sq()) : 0,
            pos.halfmove_clock(),
            (pos.side_to_move() == WHITE) ? 0 : 1);
    }
    unsigned probe_root(Position& pos, std::vector<unsigned>& results) {
        if (!can_probe(pos)) return 0;
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
        unsigned* res = tb_probe_root(pieces, squares, cnt,
            (pos.castling_rights() & 1) ? 1 : 0,
            (pos.castling_rights() & 2) ? 1 : 0,
            (pos.castling_rights() & 4) ? 1 : 0,
            (pos.castling_rights() & 8) ? 1 : 0,
            pos.ep_sq() != -1 ? file_of(pos.ep_sq()) : 0,
            pos.halfmove_clock(),
            (pos.side_to_move() == WHITE) ? 0 : 1,
            nullptr);
        if (!res) return 0;
        results.clear();
        for (int i = 0; i < TB_MAX_MOVES && res[i] != 0; ++i) results.push_back(res[i]);
        return results.size();
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
    Move best_move_from_results(const Position& pos, const std::vector<unsigned>& results) {
        if (results.empty()) return NO_MOVE;
        unsigned res = results[0];
        unsigned pg_move = (res >> 8) & 0xFFFF;
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
};

// ============================================================================
// Time Manager
// ============================================================================
class TimeManager {
private:
    int64_t start_time;
    int64_t time_left;
    int64_t increment;
    int moves_to_go;
    int move_time;
    bool infinite;
    int move_overhead;
    int64_t soft_limit, hard_limit;
    double best_move_stability;
    Value prev_score;
    int score_drop_count;

public:
    TimeManager() : start_time(0), time_left(0), increment(0), moves_to_go(40),
                    move_time(0), infinite(false), move_overhead(100),
                    best_move_stability(1.0), prev_score(0), score_drop_count(0) {}
    void set_side(Color side, int64_t wtime, int64_t btime, int64_t winc, int64_t binc,
                  int moves, int movetime, bool inf) {
        start_time = current_time();
        infinite = inf;
        if (movetime > 0) { move_time = movetime; soft_limit = hard_limit = move_time; return; }
        if (infinite) { move_time = 0; soft_limit = hard_limit = INT64_MAX; return; }
        time_left = (side == WHITE) ? wtime : btime;
        increment = (side == WHITE) ? winc : binc;
        moves_to_go = (moves > 0) ? moves : 40;
        int64_t base = time_left / std::max(moves_to_go, 5) + increment / 2;
        soft_limit = base;
        hard_limit = std::min(time_left / 2, base * 5);
    }
    int64_t current_time() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    int64_t elapsed() const { return current_time() - start_time; }
    void set_move_overhead(int ms) { move_overhead = ms; }
    bool time_for_depth(int depth) const {
        if (infinite) return true;
        int64_t elapsed_time = elapsed();
        double time_factor = best_move_stability;
        if (score_drop_count > 0) time_factor *= 1.5;
        return elapsed_time < soft_limit * time_factor;
    }
    bool stop_early() const {
        if (infinite) return false;
        if (move_time > 0) return elapsed() + move_overhead >= move_time;
        return elapsed() + move_overhead >= hard_limit;
    }
};

// ============================================================================
// Global search state
// ============================================================================
std::atomic<bool> stop_search{false};
std::atomic<uint64_t> nodes{0};
std::atomic<uint64_t> node_limit{0};
std::atomic<uint64_t> tb_hits{0};
TimeManager tm;
std::atomic<Move> shared_best_move{NO_MOVE};
std::atomic<Value> shared_best_score{-INF};

struct Stack {
    Move killers[2];
    Move counter;
    int ply;
    int static_eval;
    bool in_check;
    Move current_move;
    int excluded_move;
};

struct ScoredMove {
    Move move;
    int score;
};

// ============================================================================
// Search Thread
// ============================================================================
class SearchThread {
private:
    Position& root_pos;
    TranspositionTable& tt;
    SyzygyTablebase& tb;
    Evaluation& eval;
    OpeningBook* book;
    std::vector<Stack> stack;
    int history[2][64][64];
    int counter_moves[64][64];
    int follow_up_moves[64][64];
    int threat_history[2][64][64];
    int thread_id;
    int multi_pv;
    std::vector<Move> pv[MAX_PLY];
    Value prev_eval = 0;

    int reduction(bool improving, Depth depth, int move_idx, int move_score) {
        int r = LMR_BASE + (move_idx / LMR_DIV);
        if (depth < 3) r = 0;
        if (!improving) r += 1;
        if (move_score < 200000) r += 1; // low history
        return std::min(r, depth - 1);
    }

    int score_move(Move m, int ply, Move tt_move, const Position& pos) {
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
        s += history[us][from_sq(m)][to_sq(m)];
        s += threat_history[us][from_sq(m)][to_sq(m)];
        if (pos.piece_on(to_sq(m))) {
            int victim = pos.piece_on(to_sq(m)) & 7;
            int attacker = pos.piece_on(from_sq(m)) & 7;
            s += 500000 + PIECE_VALUES[victim] - PIECE_VALUES[attacker];
        }
        if (pos.gives_check(m)) s += 400000;
        return s;
    }

    void update_history(Move move, int depth, bool good, const Position& pos) {
        Square from = from_sq(move), to = to_sq(move);
        Color us = pos.side_to_move();
        int delta = depth * depth;
        if (good) {
            history[us][from][to] += delta - history[us][from][to] * abs(delta) / MAX_HISTORY;
        } else {
            history[us][from][to] -= delta + history[us][from][to] * abs(delta) / MAX_HISTORY;
        }
        history[us][from][to] = std::max(-MAX_HISTORY, std::min(MAX_HISTORY, history[us][from][to]));
    }

    Value quiescence(Position& pos, Value alpha, Value beta, int ply) {
        if (ply >= MAX_PLY) return eval.evaluate(pos);
        if (nodes.load() % 1024 == 0 && tm.stop_early()) { stop_search = true; return 0; }
        if (node_limit > 0 && nodes >= node_limit) { stop_search = true; return 0; }
        if (pos.is_repetition(2)) return 0;
        if (tb.can_probe(pos)) {
            int wdl = tb.probe_wdl(pos);
            if (wdl != TB_RESULT_FAILED) { tb_hits++; return tb.wdl_to_score(wdl, ply); }
        }
        Value stand_pat = eval.evaluate(pos);
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
        Move moves[256];
        int cnt = generate_moves(pos, moves, true);
        std::sort(moves, moves + cnt, [&](Move a, Move b) {
            int av = 0, bv = 0;
            if (pos.piece_on(to_sq(a))) av = PIECE_VALUES[pos.piece_on(to_sq(a)) & 7];
            if (pos.piece_on(to_sq(b))) bv = PIECE_VALUES[pos.piece_on(to_sq(b)) & 7];
            return av > bv;
        });
        for (int i = 0; i < cnt; ++i) {
            Move m = moves[i];
            // Skip if move would capture the opponent's king
            if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING))
                continue;
            int victim_val = pos.piece_on(to_sq(m)) ? PIECE_VALUES[pos.piece_on(to_sq(m)) & 7] : 0;
            if (victim_val + 200 + stand_pat < alpha) continue;
            int captured = pos.piece_on(to_sq(m));
            int old_castle = pos.castling_rights(), old_ep = pos.ep_sq(), old_fifty = pos.halfmove_clock();
            // --- FIX: set current_move before making the move ---
            stack[ply].current_move = m;
            pos.make_move(m);
            if (pos.mover_in_check()) { pos.undo_move(m, captured, old_castle, old_ep, old_fifty); continue; }
            nodes++;
            Value score = -quiescence(pos, -beta, -alpha, ply+1);
            pos.undo_move(m, captured, old_castle, old_ep, old_fifty);
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        return alpha;
    }

    Value probcut(Position& pos, Depth depth, Value alpha, Value beta, int ply) {
        if (depth < PROBCUT_DEPTH) return -INF;
        Move moves[256];
        int cnt = generate_moves(pos, moves, true);
        Value best = -INF;
        for (int i = 0; i < cnt; ++i) {
            Move m = moves[i];
            // Skip if move would capture the opponent's king
            if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING))
                continue;
            int captured = pos.piece_on(to_sq(m));
            if (!captured) continue;
            int victim = captured & 7;
            int attacker = pos.piece_on(from_sq(m)) & 7;
            Value gain = PIECE_VALUES[victim] - PIECE_VALUES[attacker];
            if (gain + PROBCUT_MARGIN < alpha) continue;
            int old_castle = pos.castling_rights(), old_ep = pos.ep_sq(), old_fifty = pos.halfmove_clock();
            // --- FIX: set current_move before making the move ---
            stack[ply].current_move = m;
            pos.make_move(m);
            nodes++;
            std::vector<Move> dummy_pv;
            Value score = -negamax(pos, depth - 4, -alpha - PROBCUT_MARGIN, -alpha + PROBCUT_MARGIN, ply+1, true, dummy_pv);
            pos.undo_move(m, captured, old_castle, old_ep, old_fifty);
            if (score > best) best = score;
            if (score >= beta) return score;
        }
        // --- FIX: only return if >= beta, else -INF ---
        return (best >= beta) ? best : -INF;
    }

    Value negamax(Position& pos, Depth depth, Value alpha, Value beta, int ply, bool cut, std::vector<Move>& pv_line) {
        pv_line.clear();
        if (ply >= MAX_PLY) return eval.evaluate(pos);
        if (nodes.load() % 1024 == 0 && tm.stop_early()) { stop_search = true; return 0; }
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
        Value static_eval = eval.evaluate(pos);
        stack[ply].static_eval = static_eval;
        stack[ply].in_check = in_check;
        U64 key = pos.get_hash();
        Move tt_move = NO_MOVE;
        Value tt_score;
        if (tt.probe(key, depth, alpha, beta, tt_score, tt_move)) {
            // --- FIX: decode mate score using ply ---
            if (std::abs(tt_score) > MATE_OFFSET) {
                int dist = (tt_score > 0) ? tt_score - MATE_OFFSET : -tt_score - MATE_OFFSET;
                if (tt_score > 0)
                    tt_score = MATE_SCORE - (ply + dist);
                else
                    tt_score = -(MATE_SCORE - (ply - dist));
            }
            return tt_score;
        }
        if (depth <= 0) return quiescence(pos, alpha, beta, ply);

        // ProbCut
        if (depth >= PROBCUT_DEPTH && !in_check && abs(beta) < MATE_SCORE - 1000) {
            Value pc_score = probcut(pos, depth, alpha, beta, ply);
            if (pc_score != -INF) return pc_score;
        }

        // Null move pruning (adaptive reduction)
        if (!in_check && depth >= 2 && cut) {
            bool has_non_pawn = false;
            for (int pt = KNIGHT; pt <= QUEEN; ++pt)
                if (pos.bb(pos.side_to_move(), PieceType(pt))) { has_non_pawn = true; break; }
            if (has_non_pawn) {
                pos.make_move(NULL_MOVE);
                int R = NULL_MOVE_R + depth / 6;
                Value score = -negamax(pos, depth - R - 1, -beta, -beta+1, ply+1, false, pv_line);
                pos.undo_null_move();
                if (score >= beta) return beta;
            }
        }
        // Razoring
        if (!in_check && depth <= 3 && alpha < static_eval - 400) {
            Value rscore = quiescence(pos, alpha, beta, ply);
            if (rscore <= alpha) return rscore;
        }
        // Futility pruning
        if (!in_check && depth <= 7 && static_eval - FUTILITY_MARGIN_FACTOR * depth >= beta)
            return static_eval;
        Move moves[256];
        int cnt = generate_moves(pos, moves);
        if (cnt == 0) return in_check ? -MATE_SCORE + ply : 0;
        std::vector<ScoredMove> scored;
        for (int i = 0; i < cnt; ++i) {
            scored.push_back({moves[i], score_move(moves[i], ply, tt_move, pos)});
        }
        std::sort(scored.begin(), scored.end(),
            [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });
        if (tt_move == NO_MOVE && depth >= IID_DEPTH) {
            std::vector<Move> dummy;
            negamax(pos, depth - IID_REDUCTION, alpha, beta, ply, false, dummy);
        }
        Value best_score = -INF;
        Move best_move = NO_MOVE;
        Bound bound = BOUND_UPPER;
        bool improving = (ply >= 2 && static_eval > stack[ply-2].static_eval);
        for (int i = 0; i < cnt; ++i) {
            Move m = scored[i].move;
            // Skip if move would capture the opponent's king
            if (pos.piece_on(to_sq(m)) && ((pos.piece_on(to_sq(m)) & 7) == KING))
                continue;
            if (depth <= 3 && !in_check && !pos.piece_on(to_sq(m)) && pos.see(m) < SEE_QUIET_MARGIN) continue;
            int captured = pos.piece_on(to_sq(m));
            int old_castle = pos.castling_rights(), old_ep = pos.ep_sq(), old_fifty = pos.halfmove_clock();
            // --- FIX: set current_move before making the move ---
            stack[ply].current_move = m;
            pos.make_move(m);
            if (pos.mover_in_check()) { pos.undo_move(m, captured, old_castle, old_ep, old_fifty); continue; }

#ifdef DEBUG
            // --- DEBUG: per-move diagnostics ---
            {
                int f = from_sq(m), t = to_sq(m);
                char ms[8];
                sprintf(ms, "%c%d%c%d", 'a' + (f & 7), 1 + (f >> 3), 'a' + (t & 7), 1 + (t >> 3));
                int cap = pos.piece_on(to_sq(m)) ? (pos.piece_on(to_sq(m)) & 7) : 0;
                fprintf(stderr, "DBG_MOVE_ENTER ply=%d depth=%d move=%s cap=%d in_check_before=%d\n",
                        ply, depth, ms, cap, in_check ? 1 : 0);
            }
#endif

            nodes++;
            Depth new_depth = depth - 1;
            if (in_check) new_depth++;
            std::vector<Move> child_pv;
            Value score;
            if (i == 0) {
                score = -negamax(pos, new_depth, -beta, -alpha, ply+1, true, child_pv);
            } else {
                int red;
                if (captured) {                     // no reduction for captures
                    red = 0;
                } else {
                    red = reduction(improving, depth, i, scored[i].score);
                }

#ifdef DEBUG
                // --- DEBUG: reduction decision ---
                {
                    int f = from_sq(m), t = to_sq(m);
                    char ms[8];
                    sprintf(ms, "%c%d%c%d", 'a' + (f & 7), 1 + (f >> 3), 'a' + (t & 7), 1 + (t >> 3));
                    fprintf(stderr, "DBG_REDUCE ply=%d depth=%d move=%s captured=%d raw_score=%d red=%d new_depth=%d\n",
                            ply, depth, ms, captured ? 1 : 0, scored[i].score, red, new_depth - red);
                }
#endif

                score = -negamax(pos, new_depth - red, -alpha-1, -alpha, ply+1, true, child_pv);
                if (score > alpha && score < beta)
                    score = -negamax(pos, new_depth, -beta, -alpha, ply+1, true, child_pv);
            }
            pos.undo_move(m, captured, old_castle, old_ep, old_fifty);

#ifdef DEBUG
            // --- DEBUG: move result ---
            {
                int f = from_sq(m), t = to_sq(m);
                char ms[8];
                sprintf(ms, "%c%d%c%d", 'a' + (f & 7), 1 + (f >> 3), 'a' + (t & 7), 1 + (t >> 3));
                fprintf(stderr, "DBG_MOVE_RESULT ply=%d depth=%d move=%s score=%d best_score=%d\n",
                        ply, depth, ms, score, best_score);
            }
#endif

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
                        if (!captured) {
                            if (stack[ply].killers[0] != m) {
                                stack[ply].killers[1] = stack[ply].killers[0];
                                stack[ply].killers[0] = m;
                            }
                            update_history(m, depth, true, pos);
                            for (int j = 0; j < i; ++j) {
                                if (!pos.piece_on(to_sq(scored[j].move))) {
                                    update_history(scored[j].move, depth, false, pos);
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
                        }
                        break;
                    }
                }
            }
        }
        // FIX: detect checkmate/stalemate when no legal move found
        if (best_score == -INF) {
            best_score = in_check ? -MATE_SCORE + ply : 0;
            bound = BOUND_EXACT;
            best_move = NO_MOVE;
        }
        // --- FIX: store mate scores as distance + offset ---
        Value store_score = best_score;
        if (std::abs(store_score) > MATE_SCORE - MAX_PLY) {
            int dist = (store_score > 0) ? MATE_SCORE - store_score : store_score + MATE_SCORE;
            store_score = (store_score > 0) ? MATE_OFFSET + dist : -(MATE_OFFSET + dist);
        }
        tt.store(key, depth, store_score, bound, best_move);
        // --- FIX: atomic compare-exchange for shared best move ---
        if (best_move != NO_MOVE) {
            Value prev = shared_best_score.load(std::memory_order_relaxed);
            while (best_score > prev && !shared_best_score.compare_exchange_weak(prev, best_score, std::memory_order_relaxed)) {
                // prev updated automatically
            }
            if (best_score >= shared_best_score.load(std::memory_order_relaxed))
                shared_best_move.store(best_move, std::memory_order_relaxed);
        }
        return best_score;
    }

    void output_info(int depth, Value score, const std::vector<Move>& pv) {
        int64_t elapsed = tm.elapsed();
        uint64_t nps = elapsed > 0 ? nodes * 1000 / elapsed : 0;
        std::string score_str;
        // --- FIX: correct mate distance ---
        if (std::abs(score) > MATE_SCORE - 1000) {
            int mate_dist = (score > 0) ? (MATE_SCORE - score) : (MATE_SCORE + score);
            if (mate_dist < 0) mate_dist = 0;
            score_str = (score > 0) ? ("mate " + std::to_string(mate_dist)) : ("mate -" + std::to_string(mate_dist));
        } else {
            score_str = "cp " + std::to_string(score);
        }
        std::cout << "info depth " << depth << " " << score_str
                  << " nodes " << nodes << " nps " << nps
                  << " time " << elapsed << " tbhits " << tb_hits << " pv";
        Position tmp = root_pos;
        for (Move m : pv) {
            std::cout << " " << char('a' + file_of(from_sq(m))) << char('1' + rank_of(from_sq(m)))
                     << char('a' + file_of(to_sq(m))) << char('1' + rank_of(to_sq(m)));
            tmp.make_move(m);
        }
        std::cout << std::endl;
    }

#ifdef DEBUG
    // DEBUG helper: returns score for single root move m searched to given depth
    Value debug_search_one_move(const Position& root_pos, Move m, Depth depth) {
        Position pos2 = root_pos;
        pos2.make_move(m);
        if (pos2.mover_in_check()) {
            fprintf(stderr, "DBG_DEBUG_FORCE move illegal after make: move illegal\n");
            return -INF;
        }
        std::vector<Move> pv;
        Value v = -negamax(const_cast<Position&>(pos2), depth - 1, -INF, INF, 1, true, pv);
        fprintf(stderr, "DBG_DEBUG_FORCE move=%d score=%d\n", m, v);
        return v;
    }
#endif

public:
    SearchThread(int id, Position& pos, TranspositionTable& t, SyzygyTablebase& tbb, Evaluation& e, OpeningBook* b)
        : thread_id(id), root_pos(pos), tt(t), tb(tbb), eval(e), book(b) {
        memset(history, 0, sizeof(history));
        memset(counter_moves, 0, sizeof(counter_moves));
        memset(follow_up_moves, 0, sizeof(follow_up_moves));
        memset(threat_history, 0, sizeof(threat_history));
        multi_pv = 1;
        // --- FIX: initialize stack entries ---
        stack.resize(MAX_PLY);
        for (int i = 0; i < MAX_PLY; ++i) {
            stack[i].killers[0] = stack[i].killers[1] = NO_MOVE;
            stack[i].counter = NO_MOVE;
            stack[i].ply = i;
            stack[i].static_eval = 0;
            stack[i].in_check = false;
            stack[i].current_move = NO_MOVE;
            stack[i].excluded_move = -1;
        }
    }
    void set_multi_pv(int mpv) { multi_pv = mpv; }
    void search(int max_depth, uint64_t max_nodes) {
        stop_search = false;
        nodes = 0;
        node_limit = max_nodes;
        tb_hits = 0;
        tt.new_search();

        Move moves[MAX_MOVES];
        int cnt = generate_moves(root_pos, moves);
        if (cnt == 0) return;

        // Filter out illegal root moves and king captures
        std::vector<ScoredMove> root_moves;
        for (int i = 0; i < cnt; ++i) {
            Move m = moves[i];
            // Skip if move would capture the opponent's king
            if (root_pos.piece_on(to_sq(m)) && ((root_pos.piece_on(to_sq(m)) & 7) == KING))
                continue;
            int cap = root_pos.piece_on(to_sq(m));
            int oc  = root_pos.castling_rights();
            int oe  = root_pos.ep_sq();
            int of_ = root_pos.halfmove_clock();
            Position tmp = root_pos;
            tmp.make_move(m);
            if (!tmp.mover_in_check())
                root_moves.push_back({m, 0});
        }
        if (root_moves.empty()) return;

        Move best_move = root_moves[0].move;
        Value best_score = -INF;

        for (int depth = 1; depth <= max_depth && !stop_search; ++depth) {
            if (depth > 1 && !tm.time_for_depth(depth)) break;

            // Re‑score root moves
            for (auto& sm : root_moves)
                sm.score = score_move(sm.move, 0, (best_move != NO_MOVE ? best_move : NO_MOVE), root_pos);
            if (best_move != NO_MOVE) {
                for (auto& sm : root_moves)
                    if (sm.move == best_move) { sm.score = 10000000; break; }
            }
            std::sort(root_moves.begin(), root_moves.end(),
                [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });

            // Aspiration windows
            Value alpha_orig = (depth >= 5) ? (best_score - ASPIRATION_WINDOW) : -INF;
            Value beta_orig  = (depth >= 5) ? (best_score + ASPIRATION_WINDOW) : INF;
            Value alpha = alpha_orig, beta = beta_orig;

            Move  depth_best  = NO_MOVE;
            Value depth_score = -INF;

            bool need_retry = true;
            while (need_retry && !stop_search) {
                need_retry  = false;
                depth_best  = NO_MOVE;
                depth_score = -INF;
                Value window_alpha = alpha;

                for (int i = 0; i < (int)root_moves.size() && !stop_search; ++i) {
                    Move m = root_moves[i].move;
                    // Skip if move would capture the opponent's king (redundant but safe)
                    if (root_pos.piece_on(to_sq(m)) && ((root_pos.piece_on(to_sq(m)) & 7) == KING))
                        continue;
                    int cap = root_pos.piece_on(to_sq(m));
                    int oc  = root_pos.castling_rights(), oe = root_pos.ep_sq(), of_ = root_pos.halfmove_clock();
                    Position pos2 = root_pos;
                    // --- FIX: set stack[0].current_move before calling negamax ---
                    stack[0].current_move = m;
                    pos2.make_move(m);
                    if (pos2.mover_in_check()) continue;

#ifdef DEBUG
                    // --- DEBUG: root move instrumentation ---
                    {
                        int f = from_sq(m), t = to_sq(m);
                        char ms[8];
                        sprintf(ms, "%c%d%c%d", 'a' + (f & 7), 1 + (f >> 3), 'a' + (t & 7), 1 + (t >> 3));
                        int cap_now = root_pos.piece_on(to_sq(m)) ? (root_pos.piece_on(to_sq(m)) & 7) : 0;
                        fprintf(stderr, "DBG_ROOT_MOVE [%s] cap=%d mover_in_check_after_make=%d\n",
                                ms, cap_now, pos2.mover_in_check() ? 1 : 0);
                    }
#endif

                    nodes++;
                    std::vector<Move> pv_line;
                    Value score;
                    if (i == 0 || window_alpha == -INF) {
                        score = -negamax(pos2, depth - 1, -beta, -window_alpha, 1, true, pv_line);
                    } else {
                        score = -negamax(pos2, depth - 1, -window_alpha - 1, -window_alpha, 1, true, pv_line);
                        if (!stop_search && score > window_alpha && score < beta)
                            score = -negamax(pos2, depth - 1, -beta, -window_alpha, 1, true, pv_line);
                    }
                    if (stop_search) break;

                    if (score > depth_score) {
                        depth_score = score;
                        depth_best  = m;
                        pv_line.insert(pv_line.begin(), m);
                        if (thread_id == 0)
                            output_info(depth, depth_score, pv_line);
                    }
                    if (score > window_alpha) window_alpha = score;
                }

                // Widen aspiration on fail
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
                best_move  = depth_best;
                best_score = depth_score;
                prev_eval  = best_score;
                shared_best_move.store(best_move, std::memory_order_relaxed);
                shared_best_score.store(best_score, std::memory_order_relaxed);
            }
        }
    }
};

// ============================================================================
// UCI Interface
// ============================================================================
class UCI {
private:
    Position pos;
    TranspositionTable tt;
    SyzygyTablebase tb;
    Evaluation eval;
    OpeningBook book;
    std::vector<std::thread> threads;
    std::atomic<bool> searching;
    int thread_count;
    int multi_pv;
    bool ponder;
    int contempt;

public:
    UCI() : tt(256), thread_count(1), multi_pv(1), searching(false), ponder(false), contempt(0) {
        Position::init_zobrist();
        Bitboards::init();
        init_magics();
    }
    void set_option(const std::string& name, const std::string& value) {
        if (name == "Hash") { tt.resize(std::stoi(value)); }
        else if (name == "Threads") thread_count = std::min(std::stoi(value), MAX_THREADS);
        else if (name == "Ponder") ponder = (value == "true");
        else if (name == "OwnBook") { if (value == "false") book = OpeningBook(); }
        else if (name == "BookFile") { if (!value.empty()) book.load(value); }
        else if (name == "BookVariety") book.set_variety(std::stod(value));
        else if (name == "SyzygyPath") { if (!value.empty()) tb.init(value); }
        else if (name == "EvalFile") { 
#ifdef USE_NNUE
            eval.set_nnue(value); 
#endif
        }
        else if (name == "MultiPV") multi_pv = std::stoi(value);
        else if (name == "Contempt") contempt = std::stoi(value);
        else if (name == "Clear Hash") tt.clear();
        else if (name == "Move Overhead") tm.set_move_overhead(std::stoi(value));
        else if (name == "UCI_Chess960") { /* ignore */ }
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
        if (i < args.size() && args[i] == "moves") {
            i++;
            while (i < args.size()) {
                std::string ms = args[i++];
                Square from = make_square(ms[0]-'a', ms[1]-'1');
                Square to = make_square(ms[2]-'a', ms[3]-'1');
                Move move = make_move(from, to);
                if (ms.length() == 5) {
                    char p = ms[4];
                    if (p == 'n') move = make_promotion(from, to, KNIGHT);
                    else if (p == 'b') move = make_promotion(from, to, BISHOP);
                    else if (p == 'r') move = make_promotion(from, to, ROOK);
                    else move = make_promotion(from, to, QUEEN);
                }
                int captured = pos.piece_on(to_sq(move));
                int old_castle = pos.castling_rights(), old_ep = pos.ep_sq(), old_fifty = pos.halfmove_clock();
                pos.make_move(move);
            }
        }
    }
    void go(const std::vector<std::string>& args) {
        int depth = 10;
        uint64_t nodes = 0;
        int64_t wtime = 0, btime = 0, winc = 0, binc = 0;
        int movestogo = 0, movetime = 0;
        bool infinite = false, ponder = false;
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
            else if (args[i] == "ponder") ponder = true;
        }
        if (!infinite && movetime == 0 && wtime == 0 && btime == 0) infinite = true;
        tm.set_side(pos.side_to_move(), wtime, btime, winc, binc, movestogo, movetime, infinite);
        Move book_move = book.probe(pos);
        if (book_move != NO_MOVE && !infinite && !ponder) {
            std::cout << "bestmove " << char('a' + file_of(from_sq(book_move))) << char('1' + rank_of(from_sq(book_move)))
                     << char('a' + file_of(to_sq(book_move))) << char('1' + rank_of(to_sq(book_move))) << std::endl;
            return;
        }
        if (tb.can_probe(pos)) {
            std::vector<unsigned> results;
            if (tb.probe_root(pos, results) > 0) {
                Move tb_move = tb.best_move_from_results(pos, results);
                if (tb_move != NO_MOVE) {
                    std::cout << "bestmove " << char('a' + file_of(from_sq(tb_move))) << char('1' + rank_of(from_sq(tb_move)))
                             << char('a' + file_of(to_sq(tb_move))) << char('1' + rank_of(to_sq(tb_move))) << std::endl;
                    return;
                }
            }
        }
        searching = true;
        shared_best_move = NO_MOVE;
        shared_best_score = -INF;
        for (int i = 0; i < thread_count; ++i) {
            threads.emplace_back([this, i, depth, nodes]() {
                SearchThread st(i, pos, tt, tb, eval, &book);
                st.set_multi_pv(multi_pv);
                st.search(depth, nodes);
            });
        }
        for (auto& t : threads) if (t.joinable()) t.join();
        threads.clear();
        Move best = shared_best_move.load();
        if (best == NO_MOVE) {
            Move moves[256];
            int cnt = generate_moves(pos, moves);
            if (cnt > 0) best = moves[0];
        }
        std::cout << "bestmove " << char('a' + file_of(from_sq(best))) << char('1' + rank_of(from_sq(best)))
                 << char('a' + file_of(to_sq(best))) << char('1' + rank_of(to_sq(best))) << std::endl;
        searching = false;
    }
    void stop() {
        stop_search = true;
        for (auto& t : threads) if (t.joinable()) t.join();
        threads.clear();
        searching = false;
    }
    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            if (token == "uci") {
                std::cout << "id name Hugine Ultimate Final\n";
                std::cout << "id author Hugine\n";
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
                std::cout << "option name Clear Hash type button\n";
                std::cout << "uciok\n";
            } else if (token == "isready") {
                std::cout << "readyok\n";
            } else if (token == "ucinewgame") {
                pos.init_startpos();
                tt.clear();
            } else if (token == "setoption") {
                std::string name, value, word;
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
                // ignore
            } else if (token == "quit") {
                stop();
                break;
            } else if (token == "d") {
                std::cout << pos.fen() << std::endl;
                for (int r = 7; r >= 0; --r) {
                    for (int f = 0; f < 8; ++f) {
                        Square sq = make_square(f, r);
                        int pc = pos.piece_on(sq);
                        if (pc == 0) std::cout << ". ";
                        else {
                            char p = " pnbrqk"[pc & 7];
                            if ((pc >> 3) == WHITE) p = toupper(p);
                            std::cout << p << ' ';
                        }
                    }
                    std::cout << std::endl;
                }
            } else if (token == "eval") {
                std::cout << "Evaluation: " << eval.evaluate(pos) << " cp\n";
            }
        }
    }
};

int main() {
    UCI uci;
    uci.run();
    return 0;
}