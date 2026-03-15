#!/usr/bin/env bash
# =============================================================================
# Hugine 5.1.0 "Iota" — Universal Build Script
# =============================================================================
# Works on: Linux, macOS, Android/Termux, Windows (Git Bash / WSL / MSYS2)
#
# NNUE support (new in 5.1.0):
#   Hugine now natively loads ANY Stockfish .nnue — no conversion tool needed.
#   Supported SF formats: HalfKP-256 (SF10-12), HalfKA-256 (SF13-14),
#   HalfKAv2-512/1024/1536 (SF15-16+), COMPRESSED_LEB128 (pytorch trainer).
#   Hugine-native .nnue (from nnue_trainer/train.py) also loads as before.
#
# Usage:
#   ./build.sh                     auto-detect everything
#   ./build.sh --syzygy            force Syzygy on
#   ./build.sh --no-syzygy         force Syzygy off
#   ./build.sh --nnue              force NNUE on
#   ./build.sh --no-nnue           force NNUE off
#   ./build.sh --nnue-large        Hugine 512-node architecture (larger net)
#   ./build.sh --pext              enable BMI2/PEXT bitboards (x86_64 only)
#   ./build.sh --avx2              force AVX2 SIMD path
#   ./build.sh --sse4              force SSE4.1 SIMD path
#   ./build.sh --pgo               2-pass profile-guided optimisation
#   ./build.sh --debug             ASAN + UBSAN debug build
#   ./build.sh --portable          no -march=native (cross-compile safe)
#   ./build.sh --compiler=clang++  pick a specific compiler
#   ./build.sh --exe=myengine      custom output name
#   ./build.sh --help              print this message
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
if [[ "${TERM:-}" != "dumb" ]] && [[ "${OS:-}" != "Windows_NT" ]]; then
    RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
    CYAN='\033[0;36m' BOLD='\033[1m' RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' RESET=''
fi

info()  { echo -e "${CYAN}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
error() { echo -e "${RED}[error]${RESET} $*" >&2; }
die()   { error "$*"; exit 1; }

# ── Defaults ─────────────────────────────────────────────────────────────────
OPT_SYZYGY="" OPT_NNUE="" OPT_NNUE_LARGE=0 OPT_NNUE_XL=0 OPT_PEXT=0
OPT_AVX2=0 OPT_SSE4=0 OPT_PGO=0 OPT_DEBUG=0 OPT_PORTABLE=0
OPT_COMPILER="" OPT_EXE=""
SRC="hugine-iota-v510.cpp"

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --syzygy)        OPT_SYZYGY=1 ;;
        --no-syzygy)     OPT_SYZYGY=0 ;;
        --nnue)          OPT_NNUE=1 ;;
        --no-nnue)       OPT_NNUE=0 ;;
        --nnue-large)    OPT_NNUE_LARGE=1 ;;
        --nnue-xl)       OPT_NNUE_XL=1; OPT_NNUE_LARGE=1 ;;
        --pext)          OPT_PEXT=1 ;;
        --avx2)          OPT_AVX2=1 ;;
        --sse4)          OPT_SSE4=1 ;;
        --pgo)           OPT_PGO=1 ;;
        --debug)         OPT_DEBUG=1 ;;
        --portable)      OPT_PORTABLE=1 ;;
        --compiler=*)    OPT_COMPILER="${arg#--compiler=}" ;;
        --exe=*)         OPT_EXE="${arg#--exe=}" ;;
        --help|-h)       sed -n '3,24p' "$0" | sed 's/^# //'; exit 0 ;;
        *)               warn "Unknown: $arg  (use --help)" ;;
    esac
done

# ── Detect OS + Arch ─────────────────────────────────────────────────────────
info "Detecting environment …"
OS_NAME="unknown"
case "$(uname -s 2>/dev/null || echo Windows)" in
    Linux) [[ -d /data/data/com.termux ]] || [[ "${PREFIX:-}" == *termux* ]] \
               && OS_NAME="android" || OS_NAME="linux" ;;
    Darwin) OS_NAME="macos" ;;
    MINGW*|MSYS*|CYGWIN*|Windows*) OS_NAME="windows" ;;
    *) OS_NAME="posix" ;;
esac
ARCH="$(uname -m 2>/dev/null || echo unknown)"
case "$ARCH" in
    x86_64|amd64)   ARCH_TAG="x86_64" ;;
    aarch64|arm64)  ARCH_TAG="arm64" ;;
    armv7*|arm*)    ARCH_TAG="arm32" ;;
    *)              ARCH_TAG="unknown" ;;
esac
info "OS: ${BOLD}${OS_NAME}${RESET}  Arch: ${BOLD}${ARCH_TAG}${RESET}"

# ── Compiler ─────────────────────────────────────────────────────────────────
if [[ -n "$OPT_COMPILER" ]]; then
    CXX="$OPT_COMPILER"
    command -v "$CXX" >/dev/null 2>&1 || die "Compiler '$CXX' not found"
    ok "Compiler: $CXX (user)"
else
    for c in clang++ g++ c++; do
        command -v "$c" >/dev/null 2>&1 && CXX="$c" && ok "Compiler: $c" && break
    done
    [[ -z "${CXX:-}" ]] && die "No C++ compiler. Install g++ or clang++."
fi
CC="${CC:-cc}"; command -v "$CC" >/dev/null 2>&1 || CC="$CXX"

# ── Optimisation ─────────────────────────────────────────────────────────────
if [[ "$OPT_DEBUG" -eq 1 ]]; then
    OPT_FLAGS="-O0 -g -fsanitize=address,undefined -DDEBUG"
    warn "Debug build (ASAN+UBSAN) — do not use for performance testing"
else
    OPT_FLAGS="-O3 -DNDEBUG"
fi

# ── Arch flags ────────────────────────────────────────────────────────────────
ARCH_FLAGS=""
if [[ "$OPT_PORTABLE" -eq 0 ]]; then
    if [[ "$OS_NAME" == "android" ]]; then
        warn "-march=native skipped on Android"
    elif [[ "$ARCH_TAG" == "x86_64" ]]; then
        ARCH_FLAGS="-march=native"
    fi
fi
[[ "$OPT_AVX2" -eq 1 ]] && ARCH_FLAGS+=" -mavx2"
[[ "$OPT_SSE4" -eq 1 ]] && ARCH_FLAGS+=" -msse4.1"
[[ "$OPT_PEXT" -eq 1 ]] && ARCH_FLAGS+=" -mbmi2"
PEXT_FLAGS=""; [[ "$OPT_PEXT" -eq 0 ]] && PEXT_FLAGS="-DNO_PEXT"

# ── Syzygy ────────────────────────────────────────────────────────────────────
FATHOM_HDR="Fathom/src/tbprobe.h"; FATHOM_SRC="Fathom/src/tbprobe.c"
FATHOM_OBJ="Fathom/src/tbprobe.o"; USE_SYZYGY=0
if [[ "$OPT_SYZYGY" == "1" ]]; then
    [[ -f "$FATHOM_HDR" ]] || die "--syzygy: Fathom not found. git clone https://github.com/jdart1/Fathom Fathom"
    USE_SYZYGY=1; info "Syzygy: ON (forced)"
elif [[ "$OPT_SYZYGY" == "0" ]]; then
    info "Syzygy: OFF (forced)"
elif [[ -f "$FATHOM_HDR" ]]; then
    USE_SYZYGY=1; ok "Syzygy: auto-detected"
else
    info "Syzygy: OFF (Fathom not found)"
fi

if [[ "$USE_SYZYGY" -eq 1 ]]; then
    info "Compiling Fathom …"
    FEXTRA=""; [[ "$OS_NAME" == android || "$OS_NAME" == linux ]] && FEXTRA="-D_DEFAULT_SOURCE"
    "$CC" -O2 -std=c11 $FEXTRA -c "$FATHOM_SRC" -o "$FATHOM_OBJ" \
        || die "Fathom compile failed. See SYZYGY_ANDROID_BUILD.md"
    ok "Fathom: $FATHOM_OBJ"
    SYZYGY_FLAGS="-DUSE_SYZYGY"; SYZYGY_OBJS="$FATHOM_OBJ"
else
    SYZYGY_FLAGS="-DNO_SYZYGY"; SYZYGY_OBJS=""
fi

# ── NNUE ─────────────────────────────────────────────────────────────────────
# In Hugine 5.1.0 the evaluator supports both Hugine-native .nnue files and
# any Stockfish .nnue directly — the loader auto-selects at runtime.
# Build flag -DUSE_NNUE only compiles in the evaluator; the actual network file
# is specified at runtime via UCI: setoption name EvalFile value <path>
USE_NNUE=0; NNUE_FLAGS=""; NNUE_NET=""

if [[ "$OPT_NNUE" == "1" ]]; then
    USE_NNUE=1; info "NNUE: ON (forced)"
elif [[ "$OPT_NNUE" == "0" ]]; then
    info "NNUE: OFF (forced)"
else
    NNUE_NET=$(ls *.nnue nn-*.bin 2>/dev/null | head -1 || true)
    if [[ -n "$NNUE_NET" ]]; then
        USE_NNUE=1; ok "NNUE: found ${NNUE_NET}"
        # Identify the format
        python3 -c "
import struct, sys
try:
    with open('$NNUE_NET','rb') as f: d=f.read(96)
    if len(d)<8: sys.exit()
    ver,arch = struct.unpack_from('<II',d,0)
    if (ver & 0xFF000000)==0x7A000000:
        print('  → Stockfish .nnue (will load natively via SFNNUEEvaluator)')
    else:
        print('  → Hugine-native .nnue')
except: pass
" 2>/dev/null || true
    else
        info "NNUE: no .nnue file found → classical eval"
        info "      Pass --nnue to compile the evaluator anyway (load net at runtime)"
    fi
fi

[[ "$USE_NNUE" -eq 1 ]] && NNUE_FLAGS="-DUSE_NNUE"
[[ "$OPT_NNUE_XL"    -eq 1 ]] && { NNUE_FLAGS+=" -DNNUE_XL";    info "NNUE_XL: 1024-node Hugine architecture"; }
[[ "$OPT_NNUE_LARGE" -eq 1 && "$OPT_NNUE_XL" -eq 0 ]] && { NNUE_FLAGS+=" -DNNUE_LARGE"; info "NNUE_LARGE: 512-node Hugine architecture"; }

# ── Output name ───────────────────────────────────────────────────────────────
[[ -n "$OPT_EXE" ]] && EXE="$OPT_EXE" \
    || { [[ "$OS_NAME" == "windows" ]] && EXE="hugine.exe" || EXE="hugine"; }

# ── Threading ─────────────────────────────────────────────────────────────────
THREAD_FLAGS="-pthread"
[[ "$OS_NAME" == "macos" ]] && THREAD_LIBS="" || THREAD_LIBS="-lpthread"

# ── Build ─────────────────────────────────────────────────────────────────────
COMPILE_FLAGS="-std=c++17 $OPT_FLAGS $ARCH_FLAGS $THREAD_FLAGS
               $SYZYGY_FLAGS $NNUE_FLAGS $PEXT_FLAGS
               -Wall -Wextra -Wno-unused-parameter"

echo ""
info "Compiling ${SRC} → ${EXE} …"

if [[ "$OPT_PGO" -eq 1 && "$OPT_DEBUG" -eq 0 ]]; then
    PGO_DIR="/tmp/hugine_pgo_$$"
    info "PGO pass 1/2: generating profile …"
    # shellcheck disable=SC2086
    $CXX $COMPILE_FLAGS -fprofile-generate="$PGO_DIR" \
        "$SRC" $SYZYGY_OBJS -o "${EXE}_pgo_gen" ${THREAD_LIBS}
    mkdir -p "$PGO_DIR"
    printf "position startpos\ngo depth 12\nquit\n" | ./"${EXE}_pgo_gen" >/dev/null 2>&1 || true
    printf "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\nperft 4\nquit\n" \
        | ./"${EXE}_pgo_gen" >/dev/null 2>&1 || true
    info "PGO pass 2/2: optimising with profile …"
    # shellcheck disable=SC2086
    $CXX $COMPILE_FLAGS -fprofile-use="$PGO_DIR" -fprofile-correction \
        "$SRC" $SYZYGY_OBJS -o "$EXE" ${THREAD_LIBS}
    rm -rf "$PGO_DIR" "${EXE}_pgo_gen"
else
    # shellcheck disable=SC2086
    $CXX $COMPILE_FLAGS "$SRC" $SYZYGY_OBJS -o "$EXE" ${THREAD_LIBS} \
        || die "Compilation failed."
fi

echo ""
ok "Build complete!"
echo ""
printf "  ${BOLD}%-10s${RESET} %s\n" "Binary:"   "$EXE"
printf "  ${BOLD}%-10s${RESET} %s\n" "OS:"       "$OS_NAME ($ARCH_TAG)"
printf "  ${BOLD}%-10s${RESET} %s\n" "Compiler:" "$CXX"
printf "  ${BOLD}%-10s${RESET} %s\n" "Syzygy:"   "$([ $USE_SYZYGY -eq 1 ] && echo 'yes (Fathom)' || echo 'no')"
printf "  ${BOLD}%-10s${RESET} %s\n" "NNUE:"     "$([ $USE_NNUE -eq 1 ] && echo 'yes  ← accepts Hugine-native + any Stockfish .nnue' || echo 'no')"
printf "  ${BOLD}%-10s${RESET} %s\n" "PGO:"      "$([ $OPT_PGO -eq 1 ] && echo 'yes (2-pass)' || echo 'no')"
printf "  ${BOLD}%-10s${RESET} %s\n" "Debug:"    "$([ $OPT_DEBUG -eq 1 ] && echo 'yes (ASAN+UBSAN)' || echo 'no')"
echo ""
echo -e "  Run:   ./$EXE"
if [[ "$USE_NNUE" -eq 1 ]]; then
    echo ""
    echo -e "  ${CYAN}NNUE usage (any Stockfish .nnue or Hugine-native .nnue):${RESET}"
    echo    "    setoption name EvalFile value /path/to/nn-XXXX.nnue"
    echo    "  Supported: HalfKP-256, HalfKA-256, HalfKAv2-512/1024/1536
             (Hugine-native: 256/512/1024-node --nnue/--nnue-large/--nnue-xl),"
    echo    "             COMPRESSED_LEB128 (SF16+ pytorch trainer output)"
fi
