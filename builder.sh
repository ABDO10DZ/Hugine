#!/usr/bin/env bash
# =============================================================================
# Hugine 5.1.0 "Iota" — Full Build Matrix Script
# =============================================================================
# Compiles all 11 standard variants in one shot.
# Every variant with -DUSE_NNUE supports loading ANY Stockfish .nnue file
# natively at runtime — no conversion required.
#
# Usage:
#   ./builder.sh                 build all 11 variants
#   ./builder.sh --clean         remove all built binaries first
#   ./builder.sh --test          run test suite after build
#   ./builder.sh --only=nnue     build only variants matching "nnue"
#   ./builder.sh --no-asan       skip the ASAN debug variant
#   ./builder.sh --help
#
# Output directory: ./build/
#
# Variants built:
#   hugine_base          -O2 no SIMD, classical eval
#   hugine_fast          -O3 -march=native, classical eval
#   hugine_pext          -O3 + BMI2 PEXT, classical eval
#   hugine_avx2          -O3 + AVX2 SIMD, classical eval
#   hugine_sse4          -O3 + SSE4.1 SIMD, classical eval
#   hugine_nnue          -O2 + NNUE (SF .nnue + Hugine-native)
#   hugine_fast_nnue     -O3 + BMI2 + NNUE
#   hugine_fast_nnue_avx2 -O3 + AVX2 + BMI2 + NNUE
#   hugine_large         -O2 + NNUE_LARGE (512-node architecture)
#   hugine_large_nnue    -O2 + NNUE_LARGE + NNUE
#   hugine_xl            -O2 + NNUE_XL    (1024-node architecture)
#   hugine_xl_nnue       -O2 + NNUE_XL   + NNUE
#   hugine_asan          -O0 + ASAN + UBSAN (memory safety)
# =============================================================================

set -euo pipefail

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
OPT_CLEAN=0 OPT_TEST=0 OPT_NO_ASAN=0 OPT_ONLY="" OPT_JOBS=0
SRC="hugine-iota-v510.cpp"
BUILDDIR="build"

for arg in "$@"; do
    case "$arg" in
        --clean)     OPT_CLEAN=1 ;;
        --test)      OPT_TEST=1 ;;
        --no-asan)   OPT_NO_ASAN=1 ;;
        --only=*)    OPT_ONLY="${arg#--only=}" ;;
        --jobs=*)    OPT_JOBS="${arg#--jobs=}" ;;
        --help|-h)   sed -n '3,25p' "$0" | sed 's/^# //'; exit 0 ;;
        *)           warn "Unknown: $arg" ;;
    esac
done

[[ "$OPT_CLEAN" -eq 1 ]] && rm -rf "$BUILDDIR" && info "Cleaned $BUILDDIR"
mkdir -p "$BUILDDIR"

# ── Detect compiler ───────────────────────────────────────────────────────────
for c in clang++ g++ c++; do
    command -v "$c" >/dev/null 2>&1 && CXX="$c" && break
done
[[ -z "${CXX:-}" ]] && die "No C++ compiler found. Install g++ or clang++."

OS_NAME="linux"
[[ "$(uname -s)" == "Darwin" ]] && OS_NAME="macos"
[[ -d /data/data/com.termux ]] || [[ "${PREFIX:-}" == *termux* ]] && OS_NAME="android"

ARCH="$(uname -m 2>/dev/null || echo unknown)"
[[ "$ARCH" =~ ^(aarch64|arm64)$ ]] && ARCH_TAG="arm64" || ARCH_TAG="x86_64"

[[ "$OS_NAME" == "macos" ]] && THREAD="-pthread" || THREAD="-pthread -lpthread"
BASE_WARN="-Wall -Wextra -Wno-unused-parameter"
NO_PEXT="-DNO_PEXT"

info "Compiler: $CXX  OS: $OS_NAME  Arch: $ARCH_TAG"
info "Source:   $SRC"
echo ""

# ── Build matrix ─────────────────────────────────────────────────────────────
# Format: "name|extra_flags"
declare -a VARIANTS=(
    "hugine_base|-O2 -std=c++17 -DNO_SYZYGY $NO_PEXT"
    "hugine_fast|-O3 -march=native -std=c++17 -DNDEBUG -DNO_SYZYGY $NO_PEXT"
    "hugine_pext|-O3 -march=native -mbmi2 -std=c++17 -DNDEBUG -DNO_SYZYGY"
    "hugine_avx2|-O3 -mavx2 -std=c++17 -DNDEBUG -DNO_SYZYGY $NO_PEXT"
    "hugine_sse4|-O3 -msse4.1 -std=c++17 -DNDEBUG -DNO_SYZYGY $NO_PEXT"
    "hugine_nnue|-O2 -std=c++17 -DUSE_NNUE -DNO_SYZYGY $NO_PEXT"
    "hugine_fast_nnue|-O3 -march=native -mbmi2 -std=c++17 -DUSE_NNUE -DNDEBUG -DNO_SYZYGY"
    "hugine_fast_nnue_avx2|-O3 -march=native -mavx2 -mbmi2 -std=c++17 -DUSE_NNUE -DNDEBUG -DNO_SYZYGY"
    "hugine_large|-O2 -std=c++17 -DNNUE_LARGE -DNO_SYZYGY $NO_PEXT"
    "hugine_large_nnue|-O2 -std=c++17 -DUSE_NNUE -DNNUE_LARGE -DNO_SYZYGY $NO_PEXT"
    "hugine_xl|-O2 -std=c++17 -DNNUE_XL -DNO_SYZYGY $NO_PEXT"
    "hugine_xl_nnue|-O2 -std=c++17 -DUSE_NNUE -DNNUE_XL -DNO_SYZYGY $NO_PEXT"
    "hugine_asan|-O0 -g -fsanitize=address,undefined -std=c++17 -DNO_SYZYGY $NO_PEXT"
)

PASS=0; FAIL=0; SKIP=0
START_TOTAL=$(date +%s%3N 2>/dev/null || date +%s)

for entry in "${VARIANTS[@]}"; do
    name="${entry%%|*}"
    flags="${entry#*|}"

    # Filter
    if [[ "$OPT_NO_ASAN" -eq 1 && "$name" == "hugine_asan" ]]; then
        warn "  SKIP  $name  (--no-asan)"
        ((SKIP++)); continue
    fi
    if [[ -n "$OPT_ONLY" && "$name" != *"$OPT_ONLY"* ]]; then
        ((SKIP++)); continue
    fi

    OUT="$BUILDDIR/$name"
    printf "  Building %-28s ... " "$name"

    START=$(date +%s%3N 2>/dev/null || date +%s)
    # shellcheck disable=SC2086
    if errs=$($CXX $flags $BASE_WARN $THREAD "$SRC" -o "$OUT" 2>&1); then
        END=$(date +%s%3N 2>/dev/null || date +%s)
        ELAPSED=$(( END - START ))
        # Count warnings
        NWARN=$(echo "$errs" | grep -c "warning:" || true)
        echo -e "${GREEN}ok${RESET}  (${ELAPSED}ms, ${NWARN} warnings)"
        ((PASS++))
    else
        echo -e "${RED}FAILED${RESET}"
        echo "$errs" | grep "error:" | head -5 | sed 's/^/    /'
        ((FAIL++))
    fi
done

echo ""
END_TOTAL=$(date +%s%3N 2>/dev/null || date +%s)
ELAPSED_TOTAL=$(( END_TOTAL - START_TOTAL ))

echo -e "  ${BOLD}Results:${RESET}  ${GREEN}${PASS} passed${RESET}  ${FAIL:+${RED}${FAIL} failed${RESET}}${SKIP:+  ${YELLOW}${SKIP} skipped${RESET}}"
echo -e "  ${BOLD}Total time:${RESET} ${ELAPSED_TOTAL}ms"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
    die "$FAIL variant(s) failed to compile."
fi

echo -e "  ${BOLD}All binaries in${RESET} ./$BUILDDIR/"
ls -lh "$BUILDDIR"/ | grep hugine | awk '{printf "    %-32s %s\n", $NF, $5}'
echo ""
echo -e "  ${CYAN}NNUE-capable variants:${RESET}"
echo    "    hugine_nnue, hugine_fast_nnue, hugine_fast_nnue_avx2, hugine_large_nnue, hugine_xl_nnue"
echo    "    All support Hugine-native .nnue + any Stockfish SF10–SF16+ .nnue"
echo    "    Load at runtime:  setoption name EvalFile value /path/to/nn-XXXX.nnue"
echo ""

if [[ "$OPT_TEST" -eq 1 ]]; then
    info "Running test suite …"
    python3 test_engine.py 2>&1 | tail -10
fi
