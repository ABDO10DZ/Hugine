#!/usr/bin/env bash
# =============================================================================
# Hugine 5.0 "Iota" — Universal Build Script
# =============================================================================
# Works on: Linux, macOS, Android/Termux, Windows (Git Bash / WSL / MSYS2)
# Fathom:   https://github.com/jdart1/Fathom  (NOT the deprecated SF fork)
#
# Usage:
#   ./builder.sh                   auto-detect everything, best settings
#   ./builder.sh --syzygy          force Syzygy on
#   ./builder.sh --no-syzygy       force Syzygy off
#   ./builder.sh --nnue            force NNUE on
#   ./builder.sh --no-nnue         force NNUE off
#   ./builder.sh --debug           debug build
#   ./builder.sh --portable        no -march=native (cross-compile safe)
#   ./builder.sh --compiler=clang++ use specific compiler
#   ./builder.sh --exe=myengine    custom output binary name
#   ./builder.sh --help            print this message
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Colours (disabled on dumb terminals / Windows CMD)
# --------------------------------------------------------------------------
if [[ "${TERM:-}" != "dumb" ]] && [[ "${OS:-}" != "Windows_NT" ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''
fi

info()  { echo -e "${CYAN}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
error() { echo -e "${RED}[error]${RESET} $*" >&2; }
die()   { error "$*"; exit 1; }

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
OPT_SYZYGY=""     # empty = auto-detect
OPT_NNUE=""       # empty = auto-detect
OPT_DEBUG=0
OPT_PORTABLE=0
OPT_COMPILER=""   # empty = auto-detect
OPT_EXE=""        # empty = auto-detect from OS
SRC="hugine.cpp"

# --------------------------------------------------------------------------
# Parse arguments
# --------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --syzygy)       OPT_SYZYGY=1 ;;
        --no-syzygy)    OPT_SYZYGY=0 ;;
        --nnue)         OPT_NNUE=1 ;;
        --no-nnue)      OPT_NNUE=0 ;;
        --debug)        OPT_DEBUG=1 ;;
        --portable)     OPT_PORTABLE=1 ;;
        --compiler=*)   OPT_COMPILER="${arg#--compiler=}" ;;
        --exe=*)        OPT_EXE="${arg#--exe=}" ;;
        --help|-h)
            sed -n '3,20p' "$0" | sed 's/^# //'
            exit 0 ;;
        *) warn "Unknown argument: $arg  (use --help)" ;;
    esac
done

# --------------------------------------------------------------------------
# Detect OS
# --------------------------------------------------------------------------
info "Detecting build environment ..."

OS_NAME="unknown"
case "$(uname -s 2>/dev/null || echo Windows)" in
    Linux)
        if [[ -d /data/data/com.termux ]] || [[ "${PREFIX:-}" == *termux* ]]; then
            OS_NAME="android"
        else
            OS_NAME="linux"
        fi ;;
    Darwin)               OS_NAME="macos" ;;
    MINGW*|MSYS*|CYGWIN*) OS_NAME="windows" ;;
    Windows*)             OS_NAME="windows" ;;
    *)                    OS_NAME="posix" ;;
esac

# --------------------------------------------------------------------------
# Detect Architecture
# --------------------------------------------------------------------------
ARCH="$(uname -m 2>/dev/null || echo unknown)"
case "$ARCH" in
    x86_64|amd64)       ARCH_TAG="x86_64" ;;
    aarch64|arm64)      ARCH_TAG="arm64"  ;;
    armv7*|armv8*|arm*) ARCH_TAG="arm32"  ;;
    *)                  ARCH_TAG="unknown" ;;
esac

info "OS: ${BOLD}${OS_NAME}${RESET}  Arch: ${BOLD}${ARCH_TAG}${RESET}"

# --------------------------------------------------------------------------
# Detect compiler
# --------------------------------------------------------------------------
if [[ -n "$OPT_COMPILER" ]]; then
    CXX="$OPT_COMPILER"
    command -v "$CXX" >/dev/null 2>&1 || die "Compiler '$CXX' not found in PATH"
    ok "Using compiler: $CXX (user-specified)"
else
    CXX=""
    for candidate in clang++ g++ c++; do
        if command -v "$candidate" >/dev/null 2>&1; then
            CXX="$candidate"
            ok "Compiler auto-selected: $CXX"
            break
        fi
    done
    [[ -z "$CXX" ]] && die "No C++ compiler found. Install g++ or clang++."
fi

# C compiler for Fathom (falls back to CXX if cc not available)
CC="${CC:-cc}"
command -v "$CC" >/dev/null 2>&1 || CC="$CXX"

# --------------------------------------------------------------------------
# Optimisation flags
# --------------------------------------------------------------------------
if [[ "$OPT_DEBUG" -eq 1 ]]; then
    OPT_FLAGS="-O0 -g -DDEBUG"
    info "Debug build enabled"
else
    OPT_FLAGS="-O3 -DNDEBUG"
fi

# Architecture-specific flags
ARCH_FLAGS=""
if [[ "$OPT_PORTABLE" -eq 0 ]]; then
    if [[ "$OS_NAME" == "android" ]]; then
        # -march=native on Termux/ARM can emit instructions the Android kernel
        # rejects at runtime; omit it and let the compiler pick a safe default.
        warn "-march=native skipped on Android (safe default used)"
    elif [[ "$ARCH_TAG" == "x86_64" ]]; then
        ARCH_FLAGS="-march=native"
    fi
fi

# --------------------------------------------------------------------------
# Detect jdart1/Fathom
# --------------------------------------------------------------------------
# Expected layout after:
#   git clone https://github.com/jdart1/Fathom Fathom && make -C Fathom
#
#   Fathom/Makefile
#   Fathom/src/tbprobe.h     <- header (static, ships with source)
#   Fathom/src/tbprobe.c
#   Fathom/src/tbconfig.h    <- also static, no need to verify separately
#   Fathom/src/stdendian.h
#   Fathom/obj/tbprobe.o     <- produced by make -C Fathom  (we link this)
#   Fathom/libfathom.so      <- also produced (not used here)
# --------------------------------------------------------------------------
FATHOM_HDR="Fathom/src/tbprobe.h"
FATHOM_OBJ="Fathom/obj/tbprobe.o"
USE_SYZYGY=0

if [[ "$OPT_SYZYGY" == "1" ]]; then
    [[ -f "$FATHOM_HDR" ]] || die \
"--syzygy requested but $FATHOM_HDR not found.
Clone jdart1 Fathom first:
  git clone https://github.com/jdart1/Fathom Fathom
Then re-run this script."
    USE_SYZYGY=1
    info "Syzygy: forced ON"
elif [[ "$OPT_SYZYGY" == "0" ]]; then
    USE_SYZYGY=0
    info "Syzygy: forced OFF"
else
    if [[ -f "$FATHOM_HDR" ]]; then
        USE_SYZYGY=1
        ok "Syzygy: auto-detected Fathom at $FATHOM_HDR -> enabling"
    else
        USE_SYZYGY=0
        info "Syzygy: Fathom not found -> building without"
        info "        (to enable: git clone https://github.com/jdart1/Fathom Fathom)"
    fi
fi

# --------------------------------------------------------------------------
# Build Fathom via its own Makefile  ->  produces Fathom/obj/tbprobe.o
# --------------------------------------------------------------------------
if [[ "$USE_SYZYGY" -eq 1 ]]; then

    [[ -f "Fathom/Makefile" ]] || die \
"Fathom/Makefile not found.
Clone with:  git clone https://github.com/jdart1/Fathom Fathom"

    info "Building jdart1 Fathom via its own Makefile ..."

    # Extra CFLAGS for glibc / musl targets that need _DEFAULT_SOURCE for mmap
    FATHOM_CFLAGS="-O2"
    if [[ "$OS_NAME" == "android" || "$OS_NAME" == "linux" ]]; then
        FATHOM_CFLAGS="$FATHOM_CFLAGS -D_DEFAULT_SOURCE"
    fi

    # Fathom's Makefile creates Fathom/obj/ and compiles tbprobe.c into it.
    # We only pass CC + CFLAGS; everything else is handled by its Makefile.
    if ! make -C Fathom CC="$CC" CFLAGS="$FATHOM_CFLAGS" 2>&1; then
        die "Fathom build failed. Run 'make -C Fathom' manually to see full errors."
    fi

    # Confirm the object file we are about to link
    [[ -f "$FATHOM_OBJ" ]] || die \
"Fathom built but $FATHOM_OBJ was not produced.
Check 'ls Fathom/obj/' and compare with Fathom/Makefile output paths."

    ok "Fathom ready: $FATHOM_OBJ"

    # -IFathom/src resolves the bare  #include "tbprobe.h"  in the engine.
    # No symlink needed — that approach caused the case-sensitive-path warning.
    SYZYGY_FLAGS="-DUSE_SYZYGY -IFathom/src"
    SYZYGY_OBJS="$FATHOM_OBJ"

else
    SYZYGY_FLAGS="-DNO_SYZYGY"
    SYZYGY_OBJS=""
fi

# --------------------------------------------------------------------------
# Detect NNUE network file
# --------------------------------------------------------------------------
USE_NNUE=0
NNUE_FLAGS=""

if [[ "$OPT_NNUE" == "1" ]]; then
    USE_NNUE=1
    info "NNUE: forced ON"
elif [[ "$OPT_NNUE" == "0" ]]; then
    USE_NNUE=0
    info "NNUE: forced OFF"
else
    NNUE_FOUND="$(ls *.nnue nn-*.bin 2>/dev/null | head -1 || true)"
    if [[ -n "$NNUE_FOUND" ]]; then
        USE_NNUE=1
        ok "NNUE: auto-detected $NNUE_FOUND -> enabling"
    else
        USE_NNUE=0
        info "NNUE: no .nnue / nn-*.bin found -> classical eval"
    fi
fi

[[ "$USE_NNUE" -eq 1 ]] && NNUE_FLAGS="-DUSE_NNUE" || NNUE_FLAGS=""

# --------------------------------------------------------------------------
# Output binary name
# --------------------------------------------------------------------------
mkdir -p build
if [[ -n "$OPT_EXE" ]]; then
    EXE="$OPT_EXE"
elif [[ "$OS_NAME" == "windows" ]]; then
    EXE="build/hugine.exe"
else
    EXE="build/hugine"
fi

# --------------------------------------------------------------------------
# Threading flags
# --------------------------------------------------------------------------
THREAD_FLAGS="-pthread"
if [[ "$OS_NAME" == "macos" ]]; then
    THREAD_LIBS=""        # macOS links pthread automatically
else
    THREAD_LIBS="-lpthread"
fi

# --------------------------------------------------------------------------
# Verify source file
# --------------------------------------------------------------------------
[[ -f "$SRC" ]] || die "Source file '$SRC' not found. Run builder.sh from the hugine directory."

# --------------------------------------------------------------------------
# Assemble and run the compile command
# --------------------------------------------------------------------------
CMD=("$CXX" -std=c++17)
for f in $OPT_FLAGS $ARCH_FLAGS $THREAD_FLAGS $SYZYGY_FLAGS $NNUE_FLAGS; do
    CMD+=("$f")
done
CMD+=("$SRC")
[[ -n "$SYZYGY_OBJS" ]] && CMD+=("$SYZYGY_OBJS")
CMD+=(-o "$EXE")
[[ -n "$THREAD_LIBS" ]] && CMD+=("$THREAD_LIBS")

echo ""
info "Build command:"
echo "  ${CMD[*]}"
echo ""

"${CMD[@]}" || die "Compilation failed."

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo ""
ok "Build successful!"
echo ""
echo -e "  ${BOLD}Binary  :${RESET} $EXE"
echo -e "  ${BOLD}OS      :${RESET} $OS_NAME"
echo -e "  ${BOLD}Arch    :${RESET} $ARCH_TAG"
echo -e "  ${BOLD}Compiler:${RESET} $CXX"
echo -e "  ${BOLD}Syzygy  :${RESET} $([[ $USE_SYZYGY -eq 1 ]] && echo 'yes (jdart1/Fathom)' || echo 'no')"
echo -e "  ${BOLD}NNUE    :${RESET} $([[ $USE_NNUE   -eq 1 ]] && echo 'yes' || echo 'no (classical eval)')"
echo -e "  ${BOLD}Debug   :${RESET} $([[ $OPT_DEBUG  -eq 1 ]] && echo 'yes' || echo 'no')"
echo ""
echo -e "  Run:  ./$EXE"
