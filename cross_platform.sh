#!/usr/bin/env bash
# =============================================================================
# Hugine 3.0 "Gama" — Cross-Platform Multi-Variant Build Script
# =============================================================================
# Produces up to 24 binaries (6 variants × 4 arches) in ./build/
#
#   Variants:
#     _base      O3, no Syzygy, no NNUE              (always built)
#     _syzygy    O3, Syzygy/Fathom                   (needs Fathom/)
#     _nnue      O3, NNUE eval                        (needs *.nnue file)
#     _full      O3, Syzygy + NNUE                    (needs both)
#     _chess960  O3, Chess960 debug tracing           (always built)
#     _debug     O0 -g, assertions on                 (always built)
#
#   Architectures:
#     hugine_winx86_{v}.exe    Windows x86-64
#     hugine_winARM_{v}.exe    Windows ARM64
#     hugine_linux86_{v}       Linux x86-64  (static)
#     hugine_linuxARM_{v}      Linux ARM64   (static)
#
# ─── How it works ────────────────────────────────────────────────────────────
#
# PATH A  x86_64 Linux host (desktop/CI):
#   • apt:       g++-mingw-w64-x86-64, g++-aarch64-linux-gnu
#   • llvm-mingw x86_64 build   → Windows ARM64 target
#   • musl.cc x86_64 cross-compilers → static Linux x86_64 + ARM64
#
# PATH B  ARM64 / Termux host:
#   • Boots proot-Ubuntu ARM64 rootfs (installs if missing; skips if present)
#   • Inside that ARM64 Ubuntu:
#       – llvm-mingw AARCH64 build  → Windows x86-64 AND ARM64 targets
#       – apt g++-x86-64-linux-gnu  → Linux x86-64 target (static glibc)
#       – native g++ -static        → Linux ARM64 target
#   NOTE: musl.cc is x86_64-hosted only — can't run inside ARM64 Ubuntu.
#
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

info()  { echo -e "${CYAN}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
die()   { echo -e "${RED}[error]${RESET} $*" >&2; exit 1; }
step()  { echo -e "\n${BOLD}▶ $*${RESET}"; }
skip()  { echo -e "${DIM}[skip]  $*${RESET}"; }

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SRC="$SCRIPT_DIR/hugine.cpp"
FATHOM_DIR="$SCRIPT_DIR/Fathom"
FATHOM_SRC="$FATHOM_DIR/src/tbprobe.c"
FATHOM_INC="$FATHOM_DIR/src"
NNUE_FILE="$(ls "$SCRIPT_DIR"/*.nnue "$SCRIPT_DIR"/nn-*.bin 2>/dev/null | head -1 || true)"

LLVM_VER="20260224"
LLVM_X86_URL="https://github.com/mstorsjo/llvm-mingw/releases/download/${LLVM_VER}/llvm-mingw-${LLVM_VER}-ucrt-ubuntu-22.04-x86_64.tar.xz"
MUSL_X86_URL="https://musl.cc/x86_64-linux-musl-cross.tgz"
MUSL_ARM_URL="https://musl.cc/aarch64-linux-musl-cross.tgz"
LLVM_DIR="$SCRIPT_DIR/toolchains/llvm-mingw"
MUSL_X86_DIR="$SCRIPT_DIR/toolchains/x86_64-linux-musl-cross"
MUSL_ARM_DIR="$SCRIPT_DIR/toolchains/aarch64-linux-musl-cross"

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
download() {
    local url="$1" dest="$2"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url"
    else
        die "Neither wget nor curl found."
    fi
}

build_fathom() {
    local cc="$1" obj="$2" extra="${3:-}"
    [[ ! -f "$FATHOM_SRC" ]] && { echo ""; return; }
    # shellcheck disable=SC2086
    "$cc" -O2 -D_DEFAULT_SOURCE $extra \
        -I"$FATHOM_INC" -c "$FATHOM_SRC" -o "$obj" 2>/dev/null \
        && echo "$obj" \
        || { warn "Fathom compile failed: $(basename "$obj")"; echo ""; }
}

compile() {
    local cxx="$1" out="$2" base_flags="$3" sflags="$4"
    local fobj="${5:-}" opt="${6:--O3 -DNDEBUG}" extra="${7:-}"
    local name; name="$(basename "$out")"

    if echo "$extra" | grep -q "USE_NNUE" && [[ -z "${NNUE_FILE:-}" ]]; then
        skip "$name  (no .nnue file)"; return
    fi

    info "  $name ..."
    local sobjs=""; [[ -n "$fobj" && -f "$fobj" ]] && sobjs="$fobj"

    # Windows (MinGW) doesn't have a separate libpthread — winpthread is bundled.
    local thread_flags="-pthread"
    local thread_libs="-lpthread"
    case "$name" in *.exe) thread_flags=""; thread_libs="" ;; esac

    local err_out
    # shellcheck disable=SC2086
    err_out=$( "$cxx" \
        -std=c++17 $opt \
        -Wall -Wextra -Wno-unused-parameter \
        $thread_flags $base_flags $sflags $extra \
        "$SRC" $sobjs -o "$out" $thread_libs 2>&1 ) || true

    if [[ -f "$out" ]]; then
        ok "    $(basename "$out")  ($(du -sh "$out" | cut -f1))"
    else
        warn "    FAILED: $name"
        echo "$err_out" | head -8 | sed 's/^/        /' || true
    fi
}

build_arch_variants() {
    local tag="$1" cxx="$2" cc="$3" cxxf="$4" cf="$5" ext="$6"
    step "Building $tag — 6 variants"

    local SFON="-DUSE_SYZYGY -I$FATHOM_INC"
    local SFOFF="-DNO_SYZYGY"
    local fr="" ff=""

    if [[ -f "$FATHOM_SRC" ]]; then
        # shellcheck disable=SC2086
        fr=$(build_fathom "$cc" "$BUILD_DIR/tb_${tag}_r.o" "$cf") || fr=""
        # shellcheck disable=SC2086
        ff=$(build_fathom "$cc" "$BUILD_DIR/tb_${tag}_f.o" "$cf") || ff=""
    fi

    compile "$cxx" "$BUILD_DIR/hugine_${tag}_base${ext}"     "$cxxf" "$SFOFF" ""   "-O3 -DNDEBUG"
    if [[ -f "$FATHOM_SRC" ]]; then
        compile "$cxx" "$BUILD_DIR/hugine_${tag}_syzygy${ext}" "$cxxf" "$SFON"  "$fr" "-O3 -DNDEBUG"
    else
        skip "hugine_${tag}_syzygy${ext}  (no Fathom)"
    fi
    compile "$cxx" "$BUILD_DIR/hugine_${tag}_nnue${ext}"     "$cxxf" "$SFOFF" ""   "-O3 -DNDEBUG" "-DUSE_NNUE"
    if [[ -f "$FATHOM_SRC" ]]; then
        compile "$cxx" "$BUILD_DIR/hugine_${tag}_full${ext}"   "$cxxf" "$SFON"  "$ff" "-O3 -DNDEBUG" "-DUSE_NNUE"
    else
        skip "hugine_${tag}_full${ext}  (no Fathom)"
    fi
    compile "$cxx" "$BUILD_DIR/hugine_${tag}_chess960${ext}" "$cxxf" "$SFOFF" ""   "-O3 -DNDEBUG" "-DCHESS960_EXTRA_DEBUG"
    compile "$cxx" "$BUILD_DIR/hugine_${tag}_debug${ext}"    "$cxxf" "$SFOFF" ""   "-O0 -g -DDEBUG"
}

print_summary() {
    echo ""
    echo -e "${BOLD}=== Build Matrix Results ===${RESET}"
    echo -e "Directory: ${BOLD}$BUILD_DIR${RESET}\n"
    local TOTAL=0 BUILT=0
    for arch in winx86 winARM linux86 linuxARM; do
        local ext=".exe"; [[ "$arch" == linux* ]] && ext=""
        echo -e "  ${BOLD}$arch${RESET}"
        for v in base syzygy nnue full chess960 debug; do
            TOTAL=$((TOTAL+1))
            local f="$BUILD_DIR/hugine_${arch}_${v}${ext}"
            if [[ -f "$f" ]]; then
                local sz; sz=$(du -sh "$f" | cut -f1)
                echo -e "    ${GREEN}✓${RESET}  hugine_${arch}_${v}${ext}  ($sz)"
                BUILT=$((BUILT+1))
            else
                echo -e "    ${DIM}–${RESET}  hugine_${arch}_${v}${ext}"
            fi
        done
        echo ""
    done
    echo -e "${BOLD}Built: $BUILT / $TOTAL${RESET}\n"
}

# --------------------------------------------------------------------------
# Detect host
# --------------------------------------------------------------------------
HOST_ARCH="$(uname -m)"
IS_TERMUX=0
[[ -d /data/data/com.termux || "${PREFIX:-}" == *termux* ]] && IS_TERMUX=1

info "Host arch : $HOST_ARCH  |  Termux: $IS_TERMUX"
[[ -f "$SRC" ]] || die "Source not found: $SRC"
[[ -n "$NNUE_FILE" ]] \
    && ok "NNUE     : $NNUE_FILE" \
    || warn "No .nnue / nn-*.bin — nnue/full variants will be skipped"
[[ -f "$FATHOM_SRC" ]] \
    && ok "Fathom   : $FATHOM_SRC" \
    || warn "Fathom not found — syzygy/full variants will be skipped"

mkdir -p "$BUILD_DIR" "$SCRIPT_DIR/toolchains"

# ============================================================================
# PATH A — x86_64 Linux host
# ============================================================================
if [[ "$HOST_ARCH" == "x86_64" && "$IS_TERMUX" -eq 0 ]]; then

    step "x86_64 Linux host — installing cross-compilation toolchains"
    command -v apt-get &>/dev/null || die "apt-get not found. Requires Debian/Ubuntu."

    info "apt-get: cross-compilers ..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
        gcc-mingw-w64-x86-64  g++-mingw-w64-x86-64 \
        2>/dev/null || warn "Some apt packages may have failed — continuing"

    if [[ ! -d "$LLVM_DIR" ]]; then
        step "Downloading llvm-mingw (x86_64 host) ..."
        download "$LLVM_X86_URL" "$SCRIPT_DIR/toolchains/llvm-mingw.tar.xz"
        mkdir -p "$LLVM_DIR"
        tar -xf "$SCRIPT_DIR/toolchains/llvm-mingw.tar.xz" -C "$LLVM_DIR" --strip-components=1
        rm -f "$SCRIPT_DIR/toolchains/llvm-mingw.tar.xz"
        ok "llvm-mingw ready"
    else
        ok "llvm-mingw already present"
    fi

    if [[ ! -d "$MUSL_X86_DIR" ]]; then
        step "Downloading musl x86_64 ..."
        download "$MUSL_X86_URL" "$SCRIPT_DIR/toolchains/x86-musl.tgz"
        tar -xf "$SCRIPT_DIR/toolchains/x86-musl.tgz" -C "$SCRIPT_DIR/toolchains/"
        rm -f "$SCRIPT_DIR/toolchains/x86-musl.tgz"
    else
        ok "musl x86_64 already present"
    fi
    if [[ ! -d "$MUSL_ARM_DIR" ]]; then
        step "Downloading musl aarch64 ..."
        download "$MUSL_ARM_URL" "$SCRIPT_DIR/toolchains/arm-musl.tgz"
        tar -xf "$SCRIPT_DIR/toolchains/arm-musl.tgz" -C "$SCRIPT_DIR/toolchains/"
        rm -f "$SCRIPT_DIR/toolchains/arm-musl.tgz"
    else
        ok "musl aarch64 already present"
    fi

    L="$LLVM_DIR/bin"
    MX="$MUSL_X86_DIR/bin"
    MA="$MUSL_ARM_DIR/bin"

    build_arch_variants "winx86" \
        "x86_64-w64-mingw32-g++" "x86_64-w64-mingw32-gcc" \
        "-static -static-libgcc -static-libstdc++" "" ".exe"

    build_arch_variants "winARM" \
        "$L/aarch64-w64-mingw32-clang++" "$L/aarch64-w64-mingw32-clang" \
        "--target=aarch64-w64-mingw32 -static" "--target=aarch64-w64-mingw32" ".exe"

    build_arch_variants "linux86" \
        "$MX/x86_64-linux-musl-g++" "$MX/x86_64-linux-musl-gcc" \
        "-static" "" ""

    build_arch_variants "linuxARM" \
        "$MA/aarch64-linux-musl-g++" "$MA/aarch64-linux-musl-gcc" \
        "-static" "" ""

    print_summary

# ============================================================================
# PATH B — ARM64 / Termux  →  proot Ubuntu ARM64 rootfs
# ============================================================================
else
    step "ARM64/Termux host — using proot Ubuntu for cross-compilation"

    if ! command -v proot-distro &>/dev/null; then
        info "Installing proot-distro ..."
        pkg install -y proot-distro || die "Failed to install proot-distro"
    fi
    ok "proot-distro available"

    # Check if rootfs directory already exists — don't rely on proot-distro list output
    ROOTFS=""
    for p in \
        "${PREFIX:-/usr}/var/lib/proot-distro/installed-rootfs/ubuntu" \
        "$HOME/../usr/var/lib/proot-distro/installed-rootfs/ubuntu" \
        "/data/data/com.termux/files/usr/var/lib/proot-distro/installed-rootfs/ubuntu"; do
        [[ -d "$p" ]] && { ROOTFS="$p"; break; }
    done

    if [[ -z "$ROOTFS" ]]; then
        info "Ubuntu rootfs not found — installing (one-time ~300 MB) ..."
        # proot-distro install may exit non-zero on some versions even when it worked
        proot-distro install ubuntu 2>&1 || true

        # Re-search
        for p in \
            "${PREFIX:-/usr}/var/lib/proot-distro/installed-rootfs/ubuntu" \
            "$HOME/../usr/var/lib/proot-distro/installed-rootfs/ubuntu" \
            "/data/data/com.termux/files/usr/var/lib/proot-distro/installed-rootfs/ubuntu"; do
            [[ -d "$p" ]] && { ROOTFS="$p"; break; }
        done
        [[ -z "$ROOTFS" ]] && die \
"Ubuntu install failed — rootfs directory not found.
Run manually:
  proot-distro install ubuntu
  proot-distro login ubuntu    # verify it works
Then rerun this script."
    else
        ok "Ubuntu rootfs already present — skipping install"
    fi
    ok "Rootfs: $ROOTFS"

    # Sync project files
    UB_HUGINE="$ROOTFS/root/hugine"
    step "Syncing project into Ubuntu rootfs ..."
    mkdir -p "$UB_HUGINE"
    cp -u "$SRC" "$UB_HUGINE/"
    [[ -d "$FATHOM_DIR" ]]                   && cp -ru "$FATHOM_DIR"  "$UB_HUGINE/"
    [[ -n "$NNUE_FILE" && -f "$NNUE_FILE" ]] && cp -u  "$NNUE_FILE"  "$UB_HUGINE/"
    ok "Files synced"

    # ------------------------------------------------------------------
    # Write inner build script.
    # This runs inside proot Ubuntu which is an aarch64 environment.
    # musl.cc ships only x86_64-hosted binaries — they cannot execute here.
    # Instead we use:
    #   llvm-mingw aarch64 host build  → all Windows targets (x86 + ARM)
    #   apt g++-x86-64-linux-gnu       → Linux x86-64 target
    #   native g++ -static             → Linux ARM64 target (we're on aarch64)
    # ------------------------------------------------------------------
    INNER="$ROOTFS/root/hugine_cross_build.sh"
    cat > "$INNER" << 'INNER_SCRIPT'
#!/usr/bin/env bash
# Runs INSIDE proot Ubuntu (aarch64 environment)
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
info()  { echo -e "${CYAN}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
die()   { echo -e "${RED}[error]${RESET} $*" >&2; exit 1; }
step()  { echo -e "\n${BOLD}▶ $*${RESET}"; }
skip()  { echo -e "${DIM}[skip]  $*${RESET}"; }

WORK="/root/hugine"
BUILD="$WORK/build"
TOOLS="/root/toolchains"
SRC="$WORK/hugine.cpp"
FATHOM_SRC="$WORK/Fathom/src/tbprobe.c"
FATHOM_INC="$WORK/Fathom/src"
NNUE_FILE="$(ls "$WORK"/*.nnue "$WORK"/nn-*.bin 2>/dev/null | head -1 || true)"

mkdir -p "$BUILD" "$TOOLS"

info "Inner environment: $(uname -m)  ($(uname -r))"
[[ "$(uname -m)" == "aarch64" ]] || warn "Expected aarch64 inside proot — got $(uname -m)"

# llvm-mingw aarch64-hosted release
LLVM_VER="20260224"
LLVM_ARM_URL="https://github.com/mstorsjo/llvm-mingw/releases/download/${LLVM_VER}/llvm-mingw-${LLVM_VER}-ucrt-ubuntu-22.04-aarch64.tar.xz"
LLVM_DIR="$TOOLS/llvm-mingw-aarch64"
LLVM_BIN="$LLVM_DIR/bin"

download() {
    local url="$1" dst="$2"
    command -v wget &>/dev/null \
        && wget -q --show-progress -O "$dst" "$url" \
        || curl -L --progress-bar -o "$dst" "$url"
}

# ── apt packages ─────────────────────────────────────────────────────────
step "apt-get: cross-compilers"
apt-get update -qq
apt-get install -y -qq \
    build-essential \
    gcc-x86-64-linux-gnu   g++-x86-64-linux-gnu \
    gcc-mingw-w64-x86-64   g++-mingw-w64-x86-64 \
    wget curl xz-utils ca-certificates \
    2>/dev/null || true
ok "apt packages done"

# ── llvm-mingw aarch64 host (needed only for Windows ARM64 target) ────────
# Check the actual binary, not just the directory — a partial/interrupted
# download leaves the dir behind and fools a simple -d check.
if [[ ! -x "$LLVM_BIN/aarch64-w64-mingw32-clang++" ]]; then
    info "Downloading llvm-mingw (aarch64-hosted) ..."
    rm -rf "$LLVM_DIR"   # clean any partial download
    download "$LLVM_ARM_URL" "$TOOLS/llvm-arm.tar.xz"
    mkdir -p "$LLVM_DIR"
    tar -xf "$TOOLS/llvm-arm.tar.xz" -C "$LLVM_DIR" --strip-components=1
    rm -f "$TOOLS/llvm-arm.tar.xz"
    [[ -x "$LLVM_BIN/aarch64-w64-mingw32-clang++" ]] \
        && ok "llvm-mingw (aarch64) ready" \
        || warn "llvm-mingw extract failed — winARM targets will be skipped"
else
    ok "llvm-mingw (aarch64) already present: $LLVM_BIN"
fi

# ── build helpers ─────────────────────────────────────────────────────────
build_fathom() {
    local cc="$1" obj="$2" extra="${3:-}"
    [[ ! -f "$FATHOM_SRC" ]] && { echo ""; return; }
    # shellcheck disable=SC2086
    "$cc" -O2 -D_DEFAULT_SOURCE $extra \
        -I"$FATHOM_INC" -c "$FATHOM_SRC" -o "$obj" 2>/dev/null \
        && echo "$obj" \
        || { warn "Fathom failed: $(basename "$obj")"; echo ""; }
}

compile() {
    local cxx="$1" out="$2" base_flags="$3" sflags="$4"
    local fobj="${5:-}" opt="${6:--O3 -DNDEBUG}" extra="${7:-}"
    local name; name="$(basename "$out")"

    if echo "$extra" | grep -q "USE_NNUE" && [[ -z "${NNUE_FILE:-}" ]]; then
        skip "$name  (no .nnue file)"; return
    fi

    info "  $name ..."
    local sobjs=""; [[ -n "$fobj" && -f "$fobj" ]] && sobjs="$fobj"

    # Windows (MinGW/llvm-mingw) uses winpthread bundled in the runtime.
    # -lpthread is a glibc/musl concept and will cause linker errors on Windows.
    local thread_flags="-pthread"
    local thread_libs="-lpthread"
    case "$name" in *.exe) thread_flags=""; thread_libs="" ;; esac

    local err_out
    # shellcheck disable=SC2086
    err_out=$( "$cxx" \
        -std=c++17 $opt \
        -Wall -Wextra -Wno-unused-parameter \
        $thread_flags $base_flags $sflags $extra \
        "$SRC" $sobjs -o "$out" $thread_libs 2>&1 ) || true

    if [[ -f "$out" ]]; then
        ok "    $(basename "$out")  ($(du -sh "$out" | cut -f1))"
    else
        warn "    FAILED: $name"
        # Show all output — don't grep, so clang driver errors are always visible
        echo "$err_out" | head -8 | sed 's/^/        /' || true
    fi
}

build_arch_variants() {
    local tag="$1" cxx="$2" cc="$3" cxxf="$4" cf="$5" ext="$6"
    step "Building $tag"

    local SFON="-DUSE_SYZYGY -I$FATHOM_INC"
    local SFOFF="-DNO_SYZYGY"
    local fr="" ff=""

    if [[ -f "$FATHOM_SRC" ]]; then
        # shellcheck disable=SC2086
        fr=$(build_fathom "$cc" "$BUILD/tb_${tag}_r.o" "$cf") || fr=""
        # shellcheck disable=SC2086
        ff=$(build_fathom "$cc" "$BUILD/tb_${tag}_f.o" "$cf") || ff=""
    fi

    compile "$cxx" "$BUILD/hugine_${tag}_base${ext}"     "$cxxf" "$SFOFF" ""   "-O3 -DNDEBUG"
    if [[ -f "$FATHOM_SRC" ]]; then
        compile "$cxx" "$BUILD/hugine_${tag}_syzygy${ext}" "$cxxf" "$SFON"  "$fr" "-O3 -DNDEBUG"
    else
        skip "hugine_${tag}_syzygy${ext}  (no Fathom)"
    fi
    compile "$cxx" "$BUILD/hugine_${tag}_nnue${ext}"     "$cxxf" "$SFOFF" ""   "-O3 -DNDEBUG" "-DUSE_NNUE"
    if [[ -f "$FATHOM_SRC" ]]; then
        compile "$cxx" "$BUILD/hugine_${tag}_full${ext}"   "$cxxf" "$SFON"  "$ff" "-O3 -DNDEBUG" "-DUSE_NNUE"
    else
        skip "hugine_${tag}_full${ext}  (no Fathom)"
    fi
    compile "$cxx" "$BUILD/hugine_${tag}_chess960${ext}" "$cxxf" "$SFOFF" ""   "-O3 -DNDEBUG" "-DCHESS960_EXTRA_DEBUG"
    compile "$cxx" "$BUILD/hugine_${tag}_debug${ext}"    "$cxxf" "$SFOFF" ""   "-O0 -g -DDEBUG"
}

# ── 4 architectures ───────────────────────────────────────────────────────
#
# Windows x86-64: apt g++-mingw-w64-x86-64 (installed above, proven on ARM64 Ubuntu)
# We deliberately avoid llvm-mingw here — apt mingw works reliably for x86
# and doesn't require the llvm-mingw download to succeed first.
build_arch_variants "winx86" \
    "x86_64-w64-mingw32-g++" \
    "x86_64-w64-mingw32-gcc" \
    "-static -static-libgcc -static-libstdc++" "" ".exe"

# Windows ARM64: llvm-mingw aarch64-hosted (only tool that can target ARM64 Windows)
# If llvm-mingw download failed above, these will fail gracefully with an error line.
build_arch_variants "winARM" \
    "$LLVM_BIN/aarch64-w64-mingw32-clang++" \
    "$LLVM_BIN/aarch64-w64-mingw32-clang" \
    "--target=aarch64-w64-mingw32 -static" \
    "--target=aarch64-w64-mingw32" ".exe"

# Linux x86-64: apt cross-compiler (aarch64 Ubuntu → x86_64 Linux) + static
build_arch_variants "linux86" \
    "x86_64-linux-gnu-g++" \
    "x86_64-linux-gnu-gcc" \
    "-static" "" ""

# Linux ARM64: native (already on aarch64) + static
build_arch_variants "linuxARM" \
    "g++" "gcc" \
    "-static" "" ""

echo ""
ok "=== Inner build complete ==="
echo ""
ls -lh "$BUILD/" 2>/dev/null || true
INNER_SCRIPT

    chmod +x "$INNER"

    step "Running inner build inside proot Ubuntu ..."
    proot-distro login ubuntu -- bash /root/hugine_cross_build.sh

    step "Copying binaries back to Termux ..."
    mkdir -p "$BUILD_DIR"
    copied=0
    while IFS= read -r -d '' f; do
        fname="$(basename "$f")"
        case "$fname" in
            hugine_*)
                cp "$f" "$BUILD_DIR/$fname"
                ok "  $fname"
                copied=$((copied+1))
                ;;
        esac
    done < <(find "$UB_HUGINE/build" -maxdepth 1 -type f -print0 2>/dev/null)
    info "Copied $copied binaries"

    print_summary
fi
