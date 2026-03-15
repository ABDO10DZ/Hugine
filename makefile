# =============================================================================
# Hugine 5.1.0 "Iota" — Cross-Platform Makefile
# =============================================================================
# Auto-detects OS, architecture, compiler, Syzygy, NNUE.
# NNUE builds (USE_NNUE=1) natively support any Stockfish .nnue file
# plus Hugine-native .nnue — no conversion tool required.
#
# Usage:
#   make                       auto-detect, best settings
#   make NNUE=1                force NNUE on (loads SF or Hugine .nnue)
#   make NNUE=0                force NNUE off (classical eval only)
#   make NNUE_LARGE=1          Hugine 512-node architecture
#   make SYZYGY=1              force Syzygy on (Fathom must be present)
#   make SYZYGY=0              force Syzygy off
#   make AVX2=1                force AVX2 SIMD path
#   make SSE4=1                force SSE4.1 SIMD path
#   make PEXT=1                enable BMI2 PEXT bitboards (x86_64 only)
#   make PGO=1                 2-pass profile-guided optimisation
#   make DEBUG=1               debug build (ASAN + UBSAN)
#   make NATIVE=0              portable, no -march=native
#   make CXX=clang++           pick compiler
#   make EXE=myengine          custom output name
#   make clean                 remove build artefacts
#   make all_variants          build all 13 standard variants
#   make help                  print this message
# =============================================================================

# ── OS detection ─────────────────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    HOST_OS  := windows
    EXE_EXT  := .exe
else
    UNAME_S  := $(shell uname -s 2>/dev/null)
    ifeq ($(UNAME_S),Darwin)
        HOST_OS := macos
    else ifeq ($(UNAME_S),Linux)
        ifneq ($(wildcard /data/data/com.termux),)
            HOST_OS := android
        else
            HOST_OS := linux
        endif
    else
        HOST_OS := posix
    endif
    EXE_EXT :=
endif

# ── Arch detection ────────────────────────────────────────────────────────────
UNAME_M := $(shell uname -m 2>/dev/null || echo unknown)
ifeq ($(filter x86_64 amd64,$(UNAME_M)),$(UNAME_M))
    HOST_ARCH := x86_64
else ifeq ($(findstring aarch64,$(UNAME_M)),aarch64)
    HOST_ARCH := arm64
else ifeq ($(findstring arm64,$(UNAME_M)),arm64)
    HOST_ARCH := arm64
else ifeq ($(findstring arm,$(UNAME_M)),arm)
    HOST_ARCH := arm32
else
    HOST_ARCH := unknown
endif

# ── Compiler ─────────────────────────────────────────────────────────────────
ifeq ($(CXX),)
    ifeq ($(HOST_OS),windows)
        ifneq ($(shell where clang++ 2>nul),)
            CXX := clang++
        else
            CXX := g++
        endif
    else
        ifneq ($(shell which clang++ 2>/dev/null),)
            CXX := clang++
        else
            CXX := g++
        endif
    endif
endif

# ── Source & output ───────────────────────────────────────────────────────────
SRC  := hugine-iota-v510.cpp
EXE  ?= build/hugine$(EXE_EXT)
STD  := -std=c++17
WARN := -Wall -Wextra -Wno-unused-parameter

# ── Optimisation ─────────────────────────────────────────────────────────────
ifeq ($(DEBUG),1)
    OPT := -O0 -g -fsanitize=address,undefined -DDEBUG
    $(info [build] DEBUG: ASAN + UBSAN)
else
    OPT := -O3 -DNDEBUG
endif

# ── Arch flags ────────────────────────────────────────────────────────────────
ifeq ($(NATIVE),0)
    ARCH_FLAGS :=
else ifeq ($(HOST_OS),android)
    ARCH_FLAGS :=
    $(info [arch]  -march=native skipped on Android)
else ifeq ($(HOST_ARCH),x86_64)
    ARCH_FLAGS := -march=native
else
    ARCH_FLAGS :=
endif

ifeq ($(AVX2),1)
    ARCH_FLAGS += -mavx2
    $(info [simd]  AVX2 forced)
endif
ifeq ($(SSE4),1)
    ARCH_FLAGS += -msse4.1
    $(info [simd]  SSE4.1 forced)
endif
ifeq ($(PEXT),1)
    ARCH_FLAGS += -mbmi2
    $(info [arch]  BMI2/PEXT enabled)
    PEXT_FLAGS :=
else
    PEXT_FLAGS := -DNO_PEXT
endif

# ── Threading ─────────────────────────────────────────────────────────────────
ifeq ($(filter cl cl.exe,$(notdir $(CXX))),)
    THREAD_FLAGS := -pthread
    ifeq ($(HOST_OS),macos)
        THREAD_LIBS :=
    else
        THREAD_LIBS := -lpthread
    endif
else
    THREAD_FLAGS :=
    THREAD_LIBS  :=
endif

# ── Syzygy / Fathom ───────────────────────────────────────────────────────────
FATHOM_HDR := Fathom/src/tbprobe.h
FATHOM_OBJ := Fathom/obj/tbprobe.o

ifeq ($(SYZYGY),1)
    USE_SYZYGY := 1
    $(info [syzygy] ON (forced))
else ifeq ($(SYZYGY),0)
    USE_SYZYGY := 0
    $(info [syzygy] OFF (forced))
else ifeq ($(wildcard $(FATHOM_HDR)),$(FATHOM_HDR))
    USE_SYZYGY := 1
    $(info [syzygy] auto-detected Fathom — enabling)
else
    USE_SYZYGY := 0
    $(info [syzygy] Fathom not found — disable or: git clone https://github.com/jdart1/Fathom Fathom)
endif

ifeq ($(USE_SYZYGY),1)
    SYZYGY_FLAGS := -DUSE_SYZYGY -IFathom/src
    SYZYGY_OBJS  := $(FATHOM_OBJ)
else
    SYZYGY_FLAGS := -DNO_SYZYGY
    SYZYGY_OBJS  :=
endif

# ── NNUE ─────────────────────────────────────────────────────────────────────
# Hugine 5.1.0 supports two NNUE backends compiled behind -DUSE_NNUE:
#   1. SFNNUEEvaluator — loads any Stockfish .nnue natively
#      (HalfKP-256, HalfKA-256, HalfKAv2-512/1024/1536, COMPRESSED_LEB128)
#   2. NNUEEvaluator   — loads Hugine-native .nnue (from nnue_trainer/train.py)
# The loader auto-selects at runtime based on the file's header.
NNUE_FILE := $(wildcard *.nnue nn-*.bin)

ifeq ($(NNUE),1)
    USE_NNUE := 1
    $(info [nnue]  ON (forced))
else ifeq ($(NNUE),0)
    USE_NNUE := 0
    $(info [nnue]  OFF (forced))
else ifneq ($(NNUE_FILE),)
    USE_NNUE := 1
    $(info [nnue]  auto-detected: $(firstword $(NNUE_FILE)))
else
    USE_NNUE := 0
    $(info [nnue]  no .nnue file found — classical eval. Use NNUE=1 to compile evaluator anyway)
endif

ifeq ($(USE_NNUE),1)
    NNUE_FLAGS := -DUSE_NNUE
else
    NNUE_FLAGS :=
endif

ifeq ($(NNUE_LARGE),1)
    NNUE_FLAGS += -DNNUE_LARGE
    $(info [nnue]  LARGE: 512-node Hugine architecture)
endif

# ── PGO ──────────────────────────────────────────────────────────────────────
PGO_DIR := /tmp/hugine_pgo
ifeq ($(PGO),1)
    $(info [pgo]   Two-pass PGO build)
endif

# ── MSVC compat ──────────────────────────────────────────────────────────────
ifeq ($(filter cl cl.exe,$(notdir $(CXX))),$(notdir $(CXX)))
    ALL_FLAGS := /std:c++17 /O2 /EHsc $(SYZYGY_FLAGS) $(NNUE_FLAGS)
    ALL_LIBS  :=
    LINK_CMD   = $(CXX) $(ALL_FLAGS) $(SRC) $(SYZYGY_OBJS) /Fe:$(EXE)
else
    ALL_FLAGS := $(STD) $(OPT) $(WARN) $(ARCH_FLAGS) $(THREAD_FLAGS) \
                 $(SYZYGY_FLAGS) $(NNUE_FLAGS) $(PEXT_FLAGS)
    ALL_LIBS  := $(THREAD_LIBS)
    LINK_CMD   = $(CXX) $(ALL_FLAGS) $(SRC) $(SYZYGY_OBJS) -o $(EXE) $(ALL_LIBS)
endif

# ── Targets ───────────────────────────────────────────────────────────────────
.PHONY: all clean help all_variants syzygy test

all: syzygy $(EXE)
	@echo ""
	@echo "  ✓  Built: $(EXE)"
	@echo "     OS:     $(HOST_OS) / $(HOST_ARCH)"
	@echo "     CXX:    $(CXX)"
	@echo "     Syzygy: $(if $(filter 1,$(USE_SYZYGY)),yes (Fathom),no)"
	@echo "     NNUE:   $(if $(filter 1,$(USE_NNUE)),yes — Hugine-native + any Stockfish .nnue,no)"
	@echo "     PGO:    $(if $(filter 1,$(PGO)),yes (2-pass),no)"
	@echo ""
	@if [ "$(USE_NNUE)" = "1" ]; then \
	    echo "  NNUE runtime usage:"; \
	    echo "    setoption name EvalFile value /path/to/nn-XXXX.nnue"; \
	    echo "  Supported: HalfKP-256, HalfKA-256, HalfKAv2-512/1024/1536,"; \
	    echo "             COMPRESSED_LEB128 (any SF pytorch trainer output)"; \
	fi

# ── Syzygy / Fathom ───────────────────────────────────────────────────────────
ifeq ($(USE_SYZYGY),1)
syzygy: $(FATHOM_OBJ)

$(FATHOM_OBJ):
	@echo "[fathom] Building Fathom …"
	$(MAKE) -C Fathom CC="$(CC)"
	@test -f $(FATHOM_OBJ) || (echo "[error] $(FATHOM_OBJ) not built"; exit 1)
else
syzygy:
	@true
endif

# ── PGO two-pass ─────────────────────────────────────────────────────────────
ifeq ($(PGO),1)
$(EXE): $(SRC) $(SYZYGY_OBJS)
	@mkdir -p build
	@echo "[pgo] Pass 1: instrumenting …"
	$(CXX) $(ALL_FLAGS) -fprofile-generate=$(PGO_DIR) $(SRC) $(SYZYGY_OBJS) \
	    -o $(EXE)_pgo_gen $(ALL_LIBS)
	@mkdir -p $(PGO_DIR)
	@echo "[pgo] Collecting profile data …"
	@printf "position startpos\ngo depth 12\nquit\n" | ./$(EXE)_pgo_gen >/dev/null 2>&1 || true
	@printf "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\nperft 4\nquit\n" \
	    | ./$(EXE)_pgo_gen >/dev/null 2>&1 || true
	@echo "[pgo] Pass 2: optimising …"
	$(CXX) $(ALL_FLAGS) -fprofile-use=$(PGO_DIR) -fprofile-correction \
	    $(SRC) $(SYZYGY_OBJS) -o $(EXE) $(ALL_LIBS)
	@rm -rf $(PGO_DIR) $(EXE)_pgo_gen
else
$(EXE): $(SRC) $(SYZYGY_OBJS)
	@mkdir -p build
	@echo "[build] $(SRC) → $(EXE)"
	$(LINK_CMD)
endif

# ── All 11 variants ───────────────────────────────────────────────────────────
all_variants:
	@mkdir -p build
	@echo "Building all 11 Hugine variants …"
	@echo ""
	$(CXX) -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT $(WARN)                       hugine-iota-v510.cpp -o build/hugine_base          -lpthread && echo "  ✓ hugine_base"
	$(CXX) -O3 -march=native -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNDEBUG $(WARN) hugine-iota-v510.cpp -o build/hugine_fast          -lpthread && echo "  ✓ hugine_fast"
	$(CXX) -O3 -march=native -mbmi2 -std=c++17 -pthread -DNO_SYZYGY -DNDEBUG $(WARN)    hugine-iota-v510.cpp -o build/hugine_pext          -lpthread && echo "  ✓ hugine_pext"
	$(CXX) -O3 -mavx2 -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNDEBUG $(WARN)        hugine-iota-v510.cpp -o build/hugine_avx2          -lpthread && echo "  ✓ hugine_avx2"
	$(CXX) -O3 -msse4.1 -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNDEBUG $(WARN)      hugine-iota-v510.cpp -o build/hugine_sse4          -lpthread && echo "  ✓ hugine_sse4"
	$(CXX) -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DUSE_NNUE $(WARN)            hugine-iota-v510.cpp -o build/hugine_nnue          -lpthread && echo "  ✓ hugine_nnue"
	$(CXX) -O3 -march=native -mbmi2 -std=c++17 -pthread -DNO_SYZYGY -DUSE_NNUE -DNDEBUG $(WARN) hugine-iota-v510.cpp -o build/hugine_fast_nnue     -lpthread && echo "  ✓ hugine_fast_nnue"
	$(CXX) -O3 -march=native -mavx2 -mbmi2 -std=c++17 -pthread -DNO_SYZYGY -DUSE_NNUE -DNDEBUG $(WARN) hugine-iota-v510.cpp -o build/hugine_fast_nnue_avx2 -lpthread && echo "  ✓ hugine_fast_nnue_avx2"
	$(CXX) -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNNUE_LARGE $(WARN)          hugine-iota-v510.cpp -o build/hugine_large         -lpthread && echo "  ✓ hugine_large"
	$(CXX) -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DUSE_NNUE -DNNUE_LARGE $(WARN) hugine-iota-v510.cpp -o build/hugine_large_nnue  -lpthread && echo "  ✓ hugine_large_nnue"
	$(CXX) -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNNUE_XL $(WARN)          hugine-iota-v510.cpp -o build/hugine_xl           -lpthread && echo "  ✓ hugine_xl"
	$(CXX) -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DUSE_NNUE -DNNUE_XL $(WARN)  hugine-iota-v510.cpp -o build/hugine_xl_nnue      -lpthread && echo "  ✓ hugine_xl_nnue"
	$(CXX) -O0  -g -fsanitize=address,undefined -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT $(WARN) hugine-iota-v510.cpp -o build/hugine_asan -lpthread && echo "  ✓ hugine_asan"
	@echo ""
	@echo "  All variants built in ./build/"
	@echo "  NNUE variants (hugine_nnue, hugine_fast_nnue, hugine_fast_nnue_avx2, hugine_large_nnue, hugine_xl_nnue)"
	@echo "  load Stockfish SF10-SF16+ .nnue files natively — no conversion required."

# ── Test suite ────────────────────────────────────────────────────────────────
test:
	python3 test_engine.py

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf build/ $(FATHOM_OBJ)
	@echo "Cleaned."

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Hugine 5.1.0 Iota — Makefile"
	@echo ""
	@echo "  make                   auto-detect and build"
	@echo "  make NNUE=1/0          force NNUE on/off"
	@echo "  make NNUE_LARGE=1      Hugine 512-node architecture"
	@echo "  make SYZYGY=1/0        force Syzygy on/off"
	@echo "  make AVX2=1            force AVX2 SIMD"
	@echo "  make SSE4=1            force SSE4.1 SIMD"
	@echo "  make PEXT=1            enable BMI2 PEXT"
	@echo "  make PGO=1             2-pass PGO build"
	@echo "  make DEBUG=1           ASAN + UBSAN debug"
	@echo "  make NATIVE=0          portable, no -march=native"
	@echo "  make CXX=clang++       pick compiler"
	@echo "  make EXE=myengine      custom output name"
	@echo "  make all_variants      build all 13 standard variants"
	@echo "  make test              run test_engine.py"
	@echo "  make clean             remove build artefacts"
	@echo ""
	@echo "  NNUE runtime (any variant built with NNUE=1):"
	@echo "    setoption name EvalFile value nn-XXXX.nnue"
	@echo "    Supported: HalfKP-256  HalfKA-256  HalfKAv2-512/1024/1536"
	@echo "               COMPRESSED_LEB128  (Stockfish SF10-SF16+ and pytorch trainer)"
