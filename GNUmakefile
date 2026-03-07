# =============================================================================
# Hugine 3.0 "Gama" — Cross-Platform Makefile
# =============================================================================
# Auto-detects: OS, architecture, best compiler, fathom/Syzygy, NNUE file.
# Lets you override any setting via command-line flags.
#
# Usage:
#   make                      # auto-detect everything, best settings
#   make CXX=clang++          # use a specific compiler
#   make SYZYGY=1             # force-enable Syzygy (fathom must be present)
#   make SYZYGY=0             # force-disable Syzygy
#   make NNUE=1               # force-enable NNUE (net file must be present)
#   make CHESS960=1           # build with Chess960 debug output
#   make DEBUG=1              # debug build (no optimisation, assertions on)
#   make NATIVE=0             # portable build (no -march=native)
#   make EXE=myengine         # custom output binary name
#   make clean                # remove build artefacts
#   make help                 # print this message
# =============================================================================

# --------------------------------------------------------------------------
# Detect host OS
# --------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    HOST_OS := windows
    EXE_EXT := .exe
else
    UNAME_S := $(shell uname -s 2>/dev/null)
    ifeq ($(UNAME_S),Darwin)
        HOST_OS := macos
    else ifeq ($(UNAME_S),Linux)
        # Distinguish Android/Termux from plain Linux
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

# --------------------------------------------------------------------------
# Detect host architecture
# --------------------------------------------------------------------------
UNAME_M := $(shell uname -m 2>/dev/null || echo unknown)
ifeq ($(UNAME_M),x86_64)
    HOST_ARCH := x86_64
else ifeq ($(UNAME_M),amd64)
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

# --------------------------------------------------------------------------
# Pick best available C++ compiler
# --------------------------------------------------------------------------
ifeq ($(HOST_OS),windows)
    # On Windows prefer clang-cl, then g++ from MinGW, then MSVC cl.exe
    ifeq ($(CXX),)
        ifneq ($(shell where clang++ 2>nul),)
            CXX := clang++
        else ifneq ($(shell where g++ 2>nul),)
            CXX := g++
        else
            CXX := cl
        endif
    endif
else
    ifeq ($(CXX),)
        ifneq ($(shell which clang++ 2>/dev/null),)
            CXX := clang++
        else ifneq ($(shell which g++ 2>/dev/null),)
            CXX := g++
        else
            CXX := c++
        endif
    endif
endif

# --------------------------------------------------------------------------
# Source / output
# --------------------------------------------------------------------------
SRC     := hugine.cpp
EXE     ?= build/hugine$(EXE_EXT)

# --------------------------------------------------------------------------
# Base flags
# --------------------------------------------------------------------------
STD     := -std=c++17
WARN    := -Wall -Wextra -Wno-unused-parameter

ifeq ($(DEBUG),1)
    OPT := -O0 -g -DDEBUG
else
    OPT := -O3 -DNDEBUG
endif

# --------------------------------------------------------------------------
# Architecture-specific flags
# --------------------------------------------------------------------------
ifeq ($(NATIVE),0)
    ARCH_FLAGS :=
else ifeq ($(HOST_OS),android)
    # -march=native can generate instructions unsupported by the Android kernel;
    # leave it out on Termux and let the compiler pick a safe baseline.
    ARCH_FLAGS :=
else ifeq ($(HOST_ARCH),x86_64)
    ARCH_FLAGS := -march=native
else ifeq ($(HOST_ARCH),arm64)
    ARCH_FLAGS :=
else
    ARCH_FLAGS :=
endif

# --------------------------------------------------------------------------
# Threading
# --------------------------------------------------------------------------
ifeq ($(HOST_OS),windows)
    # MinGW/MSYS2 needs -lpthread; MSVC (cl / cl.exe) uses native Win32 threads.
    # Use $(notdir) + $(filter) for an exact match so clang++/clang-cl are not
    # mistakenly classified as MSVC.
    ifeq ($(filter cl cl.exe,$(notdir $(CXX))),)
        THREAD_FLAGS := -pthread
        THREAD_LIBS  := -lpthread
    else
        THREAD_FLAGS :=
        THREAD_LIBS  :=
    endif
else
    THREAD_FLAGS := -pthread
    THREAD_LIBS  := -lpthread
endif

# --------------------------------------------------------------------------
# Auto-detect Syzygy / fathom
# --------------------------------------------------------------------------
FATHOM_HDR  := Fathom/src/tbprobe.h
FATHOM_OBJ  := Fathom/obj/tbprobe.o

ifeq ($(SYZYGY),1)
    USE_SYZYGY := 1
    $(info [syzygy] forced ON by SYZYGY=1)
else ifeq ($(SYZYGY),0)
    USE_SYZYGY := 0
    $(info [syzygy] forced OFF by SYZYGY=0)
else
    ifeq ($(wildcard $(FATHOM_HDR)),$(FATHOM_HDR))
        USE_SYZYGY := 1
        $(info [syzygy] auto-detected jdart1 Fathom at $(FATHOM_HDR) — enabling)
    else
        USE_SYZYGY := 0
        $(info [syzygy] Fathom not found — building without Syzygy)
        $(info [syzygy] To enable: git clone https://github.com/jdart1/Fathom Fathom && make -C Fathom)
    endif
endif

ifeq ($(USE_SYZYGY),1)
    # -IFathom/src lets the bare '#include "tbprobe.h"' resolve correctly.
    SYZYGY_FLAGS := -DUSE_SYZYGY -IFathom/src
    SYZYGY_OBJS  := $(FATHOM_OBJ)
else
    SYZYGY_FLAGS := -DNO_SYZYGY
    SYZYGY_OBJS  :=
endif

# --------------------------------------------------------------------------
# Auto-detect NNUE network file
# --------------------------------------------------------------------------
NNUE_FILE := $(wildcard *.nnue nn-*.bin)

ifeq ($(NNUE),1)
    USE_NNUE := 1
    $(info [nnue] forced ON by NNUE=1)
else ifeq ($(NNUE),0)
    USE_NNUE := 0
    $(info [nnue] forced OFF by NNUE=0)
else
    ifneq ($(NNUE_FILE),)
        USE_NNUE := 1
        $(info [nnue] auto-detected network file: $(firstword $(NNUE_FILE)) — enabling)
    else
        USE_NNUE := 0
        $(info [nnue] no .nnue or nn-*.bin file found — classical eval)
    endif
endif

ifeq ($(USE_NNUE),1)
    NNUE_FLAGS := -DUSE_NNUE
else
    NNUE_FLAGS :=
endif

# --------------------------------------------------------------------------
# Chess960 flag
# --------------------------------------------------------------------------
ifeq ($(CHESS960),1)
    C960_FLAGS := -DCHESS960_EXTRA_DEBUG
else
    C960_FLAGS :=
endif

# --------------------------------------------------------------------------
# Assemble final flags
# --------------------------------------------------------------------------
ALL_FLAGS := $(STD) $(OPT) $(WARN) $(ARCH_FLAGS) $(THREAD_FLAGS) \
             $(SYZYGY_FLAGS) $(NNUE_FLAGS) $(C960_FLAGS)
ALL_LIBS  := $(THREAD_LIBS)

# MSVC (cl / cl.exe) uses /flag syntax; everything else (gcc, clang, mingw,
# aarch64-*-clang++) uses -flag syntax.  Match ONLY the bare executable name
# so that "clang++" and "aarch64-linux-android-clang++" are never matched.
ifeq ($(filter cl cl.exe,$(notdir $(CXX))),$(notdir $(CXX)))
    ALL_FLAGS := /std:c++17 /O2 /EHsc $(SYZYGY_FLAGS) $(NNUE_FLAGS)
    ALL_LIBS  :=
    LINK_CMD   = $(CXX) $(ALL_FLAGS) $(SRC) $(SYZYGY_OBJS) /Fe:$(EXE)
else
    LINK_CMD   = $(CXX) $(ALL_FLAGS) $(SRC) $(SYZYGY_OBJS) -o $(EXE) $(ALL_LIBS)
endif

# --------------------------------------------------------------------------
# Default target
# --------------------------------------------------------------------------
.PHONY: all clean help syzygy fathom

all: syzygy $(EXE)
	@echo ""
	@echo "  ✓  Built: $(EXE)"
	@echo "     OS   : $(HOST_OS)"
	@echo "     Arch : $(HOST_ARCH)"
	@echo "     CXX  : $(CXX)"
	@echo "     Syzygy: $(if $(filter 1,$(USE_SYZYGY)),yes,no)"
	@echo "     NNUE  : $(if $(filter 1,$(USE_NNUE)),yes,no)"

# --------------------------------------------------------------------------
# Compile fathom (C, not C++) when Syzygy is enabled
# --------------------------------------------------------------------------
ifeq ($(USE_SYZYGY),1)
syzygy: $(FATHOM_OBJ)

$(FATHOM_OBJ):
	@echo "[fathom] Building jdart1 Fathom via its Makefile ..."
	$(MAKE) -C Fathom CC="$(CC)"
	@test -f $(FATHOM_OBJ) || (echo "[error] $(FATHOM_OBJ) not found after build"; exit 1)
	@echo "[fathom] $(FATHOM_OBJ) ready"
else
syzygy:
	@true
endif

# --------------------------------------------------------------------------
# Link the engine
# --------------------------------------------------------------------------
$(EXE): $(SRC) $(SYZYGY_OBJS)
	@mkdir -p build
	@echo "[build] Compiling + linking $(EXE) ..."
	$(LINK_CMD)

# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------
clean:
	rm -rf build/
	rm -f $(FATHOM_OBJ)
	@echo "Cleaned."

help:
	@echo "Hugine 5.0 Iota — Makefile help"
	@echo ""
	@echo "  make                 auto-detect everything and build"
	@echo "  make SYZYGY=1/0      force Syzygy on/off"
	@echo "  make NNUE=1/0        force NNUE on/off"
	@echo "  make CHESS960=1      enable Chess960 extra debug"
	@echo "  make DEBUG=1         debug build"
	@echo "  make NATIVE=0        portable build (no -march=native)"
	@echo "  make CXX=clang++     choose compiler"
	@echo "  make EXE=myengine    custom output name"
	@echo "  make clean           remove build artefacts"
