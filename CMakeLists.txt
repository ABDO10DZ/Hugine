cmake_minimum_required(VERSION 3.14)
project(Hugine VERSION 5.1.0 LANGUAGES CXX)

# =============================================================================
# Hugine 5.1.0 "Iota" — CMake Build
# =============================================================================
# Options (pass with -D<OPTION>=ON|OFF):
#
#   NNUE         ON   Compile NNUE evaluator (default: auto)
#                     Supports Hugine-native .nnue + any Stockfish .nnue natively
#   NNUE_LARGE   OFF  512-node Hugine-native architecture
#   NNUE_XL      OFF  1024-node Hugine-native architecture (implies NNUE_LARGE)
#   SYZYGY       OFF  Syzygy endgame tablebase support (requires Fathom)
#   PEXT         OFF  BMI2 PEXT bitboards (x86_64 only)
#   NATIVE       ON   -march=native (disable for portable builds)
#   AVX2         OFF  Force AVX2 SIMD
#   SSE4         OFF  Force SSE4.1 SIMD
#   SANITIZE     OFF  ASAN + UBSAN (debug)
#
# Quick start:
#   cmake -B build -DNNUE=ON
#   cmake --build build --config Release
#
# All-variants:
#   cmake --build build --target all_variants
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Source ────────────────────────────────────────────────────────────────────
set(SRC "${CMAKE_SOURCE_DIR}/hugine-iota-v510.cpp")
if(NOT EXISTS "${SRC}")
    message(FATAL_ERROR "Source file not found: ${SRC}")
endif()

# ── Options ───────────────────────────────────────────────────────────────────
option(NNUE       "Compile NNUE evaluator (Hugine-native + Stockfish .nnue)" ON)
option(NNUE_LARGE "512-node Hugine-native architecture" OFF)
option(NNUE_XL    "1024-node Hugine-native architecture"  OFF)
option(SYZYGY     "Syzygy endgame tablebases (Fathom)" OFF)
option(PEXT       "BMI2 PEXT bitboards" OFF)
option(NATIVE     "Enable -march=native" ON)
option(AVX2       "Force AVX2 SIMD" OFF)
option(SSE4       "Force SSE4.1 SIMD" OFF)
option(SANITIZE   "ASAN + UBSAN" OFF)

# ── Main target ───────────────────────────────────────────────────────────────
add_executable(hugine ${SRC})
target_compile_options(hugine PRIVATE
    -Wall -Wextra -Wno-unused-parameter)

# ── Build type ────────────────────────────────────────────────────────────────
if(SANITIZE)
    target_compile_options(hugine PRIVATE -O0 -g -fsanitize=address,undefined)
    target_link_options(hugine PRIVATE -fsanitize=address,undefined)
    message(STATUS "[hugine] ASAN+UBSAN enabled")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hugine PRIVATE -O0 -g)
else()
    target_compile_definitions(hugine PRIVATE NDEBUG)
    target_compile_options(hugine PRIVATE -O3)
endif()

# ── Architecture ─────────────────────────────────────────────────────────────
if(NATIVE AND NOT CMAKE_CROSSCOMPILING)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        target_compile_options(hugine PRIVATE -march=native)
    endif()
endif()
if(AVX2)
    target_compile_options(hugine PRIVATE -mavx2)
    message(STATUS "[hugine] AVX2 forced")
endif()
if(SSE4)
    target_compile_options(hugine PRIVATE -msse4.1)
    message(STATUS "[hugine] SSE4.1 forced")
endif()
if(PEXT)
    target_compile_options(hugine PRIVATE -mbmi2)
    message(STATUS "[hugine] BMI2/PEXT enabled")
else()
    target_compile_definitions(hugine PRIVATE NO_PEXT)
endif()

# ── Threading ─────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)
target_link_libraries(hugine PRIVATE Threads::Threads)

# ── NNUE ─────────────────────────────────────────────────────────────────────
# -DUSE_NNUE compiles in both the SFNNUEEvaluator (Stockfish .nnue native)
# and NNUEEvaluator (Hugine-native .nnue).  At runtime the loader auto-selects
# based on the file header.
if(NNUE)
    target_compile_definitions(hugine PRIVATE USE_NNUE)
    message(STATUS "[hugine] NNUE: ON — Hugine-native + Stockfish SF10-SF16+ .nnue")
    message(STATUS "         Runtime: setoption name EvalFile value <path>.nnue")
    if(NNUE_XL)
        target_compile_definitions(hugine PRIVATE NNUE_XL NNUE_LARGE)
        message(STATUS "[hugine] NNUE_XL: 1024-node Hugine architecture")
    elseif(NNUE_LARGE)
        target_compile_definitions(hugine PRIVATE NNUE_LARGE)
        message(STATUS "[hugine] NNUE_LARGE: 512-node Hugine architecture")
    endif()
else()
    message(STATUS "[hugine] NNUE: OFF (classical eval)")
endif()

# ── Syzygy / Fathom ───────────────────────────────────────────────────────────
if(SYZYGY)
    set(FATHOM_DIR "${CMAKE_SOURCE_DIR}/Fathom")
    if(NOT EXISTS "${FATHOM_DIR}/src/tbprobe.h")
        message(FATAL_ERROR
            "SYZYGY=ON but Fathom not found at ${FATHOM_DIR}.\n"
            "  git clone https://github.com/jdart1/Fathom Fathom")
    endif()
    add_library(fathom STATIC "${FATHOM_DIR}/src/tbprobe.c")
    target_compile_options(fathom PRIVATE -O2)
    target_compile_definitions(fathom PRIVATE _DEFAULT_SOURCE)
    target_include_directories(fathom PUBLIC "${FATHOM_DIR}/src")
    target_compile_definitions(hugine PRIVATE USE_SYZYGY)
    target_include_directories(hugine PRIVATE "${FATHOM_DIR}/src")
    target_link_libraries(hugine PRIVATE fathom)
    message(STATUS "[hugine] Syzygy: Fathom at ${FATHOM_DIR}")
else()
    target_compile_definitions(hugine PRIVATE NO_SYZYGY)
    message(STATUS "[hugine] Syzygy: OFF")
endif()

# ── Install ───────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS hugine RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ── All-variants target (convenience) ────────────────────────────────────────
add_custom_target(all_variants
    COMMAND ${CMAKE_COMMAND} -E echo "Building all 11 Hugine variants …"
    COMMAND ${CMAKE_CXX_COMPILER} -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_base          -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -march=native -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNDEBUG -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_fast          -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -march=native -mbmi2 -std=c++17 -pthread -DNO_SYZYGY -DNDEBUG -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_pext          -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -mavx2 -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNDEBUG -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_avx2          -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -msse4.1 -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNDEBUG -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_sse4          -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DUSE_NNUE -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_nnue          -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -march=native -mbmi2 -std=c++17 -pthread -DNO_SYZYGY -DUSE_NNUE -DNDEBUG -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_fast_nnue     -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -march=native -mavx2 -mbmi2 -std=c++17 -pthread -DNO_SYZYGY -DUSE_NNUE -DNDEBUG -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_fast_nnue_avx2 -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNNUE_LARGE -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_large         -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DUSE_NNUE -DNNUE_LARGE -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_large_nnue  -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DNNUE_XL -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_xl          -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O2  -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -DUSE_NNUE -DNNUE_XL -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_xl_nnue     -lpthread
    COMMAND ${CMAKE_CXX_COMPILER} -O0  -g -fsanitize=address,undefined -std=c++17 -pthread -DNO_SYZYGY -DNO_PEXT -Wall -Wextra -Wno-unused-parameter ${SRC} -o ${CMAKE_BINARY_DIR}/hugine_asan -lpthread
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "All 11 variants built in ${CMAKE_BINARY_DIR}/"
    COMMENT "Building all Hugine variants"
)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "Hugine 5.1.0 Iota — configuration summary")
message(STATUS "  Source:    ${SRC}")
message(STATUS "  NNUE:      ${NNUE}")
message(STATUS "  NNUE_LARGE:${NNUE_LARGE}")
message(STATUS "  NNUE_XL:   ${NNUE_XL}")
message(STATUS "  Syzygy:    ${SYZYGY}")
message(STATUS "  PEXT:      ${PEXT}")
message(STATUS "  Sanitize:  ${SANITIZE}")
message(STATUS "")
message(STATUS "  Build:    cmake --build <build_dir>")
message(STATUS "  Variants: cmake --build <build_dir> --target all_variants")
message(STATUS "")
