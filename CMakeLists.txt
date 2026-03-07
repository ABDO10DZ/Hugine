# =============================================================================
# Hugine 5.0 "Iota" — CMakeLists.txt
# =============================================================================
# Mirrors builder.sh / Makefile auto-detection logic exactly.
#
# Quick start:
#   mkdir build && cd build
#   cmake ..                        # auto-detect everything
#   cmake --build .
#
# Override options:
#   cmake .. -DSYZYGY=ON            # force Syzygy on  (Fathom must exist)
#   cmake .. -DSYZYGY=OFF           # force Syzygy off
#   cmake .. -DNNUE=ON              # force NNUE on    (.nnue file must exist)
#   cmake .. -DNNUE=OFF             # force NNUE off
#   cmake .. -DNATIVE=OFF           # no -march=native (portable / cross build)
#   cmake .. -DDEBUG_BUILD=ON       # -O0 -g, assertions enabled
#   cmake .. -DCHESS960_DEBUG=ON    # Chess960 extra debug output
#   cmake .. -DCMAKE_CXX_COMPILER=clang++
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#
# Out-of-source build (recommended):
#   cmake -B build && cmake --build build
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(Hugine VERSION 5.0 LANGUAGES CXX C)

# --------------------------------------------------------------------------
# Options  (tri-state: ON / OFF / AUTO where AUTO means detect at config time)
# --------------------------------------------------------------------------
option(SYZYGY        "Enable Syzygy tablebases via jdart1/Fathom (AUTO=detect)" AUTO)
option(NNUE          "Enable NNUE neural-network eval (AUTO=detect .nnue file)" AUTO)
option(NATIVE        "Use -march=native for maximum host performance"           ON)
option(DEBUG_BUILD   "Debug build: -O0 -g, assertions enabled"                  OFF)
option(CHESS960_DEBUG "Enable Chess960 extra debug output"                      OFF)

# Tri-state helper — cmake options are bool, so we use a string cache entry
# for SYZYGY and NNUE so users can set them to "AUTO".
set(SYZYGY "AUTO" CACHE STRING "Syzygy support: AUTO | ON | OFF")
set(NNUE   "AUTO" CACHE STRING "NNUE support:   AUTO | ON | OFF")
set_property(CACHE SYZYGY PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE NNUE   PROPERTY STRINGS AUTO ON OFF)

# --------------------------------------------------------------------------
# Require C++17
# --------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------
set(ENGINE_SRC "${CMAKE_SOURCE_DIR}/hugine.cpp")
if(NOT EXISTS "${ENGINE_SRC}")
    message(FATAL_ERROR "Source file not found: ${ENGINE_SRC}\n"
                        "Run cmake from the hugine directory.")
endif()

# --------------------------------------------------------------------------
# Detect Android / Termux
# --------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   (EXISTS "/data/data/com.termux" OR DEFINED ENV{PREFIX} AND
    "$ENV{PREFIX}" MATCHES "termux"))
    set(IS_ANDROID TRUE)
    message(STATUS "[platform] Android/Termux detected")
else()
    set(IS_ANDROID FALSE)
endif()

# --------------------------------------------------------------------------
# Optimisation flags
# --------------------------------------------------------------------------
if(DEBUG_BUILD)
    message(STATUS "[build]    Debug build (-O0 -g)")
    set(OPT_FLAGS -O0 -g)
    add_compile_definitions(DEBUG)
else()
    if(MSVC)
        set(OPT_FLAGS /O2)
    else()
        set(OPT_FLAGS -O3)
    endif()
    add_compile_definitions(NDEBUG)
endif()

# -march=native: skip on Android (can emit instructions the kernel rejects),
# skip on MSVC (uses /arch: instead), honour NATIVE=OFF for cross builds.
if(NATIVE AND NOT IS_ANDROID AND NOT MSVC)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        list(APPEND OPT_FLAGS -march=native)
        message(STATUS "[arch]     -march=native enabled (x86_64)")
    else()
        message(STATUS "[arch]     -march=native skipped (non-x86 host)")
    endif()
elseif(IS_ANDROID)
    message(STATUS "[arch]     -march=native skipped (Android/Termux)")
elseif(NOT NATIVE)
    message(STATUS "[arch]     -march=native disabled by NATIVE=OFF")
endif()

# --------------------------------------------------------------------------
# Threads
# --------------------------------------------------------------------------
find_package(Threads REQUIRED)

# --------------------------------------------------------------------------
# Detect / configure jdart1/Fathom (Syzygy)
# --------------------------------------------------------------------------
# Fathom layout expected at:
#   ${CMAKE_SOURCE_DIR}/Fathom/src/tbprobe.h    <- header
#   ${CMAKE_SOURCE_DIR}/Fathom/obj/tbprobe.o    <- pre-built object
#   ${CMAKE_SOURCE_DIR}/Fathom/Makefile
#
# CMake does NOT recompile Fathom's C sources directly — it delegates to
# Fathom's own Makefile via a custom command, exactly like builder.sh does.
# This is the correct approach because jdart1/Fathom's Makefile sets up
# obj/ and handles platform-specific C11 flags internally.
# --------------------------------------------------------------------------
set(FATHOM_DIR  "${CMAKE_SOURCE_DIR}/Fathom")
set(FATHOM_HDR  "${FATHOM_DIR}/src/tbprobe.h")
set(FATHOM_OBJ  "${FATHOM_DIR}/obj/tbprobe.o")
set(FATHOM_MK   "${FATHOM_DIR}/Makefile")

# Resolve SYZYGY setting
if(SYZYGY STREQUAL "ON")
    if(NOT EXISTS "${FATHOM_HDR}")
        message(FATAL_ERROR
            "[syzygy] SYZYGY=ON but ${FATHOM_HDR} not found.\n"
            "Clone jdart1 Fathom first:\n"
            "  git clone https://github.com/jdart1/Fathom Fathom\n"
            "Then re-run cmake.")
    endif()
    set(USE_SYZYGY TRUE)
    message(STATUS "[syzygy]   forced ON")
elseif(SYZYGY STREQUAL "OFF")
    set(USE_SYZYGY FALSE)
    message(STATUS "[syzygy]   forced OFF")
else()  # AUTO
    if(EXISTS "${FATHOM_HDR}")
        set(USE_SYZYGY TRUE)
        message(STATUS "[syzygy]   auto-detected Fathom at ${FATHOM_HDR} -> ON")
    else()
        set(USE_SYZYGY FALSE)
        message(STATUS "[syzygy]   Fathom not found -> OFF")
        message(STATUS "           (to enable: git clone https://github.com/jdart1/Fathom Fathom)")
    endif()
endif()

# --------------------------------------------------------------------------
# Build Fathom via its Makefile as an external project command
# --------------------------------------------------------------------------
if(USE_SYZYGY)
    if(NOT EXISTS "${FATHOM_MK}")
        message(FATAL_ERROR
            "[syzygy] Fathom/Makefile not found.\n"
            "Clone with: git clone https://github.com/jdart1/Fathom Fathom")
    endif()

    # Extra CFLAGS for platforms that need _DEFAULT_SOURCE
    set(FATHOM_EXTRA_CFLAGS "-O2")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR IS_ANDROID)
        string(APPEND FATHOM_EXTRA_CFLAGS " -D_DEFAULT_SOURCE")
    endif()

    # Invoke Fathom's own Makefile.  The stamp file lets cmake know whether
    # the object is up-to-date without re-running make every build.
    set(FATHOM_STAMP "${CMAKE_BINARY_DIR}/fathom_built.stamp")

    add_custom_command(
        OUTPUT  "${FATHOM_OBJ}" "${FATHOM_STAMP}"
        COMMAND ${CMAKE_MAKE_PROGRAM}
                -C "${FATHOM_DIR}"
                CC="${CMAKE_C_COMPILER}"
                CFLAGS="${FATHOM_EXTRA_CFLAGS}"
        COMMAND ${CMAKE_COMMAND} -E touch "${FATHOM_STAMP}"
        COMMENT "[fathom] Building jdart1/Fathom via its Makefile..."
        VERBATIM
    )

    add_custom_target(fathom_build DEPENDS "${FATHOM_OBJ}" "${FATHOM_STAMP}")

    # Validate that the object was actually produced
    # (checked at link time — cmake will error if FATHOM_OBJ is missing)
endif()

# --------------------------------------------------------------------------
# Detect NNUE network file
# --------------------------------------------------------------------------
if(NNUE STREQUAL "ON")
    set(USE_NNUE TRUE)
    message(STATUS "[nnue]     forced ON")
elseif(NNUE STREQUAL "OFF")
    set(USE_NNUE FALSE)
    message(STATUS "[nnue]     forced OFF")
else()  # AUTO
    file(GLOB NNUE_FILES
        "${CMAKE_SOURCE_DIR}/*.nnue"
        "${CMAKE_SOURCE_DIR}/nn-*.bin")
    if(NNUE_FILES)
        list(GET NNUE_FILES 0 NNUE_FOUND)
        set(USE_NNUE TRUE)
        message(STATUS "[nnue]     auto-detected ${NNUE_FOUND} -> ON")
    else()
        set(USE_NNUE FALSE)
        message(STATUS "[nnue]     no .nnue / nn-*.bin found -> classical eval")
    endif()
endif()

# --------------------------------------------------------------------------
# Chess960 extra debug
# --------------------------------------------------------------------------
if(CHESS960_DEBUG)
    message(STATUS "[chess960] extra debug output ON")
endif()

# --------------------------------------------------------------------------
# Define the executable
# --------------------------------------------------------------------------
add_executable(hugine "${ENGINE_SRC}")

# Always place the compiled binary in <cmake_binary_dir>/build/
# so both `cmake -B build` and in-source cmake produce the same layout.
set_target_properties(hugine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build"
)

# --------------------------------------------------------------------------
# Compile definitions
# --------------------------------------------------------------------------
if(USE_SYZYGY)
    target_compile_definitions(hugine PRIVATE USE_SYZYGY)
else()
    target_compile_definitions(hugine PRIVATE NO_SYZYGY)
endif()

if(USE_NNUE)
    target_compile_definitions(hugine PRIVATE USE_NNUE)
endif()

if(CHESS960_DEBUG)
    target_compile_definitions(hugine PRIVATE CHESS960_EXTRA_DEBUG)
endif()

# --------------------------------------------------------------------------
# Compile options
# --------------------------------------------------------------------------
if(MSVC)
    target_compile_options(hugine PRIVATE
        /std:c++17 /W3 /EHsc
        $<IF:$<BOOL:${DEBUG_BUILD}>,/Od /Zi,/O2>
    )
else()
    target_compile_options(hugine PRIVATE
        -Wall -Wextra -Wno-unused-parameter
        ${OPT_FLAGS}
    )
endif()

# --------------------------------------------------------------------------
# Include paths
# --------------------------------------------------------------------------
if(USE_SYZYGY)
    # Bare  #include "tbprobe.h"  resolves via this include path.
    target_include_directories(hugine PRIVATE "${FATHOM_DIR}/src")
endif()

# --------------------------------------------------------------------------
# Link libraries
# --------------------------------------------------------------------------
target_link_libraries(hugine PRIVATE Threads::Threads)

if(USE_SYZYGY)
    # Link Fathom's pre-built object file directly
    target_sources(hugine PRIVATE "${FATHOM_OBJ}")
    add_dependencies(hugine fathom_build)
endif()

# MSVC: no extra thread lib needed (uses native Win32 threads)
if(NOT MSVC AND NOT APPLE)
    target_link_libraries(hugine PRIVATE pthread)
endif()

# --------------------------------------------------------------------------
# Print final summary
# --------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== Hugine 5.0 Iota — Build Configuration ===")
message(STATUS "  Source   : ${ENGINE_SRC}")
message(STATUS "  Compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Platform : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Syzygy   : ${USE_SYZYGY}")
message(STATUS "  NNUE     : ${USE_NNUE}")
message(STATUS "  Debug    : ${DEBUG_BUILD}")
message(STATUS "  Native   : ${NATIVE}")
message(STATUS "  Chess960 : ${CHESS960_DEBUG}")
message(STATUS "  Output   : build/hugine")
message(STATUS "")
message(STATUS "  Build with:  cmake --build .")
message(STATUS "")

# --------------------------------------------------------------------------
# Install target (optional — copies binary to CMAKE_INSTALL_PREFIX/bin)
# --------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS hugine RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
