# =============================================================================
# Hugine 3.0 "Gama" — Windows Cross-Build Script (All Variants)
# =============================================================================
# Produces 24 binaries (6 variants × 4 arches) in .\build\ :
#
#   Variants per arch:
#     _base      O3, no Syzygy, no NNUE
#     _syzygy    O3, Syzygy via jdart1/Fathom   (needs Fathom\)
#     _nnue      O3, NNUE neural-net eval         (needs *.nnue file)
#     _full      O3, Syzygy + NNUE               (needs both)
#     _chess960  O3, Chess960 extra debug tracing
#     _debug     O0 -g, debug assertions
#
#   Architectures:
#     hugine_winx86_{v}.exe    Windows x86-64   (llvm-mingw)
#     hugine_winARM_{v}.exe    Windows ARM64    (llvm-mingw)
#     hugine_linux86_{v}       Linux x86-64     (musl.cc static)
#     hugine_linuxARM_{v}      Linux ARM64      (musl.cc static)
#
# Toolchains downloaded automatically on first run.
#
# Usage:
#   .\windows_cross.ps1
#   .\windows_cross.ps1 -SkipLinux
#   .\windows_cross.ps1 -SkipWindows
#   .\windows_cross.ps1 -OnlyVariants base,syzygy
#   .\windows_cross.ps1 -OnlyArches winx86,linux86
# =============================================================================

param(
    [switch]$SkipLinux                  = $false,
    [switch]$SkipWindows                = $false,
    [string]$OnlyVariants               = "",  # comma-sep: base,syzygy,nnue,full,chess960,debug
    [string]$OnlyArches                 = "",  # comma-sep: winx86,winARM,linux86,linuxARM
    [switch]$Help                       = $false
)

if ($Help) {
    Write-Host @"
Hugine Windows Cross-Build Script
  -SkipLinux              Build Windows targets only
  -SkipWindows            Build Linux targets only
  -OnlyVariants v1,v2     Restrict to specific variants (base|syzygy|nnue|full|chess960|debug)
  -OnlyArches a1,a2       Restrict to specific arches  (winx86|winARM|linux86|linuxARM)
  -Help                   Show this message
"@
    exit 0
}

$ErrorActionPreference = "Stop"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir    = Join-Path $ScriptDir "build"
$ToolDir     = Join-Path $ScriptDir "toolchains"
$SrcFile     = Join-Path $ScriptDir "hugine-gama-v5.cpp"
$FathomDir   = Join-Path $ScriptDir "Fathom"
$FathomSrc   = Join-Path $FathomDir "src\tbprobe.c"
$FathomInc   = Join-Path $FathomDir "src"

$NnueFile = (Get-ChildItem $ScriptDir -Filter "*.nnue" -ErrorAction SilentlyContinue |
             Select-Object -First 1)
if (-not $NnueFile) {
    $NnueFile = (Get-ChildItem $ScriptDir -Filter "nn-*.bin" -ErrorAction SilentlyContinue |
                 Select-Object -First 1)
}
$HasNnue   = ($NnueFile -ne $null)
$HasFathom = (Test-Path $FathomSrc)

# Toolchain URLs
$LlvmVer    = "20240619"
$LlvmUrl    = "https://github.com/mstorsjo/llvm-mingw/releases/download/$LlvmVer/llvm-mingw-$LlvmVer-ucrt-x86_64.zip"
$LlvmDir    = Join-Path $ToolDir "llvm-mingw"
$MuslX86Url = "https://musl.cc/x86_64-linux-musl-cross.tgz"
$MuslArmUrl = "https://musl.cc/aarch64-linux-musl-cross.tgz"
$MuslX86Dir = Join-Path $ToolDir "x86_64-linux-musl-cross"
$MuslArmDir = Join-Path $ToolDir "aarch64-linux-musl-cross"

# --------------------------------------------------------------------------
# Filter sets
# --------------------------------------------------------------------------
$AllVariants = @("base","syzygy","nnue","full","chess960","debug")
$AllArches   = @("winx86","winARM","linux86","linuxARM")

$ActiveVariants = if ($OnlyVariants) { $OnlyVariants -split "," | ForEach-Object { $_.Trim() } } else { $AllVariants }
$ActiveArches   = if ($OnlyArches)   { $OnlyArches   -split "," | ForEach-Object { $_.Trim() } } else { $AllArches }
if ($SkipWindows) { $ActiveArches = $ActiveArches | Where-Object { $_ -notmatch "^win" } }
if ($SkipLinux)   { $ActiveArches = $ActiveArches | Where-Object { $_ -notmatch "^linux" } }

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
function Write-Info  { param($m) Write-Host "[info]  $m" -ForegroundColor Cyan }
function Write-Ok    { param($m) Write-Host "[ok]    $m" -ForegroundColor Green }
function Write-Warn  { param($m) Write-Host "[warn]  $m" -ForegroundColor Yellow }
function Write-Fail  { param($m) Write-Host "[error] $m" -ForegroundColor Red }
function Write-Step  { param($m) Write-Host "`n==> $m" -ForegroundColor White }
function Write-Skip  { param($m) Write-Host "[skip]  $m" -ForegroundColor DarkGray }

function Download-File {
    param([string]$Url, [string]$Dest)
    $leaf = Split-Path -Leaf $Dest
    Write-Info "Downloading $leaf ..."
    if (Get-Command "curl.exe" -ErrorAction SilentlyContinue) {
        & curl.exe -L --progress-bar -o $Dest $Url
    } else {
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
        $ProgressPreference = "Continue"
    }
    Write-Ok "Downloaded: $leaf"
}

function Expand-Archive-Any {
    param([string]$Archive, [string]$DestDir)
    $leaf = Split-Path -Leaf $Archive
    if ($leaf -match "\.zip$") {
        Expand-Archive -Path $Archive -DestinationPath $DestDir -Force
    } elseif ($leaf -match "\.(tgz|tar\.gz|tar\.xz)$") {
        if (Get-Command "tar.exe" -ErrorAction SilentlyContinue) {
            & tar.exe -xf $Archive -C $DestDir
        } elseif (Get-Command "7z" -ErrorAction SilentlyContinue) {
            $tmpTar = $Archive -replace "\.tgz$|\.tar\.gz$|\.tar\.xz$",".tar"
            & 7z x $Archive -o"$DestDir" -y | Out-Null
            if (Test-Path $tmpTar) { & 7z x $tmpTar -o"$DestDir" -y | Out-Null; Remove-Item $tmpTar -Force }
        } else {
            throw "Cannot extract $leaf — requires tar.exe (Win10+) or 7-Zip"
        }
    }
}

# Build a Fathom C object file for a given CC
function Build-FathomObj {
    param([string]$CC, [string]$ObjOut, [string[]]$ExtraArgs = @())
    if (-not $HasFathom) { return $null }
    Write-Info "  Fathom obj: $(Split-Path -Leaf $ObjOut) ..."
    $args = @("-O2","-D_DEFAULT_SOURCE") + $ExtraArgs + @("-I$FathomInc","-c",$FathomSrc,"-o",$ObjOut)
    & $CC @args
    if ($LASTEXITCODE -eq 0) { return $ObjOut } else { Write-Warn "  Fathom failed"; return $null }
}

# Compile one binary
function Compile-One {
    param(
        [string]   $CXX,
        [string]   $Output,
        [string[]] $BaseFlags,     # target, static, etc
        [string[]] $SyzygyFlags,   # -DUSE_SYZYGY/-DNO_SYZYGY + include
        [string]   $FathomObj,     # path or ""
        [string[]] $OptFlags,      # -O3 -DNDEBUG  or  -O0 -g -DDEBUG
        [string[]] $ExtraDefs      # -DUSE_NNUE, -DCHESS960_EXTRA_DEBUG, etc
    )
    $name = Split-Path -Leaf $Output

    # Skip NNUE variants when no .nnue file present
    if ($ExtraDefs -contains "-DUSE_NNUE" -and -not $HasNnue) {
        Write-Skip "$name  (no .nnue file)"
        return $false
    }

    Write-Info "  $name ..."

    $sobjs = @()
    if ($FathomObj -and (Test-Path $FathomObj)) { $sobjs = @($FathomObj) }

    $cmdArgs = @("-std=c++17") `
        + $OptFlags `
        + @("-Wall","-Wextra","-Wno-unused-parameter","-pthread") `
        + $BaseFlags `
        + $SyzygyFlags `
        + $ExtraDefs `
        + @($SrcFile) `
        + $sobjs `
        + @("-o",$Output,"-lpthread")

    & $CXX @cmdArgs 2>&1 | Where-Object { $_ -match "error:" } | Select-Object -First 3 | Write-Warn

    if (Test-Path $Output) {
        $kb = [math]::Round((Get-Item $Output).Length / 1KB)
        Write-Ok ("    {0,-40} {1,6} KB" -f $name, $kb)
        return $true
    } else {
        Write-Warn "    FAILED: $name"
        return $false
    }
}

# Build all requested variants for one architecture
function Build-ArchVariants {
    param(
        [string]   $Tag,           # winx86 | winARM | linux86 | linuxARM
        [string]   $CXX,           # C++ compiler
        [string]   $CC,            # C   compiler (for Fathom)
        [string[]] $BaseFlags,     # -static --target=... etc
        [string[]] $CFathomFlags,  # extra C flags for Fathom (--target=...)
        [string]   $Ext            # ".exe" or ""
    )

    if ($ActiveArches -notcontains $Tag) {
        Write-Skip "Architecture $Tag"
        return
    }

    Write-Step "Building $Tag  ($($ActiveVariants.Count) variants)"

    # Pre-build Fathom objects (release + full)
    $FobjRel  = $null
    $FobjFull = $null
    if ($HasFathom) {
        $FobjRel  = Build-FathomObj -CC $CC -ObjOut (Join-Path $BuildDir "tb_${Tag}_r.o") -ExtraArgs $CFathomFlags
        $FobjFull = Build-FathomObj -CC $CC -ObjOut (Join-Path $BuildDir "tb_${Tag}_f.o") -ExtraArgs $CFathomFlags
    }

    $SfOn  = @("-DUSE_SYZYGY", "-I$FathomInc")
    $SfOff = @("-DNO_SYZYGY")
    $Opt   = @("-O3","-DNDEBUG")
    $DbgO  = @("-O0","-g","-DDEBUG")

    $VariantMap = [ordered]@{
        "base"       = @{ SF=$SfOff; FO=$null;     Opt=$Opt;  Extra=@() }
        "syzygy"     = @{ SF=$SfOn;  FO=$FobjRel;  Opt=$Opt;  Extra=@() }
        "nnue"       = @{ SF=$SfOff; FO=$null;     Opt=$Opt;  Extra=@("-DUSE_NNUE") }
        "nnue_large" = @{ SF=$SfOff; FO=$null;     Opt=$Opt;  Extra=@("-DUSE_NNUE", "-DNNUE_LARGE") }
        "full"       = @{ SF=$SfOn;  FO=$FobjFull; Opt=$Opt;  Extra=@("-DUSE_NNUE") }
        "chess960"   = @{ SF=$SfOff; FO=$null;     Opt=$Opt;  Extra=@("-DCHESS960_EXTRA_DEBUG") }
        "debug"      = @{ SF=$SfOff; FO=$null;     Opt=$DbgO; Extra=@() }
    }

    foreach ($v in $ActiveVariants) {
        if (-not $VariantMap.Contains($v)) { Write-Warn "Unknown variant: $v"; continue }
        $vd = $VariantMap[$v]

        # Skip syzygy/full when Fathom not present
        if (($v -eq "syzygy" -or $v -eq "full") -and -not $HasFathom) {
            Write-Skip "hugine_${Tag}_${v}${Ext}  (Fathom not found)"
            continue
        }

        $out = Join-Path $BuildDir "hugine_${Tag}_${v}${Ext}"
        Compile-One `
            -CXX $CXX -Output $out `
            -BaseFlags $BaseFlags `
            -SyzygyFlags $vd.SF `
            -FathomObj ($vd.FO ?? "") `
            -OptFlags $vd.Opt `
            -ExtraDefs $vd.Extra | Out-Null
    }
}

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
if (-not (Test-Path $SrcFile)) {
    Write-Fail "Source not found: $SrcFile"
    exit 1
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $ToolDir  | Out-Null

Write-Host "`nHugine 3.0 Gama — Windows Cross-Build" -ForegroundColor White
Write-Host "  Variants : $($ActiveVariants -join ', ')"
Write-Host "  Arches   : $($ActiveArches -join ', ')"
Write-Host "  Fathom   : $(if ($HasFathom) { 'found' } else { 'NOT found — syzygy/full will be skipped' })"
Write-Host "  NNUE     : $(if ($HasNnue)   { $NnueFile.Name } else { 'NOT found — nnue/full will be skipped' })"

# --------------------------------------------------------------------------
# Download llvm-mingw (handles all Windows targets)
# --------------------------------------------------------------------------
Write-Step "Toolchain: llvm-mingw"
if (-not (Test-Path $LlvmDir)) {
    $zip = Join-Path $ToolDir "llvm-mingw.zip"
    Download-File -Url $LlvmUrl -Dest $zip
    Write-Info "Extracting llvm-mingw ..."
    Expand-Archive-Any -Archive $zip -DestDir $ToolDir
    $extracted = Get-ChildItem $ToolDir -Directory | Where-Object { $_.Name -like "llvm-mingw-*" } | Select-Object -First 1
    if ($extracted -and $extracted.FullName -ne $LlvmDir) {
        Rename-Item -Path $extracted.FullName -NewName "llvm-mingw"
    }
    Remove-Item $zip -Force
    Write-Ok "llvm-mingw ready"
} else { Write-Ok "llvm-mingw already present" }

$LlvmBin = Join-Path $LlvmDir "bin"

# --------------------------------------------------------------------------
# Download musl cross-compilers (Linux static targets)
# --------------------------------------------------------------------------
if (-not $SkipLinux -and ($ActiveArches -match "linux")) {
    Write-Step "Toolchain: musl cross-compilers (Linux static)"

    if (-not (Test-Path $MuslX86Dir)) {
        $t = Join-Path $ToolDir "x86-musl.tgz"
        Download-File -Url $MuslX86Url -Dest $t
        Write-Info "Extracting musl x86_64 ..."
        Expand-Archive-Any -Archive $t -DestDir $ToolDir
        Remove-Item $t -Force -ErrorAction SilentlyContinue
        Write-Ok "musl x86_64 ready"
    } else { Write-Ok "musl x86_64 already present" }

    if (-not (Test-Path $MuslArmDir)) {
        $t = Join-Path $ToolDir "arm-musl.tgz"
        Download-File -Url $MuslArmUrl -Dest $t
        Write-Info "Extracting musl aarch64 ..."
        Expand-Archive-Any -Archive $t -DestDir $ToolDir
        Remove-Item $t -Force -ErrorAction SilentlyContinue
        Write-Ok "musl aarch64 ready"
    } else { Write-Ok "musl aarch64 already present" }
}

# --------------------------------------------------------------------------
# Build all architectures
# --------------------------------------------------------------------------

# hugine_winx86 — llvm-mingw x86_64
Build-ArchVariants -Tag "winx86" `
    -CXX (Join-Path $LlvmBin "x86_64-w64-mingw32-clang++.exe") `
    -CC  (Join-Path $LlvmBin "x86_64-w64-mingw32-clang.exe") `
    -BaseFlags    @("-static","--target=x86_64-w64-mingw32") `
    -CFathomFlags @() `
    -Ext ".exe"

# hugine_winARM — llvm-mingw aarch64
Build-ArchVariants -Tag "winARM" `
    -CXX (Join-Path $LlvmBin "aarch64-w64-mingw32-clang++.exe") `
    -CC  (Join-Path $LlvmBin "aarch64-w64-mingw32-clang.exe") `
    -BaseFlags    @("-static","--target=aarch64-w64-mingw32") `
    -CFathomFlags @("--target=aarch64-w64-mingw32") `
    -Ext ".exe"

# hugine_linux86 — musl x86_64
if (-not $SkipLinux) {
    $MX = Join-Path $MuslX86Dir "bin"
    $cxx = Join-Path $MX "x86_64-linux-musl-g++"
    if (-not (Test-Path $cxx)) { $cxx = "$cxx.exe" }
    $cc  = (Join-Path $MX "x86_64-linux-musl-gcc")
    if (-not (Test-Path $cc)) { $cc = "$cc.exe" }
    Build-ArchVariants -Tag "linux86" `
        -CXX $cxx -CC $cc `
        -BaseFlags @("-static") -CFathomFlags @() -Ext ""

    # hugine_linuxARM — musl aarch64
    $MA = Join-Path $MuslArmDir "bin"
    $cxx = Join-Path $MA "aarch64-linux-musl-g++"
    if (-not (Test-Path $cxx)) { $cxx = "$cxx.exe" }
    $cc  = Join-Path $MA "aarch64-linux-musl-gcc"
    if (-not (Test-Path $cc)) { $cc = "$cc.exe" }
    Build-ArchVariants -Tag "linuxARM" `
        -CXX $cxx -CC $cc `
        -BaseFlags @("-static") -CFathomFlags @() -Ext ""
}

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
Write-Host "`n=== Build Matrix Results ===" -ForegroundColor White
Write-Host "Directory: $BuildDir`n"

$total = 0; $built = 0
foreach ($arch in @("winx86","winARM","linux86","linuxARM")) {
    $ext = if ($arch -match "^win") { ".exe" } else { "" }
    Write-Host "  $arch" -ForegroundColor White
    foreach ($v in $AllVariants) {
        $total++
        $f = Join-Path $BuildDir "hugine_${arch}_${v}${ext}"
        if (Test-Path $f) {
            $kb = [math]::Round((Get-Item $f).Length / 1KB)
            Write-Host ("    [OK]  hugine_{0}_{1}{2}  {3,6} KB" -f $arch,$v,$ext,$kb) -ForegroundColor Green
            $built++
        } else {
            Write-Host ("    [--]  hugine_{0}_{1}{2}" -f $arch,$v,$ext) -ForegroundColor DarkGray
        }
    }
    Write-Host ""
}
Write-Host ("Built: {0} / {1}" -f $built, $total) -ForegroundColor $(if ($built -gt 0) { "Green" } else { "Red" })
