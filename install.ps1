<#
.SYNOPSIS
  One-shot setup: virtuelle Umgebung, Abhaengigkeiten, PixOOD.

.AUFRUF (aus dem Repo-Root):
  .\install.ps1
  .\install.ps1 -SkipPixOOD
#>

param(
    [switch]$SkipPixOOD,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

Write-Host "=== OoD-ROI-Benchmark Setup ===" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Virtuelle Umgebung
# ---------------------------------------------------------------------------
$venv = Join-Path $Root ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "[1/4] Erstelle virtuelle Umgebung (.venv)..."
    & $Python -m venv $venv
} else {
    Write-Host "[1/4] .venv existiert bereits - wird wiederverwendet."
}
$pip = Join-Path $venv "Scripts\pip.exe"
$py  = Join-Path $venv "Scripts\python.exe"

# ---------------------------------------------------------------------------
# 2. PyTorch mit CUDA (vor requirements, damit pip nicht die CPU-Version zieht)
# ---------------------------------------------------------------------------
Write-Host "[2/4] Installiere PyTorch (CUDA 12.1)..."
& $pip install --upgrade pip
& $pip install "torch==2.2.2" "torchvision==0.17.2" --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 3. Restliche Abhaengigkeiten
# ---------------------------------------------------------------------------
Write-Host "[3/4] Installiere requirements.txt..."
& $pip install -r (Join-Path $Root "requirements.txt")

# ---------------------------------------------------------------------------
# 4. PixOOD als Schwester-Repo (compute_score_maps.py erwartet ../PixOOD)
# ---------------------------------------------------------------------------
if (-not $SkipPixOOD) {
    $pixood = Join-Path (Split-Path $Root -Parent) "PixOOD"
    if (-not (Test-Path $pixood)) {
        Write-Host "[4/4] Klone PixOOD nach $pixood ..."
        git clone https://github.com/vojirt/PixOOD.git $pixood
        Write-Host "      WICHTIG: Checkpoints gemaess PixOOD-README herunterladen" -ForegroundColor Yellow
        Write-Host "      (offizielle Cityscapes-Checkpoints) und in $pixood ablegen." -ForegroundColor Yellow
    } else {
        Write-Host "[4/4] PixOOD existiert bereits: $pixood"
    }
} else {
    Write-Host "[4/4] PixOOD uebersprungen (-SkipPixOOD)."
}

Write-Host ""
Write-Host "=== Setup abgeschlossen ===" -ForegroundColor Green
Write-Host "Aktivieren mit:  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Hinweise:"
Write-Host " - SegFormer-B2 & DINOv2 Gewichte werden beim ersten Lauf automatisch geladen."
Write-Host " - RbA wird NICHT lokal installiert (Detectron2 + CUDA-Kernel, s. README §2)."
Write-Host "   Score Maps stattdessen laden:   python scripts/download_score_maps.py"
Write-Host "   oder selbst berechnen (Colab):  colab/README.md"
Write-Host " - Datensaetze: siehe data/README.md"
