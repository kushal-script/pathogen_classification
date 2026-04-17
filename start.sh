#!/usr/bin/env bash
# start.sh — Environment setup for Pathogen Classification project
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
section() { echo -e "\n${BOLD}━━━ $* ━━━${RESET}"; }

# ── Header ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║   Pathogen Classification — Environment Setup    ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── 1. Python check ───────────────────────────────────────────────────────────
section "Python"
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.10+ and retry."
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_MAJOR=3; REQUIRED_MINOR=10
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt "$REQUIRED_MAJOR" ]] || \
   [[ "$PY_MAJOR" -eq "$REQUIRED_MAJOR" && "$PY_MINOR" -lt "$REQUIRED_MINOR" ]]; then
    error "Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ required (found $PY_VERSION)."
fi
ok "Python $PY_VERSION"

# ── 2. Virtual environment ────────────────────────────────────────────────────
section "Virtual Environment"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at .venv ..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Activated: $(which python)"

# ── 3. Dependencies ───────────────────────────────────────────────────────────
section "Dependencies"
info "Installing / verifying requirements ..."
pip install --quiet --upgrade pip
pip install --quiet -r "$PROJECT_DIR/requirements.txt"
ok "All packages installed"

# Check torch & device
python3 - <<'PYEOF'
import torch, sys
v = torch.__version__
device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"\033[0;32m[OK]\033[0m    PyTorch {v}  |  compute device: {device}")
PYEOF

# ── 4. Checkpoints ───────────────────────────────────────────────────────────
section "Model Checkpoints"
CKPT_DIR="$PROJECT_DIR/checkpoints"
REQUIRED_CHECKPOINTS=(
    "best_efficientnet_b3.pt"
    "best_resnet50.pt"
    "best_vgg16.pt"
    "best_densenet121.pt"
    "best_mobilenet_v3_large.pt"
)
ALL_PRESENT=true
for ckpt in "${REQUIRED_CHECKPOINTS[@]}"; do
    path="$CKPT_DIR/$ckpt"
    if [[ -f "$path" ]]; then
        size=$(du -sh "$path" | cut -f1)
        ok "$ckpt  ($size)"
    else
        warn "MISSING: $ckpt"
        ALL_PRESENT=false
    fi
done
$ALL_PRESENT || warn "Some checkpoints are missing — ensemble inference will fail."

# ── 5. Dataset / sample data ──────────────────────────────────────────────────
section "Data"
SAMPLE_DIR="$PROJECT_DIR/sample_data"
DATASET_DIR="$PROJECT_DIR/dataset/flat"
for cls in bacterial fungal healthy mould; do
    n=$(find "$SAMPLE_DIR/$cls" -type f 2>/dev/null | wc -l | tr -d ' ')
    ok "sample_data/$cls — $n images"
done
if [[ -d "$DATASET_DIR" ]]; then
    total=$(find "$DATASET_DIR" -type f | wc -l | tr -d ' ')
    ok "Full dataset (dataset/flat) — $total images"
else
    warn "Full dataset not found at dataset/flat"
fi

# ── 6. Results folder ─────────────────────────────────────────────────────────
section "Output Directories"
mkdir -p "$PROJECT_DIR/results" "$PROJECT_DIR/predictions"
ok "results/  and  predictions/  ready"

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}Setup complete!${RESET}"
echo -e "  Run ${CYAN}./demo.sh${RESET} to start the presentation demo."
echo -e "  Activate env manually:  ${CYAN}source .venv/bin/activate${RESET}\n"
