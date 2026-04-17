#!/usr/bin/env bash
#
# start.sh
# One-time environment setup for the Pathogen Classification project.
#
# Steps performed:
#   1. Validate Python >= 3.8
#   2. Create and activate a virtual environment (.venv)
#   3. Install all Python dependencies from requirements.txt
#   4. Verify trained model checkpoints are present
#   5. Report dataset statistics
#   6. Build the demonstration_images/ directory used by demo.sh
#   7. Create output directories (results/, predictions/)
#
# Usage:
#   chmod +x start.sh && ./start.sh
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
section() { echo -e "\n${BOLD}━━━ $* ━━━${RESET}"; }

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║   Pathogen Classification — Environment Setup    ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ---------------------------------------------------------------------------
# 1. Python version check
# ---------------------------------------------------------------------------
section "Python"
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.8+ and retry."
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_MAJOR=3; REQUIRED_MINOR=8
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt "$REQUIRED_MAJOR" ]] || \
   [[ "$PY_MAJOR" -eq "$REQUIRED_MAJOR" && "$PY_MINOR" -lt "$REQUIRED_MINOR" ]]; then
    error "Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ required (found $PY_VERSION)."
fi
ok "Python $PY_VERSION"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------
section "Virtual Environment"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at .venv ..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Activated: $(which python)"

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
section "Dependencies"
info "Installing / verifying requirements ..."
pip install --quiet --upgrade pip
pip install --quiet -r "$PROJECT_DIR/requirements.txt"
ok "All packages installed"

python3 - <<'PYEOF'
import torch
device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"\033[0;32m[OK]\033[0m    PyTorch {torch.__version__}  |  compute device: {device}")
PYEOF

# ---------------------------------------------------------------------------
# 4. Model checkpoints
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 5. Dataset statistics
# ---------------------------------------------------------------------------
section "Dataset"
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

# ---------------------------------------------------------------------------
# 6. Demonstration images
#
# Selects up to 10 images per disease group from the latter half of each
# class folder (sorted alphabetically), copies them into a flat directory,
# and writes manifest.json mapping each image_N file to its class and disease.
#
# Re-run: delete demonstration_images/ and run start.sh again.
# ---------------------------------------------------------------------------
section "Demonstration Images"
DEMO_DIR="$PROJECT_DIR/demonstration_images"

if [[ -d "$DEMO_DIR" ]]; then
    warn "demonstration_images/ already exists — skipping regeneration."
    info "Delete the folder and rerun start.sh to regenerate."
else
    info "Selecting random images from dataset (last half of each disease group)..."
    python3 - "$PROJECT_DIR" <<'PYEOF'
import os, re, random, shutil, sys, json

PROJECT_DIR = sys.argv[1]
FLAT_DIR    = os.path.join(PROJECT_DIR, "dataset", "flat")
DEMO_DIR    = os.path.join(PROJECT_DIR, "demonstration_images")
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
PICK_COUNT  = 10

def get_prefix(filename):
    """Strip the trailing numeric index and extension to get the disease prefix."""
    return re.sub(r'_\d+\.[^.]+$', '', filename)

random.seed(42)
os.makedirs(DEMO_DIR, exist_ok=True)

manifest = {}
counter  = 1

for cls in ["bacterial", "fungal", "healthy", "mould"]:
    cls_dir = os.path.join(FLAT_DIR, cls)
    if not os.path.isdir(cls_dir):
        continue

    all_files = sorted(
        f for f in os.listdir(cls_dir)
        if os.path.splitext(f)[1] in IMG_EXTS
    )

    groups = {}
    for f in all_files:
        groups.setdefault(get_prefix(f), []).append(f)

    for pfx, files in sorted(groups.items()):
        files  = sorted(files)
        pool   = files[len(files) // 2:]   # avoid early/repeated images
        count  = min(PICK_COUNT, len(pool))
        chosen = sorted(random.sample(pool, count))

        for fname in chosen:
            ext       = os.path.splitext(fname)[1].lower()
            dest_name = f"image_{counter}{ext}"
            shutil.copy2(os.path.join(cls_dir, fname),
                         os.path.join(DEMO_DIR, dest_name))
            manifest[dest_name] = {
                "class":   cls,
                "disease": pfx.replace("_", " ")
            }
            counter += 1

        print(f"\033[0;32m[OK]\033[0m    {cls} / {pfx.replace('_',' ')}  →  {count} images")

with open(os.path.join(DEMO_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

total = counter - 1
print(f"\n\033[0;32m[OK]\033[0m    {total} demonstration images  (image_1 – image_{total})")
print(f"\033[0;32m[OK]\033[0m    Manifest written to demonstration_images/manifest.json")
PYEOF
fi

# ---------------------------------------------------------------------------
# 7. Output directories
# ---------------------------------------------------------------------------
section "Output Directories"
mkdir -p "$PROJECT_DIR/results" "$PROJECT_DIR/predictions"
ok "results/  and  predictions/  ready"

echo -e "\n${GREEN}${BOLD}Setup complete!${RESET}"
echo -e "  Run ${CYAN}./demo.sh${RESET} to launch the presentation demo."
echo -e "  Activate env manually: ${CYAN}source .venv/bin/activate${RESET}\n"
