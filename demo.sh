#!/usr/bin/env bash
# demo.sh — Presentation demo for Pathogen Classification project
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SRC_DIR="$PROJECT_DIR/src"
SAMPLE_DIR="$PROJECT_DIR/sample_data"
RESULTS_DIR="$PROJECT_DIR/results"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; MAGENTA='\033[0;35m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}  ▸ $*${RESET}"; }
ok()      { echo -e "${GREEN}  ✔ $*${RESET}"; }
warn()    { echo -e "${YELLOW}  ⚠ $*${RESET}"; }
section() { echo -e "\n${BOLD}${MAGENTA}━━━ $* ━━━${RESET}"; }
pause()   { echo -e "\n${YELLOW}  [Press Enter to continue...]${RESET}"; read -r; }

# ── Guard: env must exist ─────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${RED}[ERROR]${RESET} Virtual environment not found. Run ./start.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

clear

# ── Title slide ───────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════════════╗"
echo "  ║         PATHOGEN CLASSIFICATION IN PLANT LEAVES               ║"
echo "  ║         Deep Learning Ensemble — Live Demo                    ║"
echo "  ╚═══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo "  5 CNN architectures · 4 pathogen classes · 92.75% ensemble accuracy"
echo "  Classes: Bacterial | Fungal | Mould | Healthy"
echo ""
echo "  Models: EfficientNet-B3 · ResNet-50 · VGG-16 · DenseNet-121 · MobileNet-V3"

pause

# ── Section 1: Project overview ───────────────────────────────────────────────
section "1 / 4  Project Overview"
python3 - <<'PYEOF'
import torch

device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"

models = [
    ("EfficientNet-B3",     "89.69%",  "10.7M",  "best_efficientnet_b3.pt"),
    ("ResNet-50",           "91.60%",  "23.5M",  "best_resnet50.pt"),
    ("VGG-16",              "87.40%", "134.3M",  "best_vgg16.pt"),
    ("DenseNet-121",        "91.98%",   "7.0M",  "best_densenet121.pt"),
    ("MobileNet-V3-Large",  "91.22%",   "4.2M",  "best_mobilenet_v3_large.pt"),
]

print(f"\n  Compute device: \033[1;32m{device.upper()}\033[0m\n")
print(f"  {'Model':<24} {'Test Acc':>9}  {'Params':>9}  {'Checkpoint'}")
print(f"  {'─'*24} {'─'*9}  {'─'*9}  {'─'*30}")
for name, acc, params, ckpt in models:
    print(f"  {name:<24} {acc:>9}  {params:>9}  {ckpt}")
print(f"\n  {'Ensemble (probability avg)':<24} {'92.75%':>9}")
PYEOF

pause

# ── Section 2: Show existing results ─────────────────────────────────────────
section "2 / 4  Pre-computed Results"

if [[ -f "$RESULTS_DIR/ensemble_report.txt" ]]; then
    echo ""
    cat "$RESULTS_DIR/ensemble_report.txt"
else
    warn "ensemble_report.txt not found — run ensemble evaluation to generate it."
fi

echo ""
info "Opening result visualizations ..."
OPENED=0
for img in "$RESULTS_DIR/model_comparison.png" \
            "$RESULTS_DIR/ensemble_confusion_matrix.png" \
            "$RESULTS_DIR/training_curves.png"; do
    if [[ -f "$img" ]]; then
        open "$img" 2>/dev/null || true
        ok "Opened: $(basename "$img")"
        OPENED=$((OPENED+1))
    fi
done
[[ $OPENED -eq 0 ]] && warn "No result images found in results/"

pause

# ── Section 3: Live single-image prediction ───────────────────────────────────
section "3 / 4  Live Inference — Single Image Prediction"

# Pick one representative image from each class
declare -A DEMO_IMAGES
for cls in bacterial fungal healthy mould; do
    img=$(find "$SAMPLE_DIR/$cls" -type f \( -iname "*.jpg" -o -iname "*.png" \) | head -1)
    if [[ -n "$img" ]]; then
        DEMO_IMAGES[$cls]="$img"
    fi
done

echo ""
echo "  Available sample classes:"
i=1
declare -A IDX_TO_CLS
for cls in bacterial fungal healthy mould; do
    if [[ -v "DEMO_IMAGES[$cls]" ]]; then
        echo "    $i) $cls  →  $(basename "${DEMO_IMAGES[$cls]}")"
        IDX_TO_CLS[$i]=$cls
        i=$((i+1))
    fi
done

echo ""
echo -n "  Choose a class to predict (1-$((i-1)), or Enter for all): "
read -r CHOICE

run_prediction() {
    local cls="$1"
    local img="${DEMO_IMAGES[$cls]}"
    echo ""
    echo -e "  ${BOLD}Running ensemble prediction on: $(basename "$img")${RESET}"
    echo -e "  True label: ${CYAN}$cls${RESET}"
    echo ""
    (cd "$SRC_DIR" && python3 predict.py "$img")
    echo ""
    # Open the latest prediction folder
    LATEST_PRED=$(ls -td "$PROJECT_DIR/predictions"/*/ 2>/dev/null | head -1)
    if [[ -n "$LATEST_PRED" ]]; then
        ok "Grad-CAM saved to: $LATEST_PRED"
        open "$LATEST_PRED" 2>/dev/null || true
    fi
}

if [[ -z "$CHOICE" ]]; then
    for cls in bacterial fungal healthy mould; do
        [[ -v "DEMO_IMAGES[$cls]" ]] && run_prediction "$cls"
    done
elif [[ -v "IDX_TO_CLS[$CHOICE]" ]]; then
    run_prediction "${IDX_TO_CLS[$CHOICE]}"
else
    warn "Invalid choice — skipping live prediction."
fi

pause

# ── Section 4: (Optional) Full ensemble evaluation ────────────────────────────
section "4 / 4  Full Ensemble Evaluation (optional)"
echo ""
echo -n "  Run full ensemble evaluation on the test set? (y/N): "
read -r RUN_ENSEMBLE

if [[ "${RUN_ENSEMBLE,,}" == "y" ]]; then
    echo ""
    info "Running ensemble evaluation — this may take a few minutes ..."
    (cd "$SRC_DIR" && python3 ensemble.py)
    ok "Evaluation complete. Results saved to results/"
    open "$RESULTS_DIR" 2>/dev/null || true
else
    info "Skipped — pre-computed results shown in Section 2."
fi

# ── Closing ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║   Demo complete — thank you!                  ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo -e "${RESET}"
echo "  Key results:"
echo "    • Ensemble accuracy : 92.75%"
echo "    • Macro AUC-ROC     : 99.05%"
echo "    • Dataset           : 1,742 leaf images (4 classes)"
echo "    • Architectures     : EfficientNet-B3, ResNet-50, VGG-16,"
echo "                          DenseNet-121, MobileNet-V3-Large"
echo ""
