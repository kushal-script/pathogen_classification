#!/usr/bin/env bash
#
# demo.sh
# Interactive presentation demo for the Pathogen Classification project.
#
# Walks through four sections:
#   1. Project overview  — model architectures and test accuracies
#   2. Pre-computed results — ensemble report and saved visualizations
#   3. Live inference — user picks an image from demonstration_images/,
#                       ensemble predicts, actual vs predicted class is shown
#   4. Full evaluation (optional) — reruns ensemble.py on the entire test set
#
# Prerequisites:
#   Run ./start.sh at least once to create the virtual environment and
#   demonstration_images/ before launching this script.
#
# Usage:
#   chmod +x demo.sh && ./demo.sh
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SRC_DIR="$PROJECT_DIR/src"
DEMO_IMG_DIR="$PROJECT_DIR/demonstration_images"
RESULTS_DIR="$PROJECT_DIR/results"
PRED_DIR="$PROJECT_DIR/predictions"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; MAGENTA='\033[0;35m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}  ▸ $*${RESET}"; }
ok()      { echo -e "${GREEN}  ✔ $*${RESET}"; }
warn()    { echo -e "${YELLOW}  ⚠ $*${RESET}"; }
section() { echo -e "\n${BOLD}${MAGENTA}━━━ $* ━━━${RESET}"; }
pause()   { echo -e "\n${YELLOW}  [Press Enter to continue...]${RESET}"; read -r _dummy || true; }
hr()      { echo -e "  ${BOLD}──────────────────────────────────────────${RESET}"; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${RED}[ERROR]${RESET} Virtual environment not found. Run ./start.sh first."
    exit 1
fi
if [[ ! -d "$DEMO_IMG_DIR" || ! -f "$DEMO_IMG_DIR/manifest.json" ]]; then
    echo -e "${RED}[ERROR]${RESET} demonstration_images/ not found. Run ./start.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
clear 2>/dev/null || true

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════════════╗"
echo "  ║       PATHOGEN CLASSIFICATION IN PLANT LEAVES                 ║"
echo "  ║       Deep Learning Ensemble — Live Demo                      ║"
echo "  ╚═══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo "  5 CNN architectures · 4 pathogen classes · 92.75% ensemble accuracy"
echo "  Classes: Bacterial | Fungal | Mould | Healthy"
echo ""
echo "  Models: EfficientNet-B3 · ResNet-50 · VGG-16 · DenseNet-121 · MobileNet-V3"

pause

# ---------------------------------------------------------------------------
# Section 1 — Project overview
# Prints active compute device and per-model accuracy / parameter count.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Section 2 — Pre-computed results
# Displays the ensemble classification report and lists saved visualizations.
# ---------------------------------------------------------------------------
section "2 / 4  Pre-computed Results"

if [[ -f "$RESULTS_DIR/ensemble_report.txt" ]]; then
    echo ""
    cat "$RESULTS_DIR/ensemble_report.txt"
else
    warn "ensemble_report.txt not found — run ensemble evaluation to generate it."
fi

echo ""
info "Result visualizations saved in:  results/"
for img in model_comparison.png ensemble_confusion_matrix.png training_curves.png; do
    [[ -f "$RESULTS_DIR/$img" ]] && ok "$img" || warn "Missing: $img"
done

pause

# ---------------------------------------------------------------------------
# Section 3 — Live inference
#
# The user selects an image number from demonstration_images/.
# The ensemble runs predict.py and the terminal displays:
#   - per-model predictions and confidences
#   - ensemble result
#   - actual class (from manifest.json) vs predicted class
#   - CORRECT / INCORRECT verdict
#
# All interactive UI output goes to stderr so it appears immediately even
# though stdout is captured by the surrounding $() subshell.
# ---------------------------------------------------------------------------
section "3 / 4  Live Inference — Demonstration Images"
echo ""

CHOSEN_IMAGE=$(python3 - "$DEMO_IMG_DIR" <<'PYEOF'
import sys, os, json, re

DEMO_DIR = sys.argv[1]
ui       = sys.stderr

def p(*args, **kwargs):
    print(*args, **kwargs, file=ui)

def prompt(msg):
    ui.write(msg)
    ui.flush()
    with open("/dev/tty") as tty:
        return tty.readline().strip()

def img_num(name):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 0

with open(os.path.join(DEMO_DIR, "manifest.json")) as f:
    manifest = json.load(f)

images = sorted(manifest.keys(), key=img_num)
total  = len(images)

p(f"  Available images:  image_1  –  image_{total}\n")

while True:
    raw = prompt(f"  Enter image number (1-{total}): ")
    if raw.isdigit() and 1 <= int(raw) <= total:
        chosen = images[int(raw) - 1]
        p(f"\n  \033[0;32m  ✔ Selected: \033[1m{chosen}\033[0m")
        break
    p(f"  \033[1;33m  ⚠ Please enter a number between 1 and {total}.\033[0m")

print(os.path.join(DEMO_DIR, chosen), end="")
PYEOF
)

# ---------------------------------------------------------------------------
# run_prediction: given an image path, looks up its ground-truth class from
# manifest.json, runs ensemble inference, and prints the result verdict.
# ---------------------------------------------------------------------------
run_prediction() {
    local image_path="$1"
    local image_name
    image_name=$(basename "$image_path")

    local actual_class actual_disease
    actual_class=$(python3 -c "
import json
with open('$DEMO_IMG_DIR/manifest.json') as f:
    m = json.load(f)
print(m.get('$image_name', {}).get('class', 'unknown'), end='')
" 2>/dev/null || echo "unknown")
    actual_disease=$(python3 -c "
import json
with open('$DEMO_IMG_DIR/manifest.json') as f:
    m = json.load(f)
print(m.get('$image_name', {}).get('disease', ''), end='')
" 2>/dev/null || echo "")

    echo ""
    hr
    echo ""
    echo -e "  ${BOLD}Running ensemble prediction...${RESET}"
    echo -e "  Image   : ${CYAN}${image_name}${RESET}"
    echo -e "  Actual  : ${CYAN}${actual_class}${RESET}  (${actual_disease})"
    echo ""

    mkdir -p "$PRED_DIR"
    (cd "$SRC_DIR" && python3 predict.py "$image_path")

    local latest_pred
    latest_pred=$(ls -td "$PRED_DIR"/*/ 2>/dev/null | head -1)
    if [[ -n "$latest_pred" && -f "${latest_pred}summary.txt" ]]; then
        local predicted_class actual_lower predicted_lower
        predicted_class=$(grep -i "^ENSEMBLE (avg)" "${latest_pred}summary.txt" | awk '{print $3}')
        actual_lower=$(echo "$actual_class"    | tr '[:upper:]' '[:lower:]')
        predicted_lower=$(echo "$predicted_class" | tr '[:upper:]' '[:lower:]')

        echo ""
        echo -e "  ${BOLD}━━━ Classification Result ━━━━━━━━━━━━━━━━━━━━${RESET}"
        echo -e "  Actual class    : ${CYAN}${BOLD}${actual_class}${RESET}"
        echo -e "  Predicted class : ${CYAN}${BOLD}${predicted_class}${RESET}"
        if [[ "$actual_lower" == "$predicted_lower" ]]; then
            echo -e "  Verdict         : ${GREEN}${BOLD}✔  CORRECT${RESET}"
        else
            echo -e "  Verdict         : ${RED}${BOLD}✘  INCORRECT${RESET}"
        fi
        echo -e "  ${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
        echo ""
        ok "Grad-CAM + full report saved to:  $latest_pred"
    else
        warn "Could not read prediction output. Check predictions/ folder."
    fi
}

# Run the first prediction (image already chosen above)
run_prediction "$CHOSEN_IMAGE"

# ---------------------------------------------------------------------------
# Section 4 — Continue predicting or end
# Loops until the user enters N, allowing multiple live predictions.
# ---------------------------------------------------------------------------
section "4 / 4  Continue Demo"

MANIFEST_TOTAL=$(python3 -c "
import json
with open('$DEMO_IMG_DIR/manifest.json') as f:
    print(len(json.load(f)))
")

while true; do
    echo ""
    echo -n "  Predict another image? (y/N): "
    read -r ANOTHER || ANOTHER="N"

    if [[ "$(echo "$ANOTHER" | tr '[:upper:]' '[:lower:]')" != "y" ]]; then
        break
    fi

    NEXT_IMAGE=$(python3 - "$DEMO_IMG_DIR" "$MANIFEST_TOTAL" <<'PYEOF'
import sys, os, json, re

DEMO_DIR = sys.argv[1]
total    = int(sys.argv[2])
ui       = sys.stderr

def p(*args, **kwargs):
    print(*args, **kwargs, file=ui)

def prompt(msg):
    ui.write(msg); ui.flush()
    with open("/dev/tty") as tty:
        return tty.readline().strip()

def img_num(name):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 0

with open(os.path.join(DEMO_DIR, "manifest.json")) as f:
    manifest = json.load(f)
images = sorted(manifest.keys(), key=img_num)

p(f"\n  Available images:  image_1  –  image_{total}\n")
while True:
    raw = prompt(f"  Enter image number (1-{total}): ")
    if raw.isdigit() and 1 <= int(raw) <= total:
        chosen = images[int(raw) - 1]
        p(f"\n  \033[0;32m  ✔ Selected: \033[1m{chosen}\033[0m")
        break
    p(f"  \033[1;33m  ⚠ Please enter a number between 1 and {total}.\033[0m")

print(os.path.join(DEMO_DIR, chosen), end="")
PYEOF
    )

    run_prediction "$NEXT_IMAGE"
done

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
