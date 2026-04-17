#!/usr/bin/env bash
# demo.sh — Presentation demo for Pathogen Classification project
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SRC_DIR="$PROJECT_DIR/src"
DATASET_DIR="$PROJECT_DIR/dataset/flat"
RESULTS_DIR="$PROJECT_DIR/results"
PRED_DIR="$PROJECT_DIR/predictions"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; MAGENTA='\033[0;35m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}  ▸ $*${RESET}"; }
ok()      { echo -e "${GREEN}  ✔ $*${RESET}"; }
warn()    { echo -e "${YELLOW}  ⚠ $*${RESET}"; }
section() { echo -e "\n${BOLD}${MAGENTA}━━━ $* ━━━${RESET}"; }
pause()   { echo -e "\n${YELLOW}  [Press Enter to continue...]${RESET}"; read -r _pause_dummy || true; }
hr()      { echo -e "  ${BOLD}──────────────────────────────────────────${RESET}"; }

# ── Guard: env must exist ─────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${RED}[ERROR]${RESET} Virtual environment not found. Run ./start.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

clear 2>/dev/null || true

# ── Title slide ───────────────────────────────────────────────────────────────
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

# ── Section 3: Live inference with interactive dataset selection ───────────────
section "3 / 4  Live Inference — Choose from Dataset"
echo ""

# Python handles all three selection steps.
# All UI output goes to stderr (not buffered, not captured by $()).
# Only the final image path is printed to stdout for bash to capture.
CHOSEN_IMAGE=$(python3 - "$DATASET_DIR" <<'PYEOF'
import sys, os, re

DATASET_DIR = sys.argv[1]
CLASSES = ["bacterial", "fungal", "healthy", "mould"]
ui = sys.stderr  # all display output here — never buffered by bash $()

def p(*args, **kwargs):
    print(*args, **kwargs, file=ui)

def prompt(msg):
    ui.write(msg)
    ui.flush()
    with open("/dev/tty") as tty:
        return tty.readline().strip()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def count_images(directory):
    return sum(1 for f in os.listdir(directory) if os.path.splitext(f)[1] in IMG_EXTS)

def get_prefix(filename):
    return re.sub(r'_\d+\.[^.]+$', '', filename)

# ── Step 1: Pathogen class ────────────────────────────────────────────────────
p("  \033[1mStep 1: Choose a pathogen class\033[0m")
class_options = []
for cls in CLASSES:
    cls_dir = os.path.join(DATASET_DIR, cls)
    n = count_images(cls_dir) if os.path.isdir(cls_dir) else 0
    class_options.append((cls, n))

for i, (cls, n) in enumerate(class_options, 1):
    p(f"    {i})  {cls}  ({n} images)")

p()
while True:
    raw = prompt("  Enter class number (1-4): ")
    if raw.isdigit() and 1 <= int(raw) <= 4:
        chosen_class, _ = class_options[int(raw) - 1]
        p(f"  \033[0;32m  ✔ Selected class: \033[1m{chosen_class}\033[0m")
        break
    p("  \033[1;33m  ⚠ Please enter a number between 1 and 4.\033[0m")

# ── Step 2: Disease / plant ───────────────────────────────────────────────────
p("\n  \033[1m──────────────────────────────────────────\033[0m")
p(f"\n  \033[1mStep 2: Choose a disease / plant\033[0m\n")

cls_dir = os.path.join(DATASET_DIR, chosen_class)
all_files = sorted(
    f for f in os.listdir(cls_dir)
    if os.path.splitext(f)[1] in IMG_EXTS
)

seen = {}
for f in all_files:
    pfx = get_prefix(f)
    seen.setdefault(pfx, 0)
    seen[pfx] += 1

prefixes = sorted(seen.keys())
for i, pfx in enumerate(prefixes, 1):
    p(f"    {i})  {pfx.replace('_', ' ')}  ({seen[pfx]} images)")

p()
while True:
    raw = prompt(f"  Enter disease number (1-{len(prefixes)}): ")
    if raw.isdigit() and 1 <= int(raw) <= len(prefixes):
        chosen_prefix = prefixes[int(raw) - 1]
        p(f"  \033[0;32m  ✔ Selected: \033[1m{chosen_prefix.replace('_', ' ')}\033[0m")
        break
    p(f"  \033[1;33m  ⚠ Please enter a number between 1 and {len(prefixes)}.\033[0m")

# ── Step 3: Image number ──────────────────────────────────────────────────────
p("\n  \033[1m──────────────────────────────────────────\033[0m")
p(f"\n  \033[1mStep 3: Choose an image number\033[0m\n")

disease_files = sorted(f for f in all_files if get_prefix(f) == chosen_prefix)
total = len(disease_files)

p(f"  Available images: \033[1m1 – {total}\033[0m\n")
p("  Examples:")
for j in range(min(3, total)):
    p(f"    {j+1} →  {disease_files[j]}")
if total > 3:
    p("    ...")
    p(f"    {total} →  {disease_files[-1]}")

p()
while True:
    raw = prompt(f"  Enter image number (1-{total}): ")
    if raw.isdigit() and 1 <= int(raw) <= total:
        chosen_file = disease_files[int(raw) - 1]
        p(f"  \033[0;32m  ✔ Selected: \033[1m{chosen_file}\033[0m")
        break
    p(f"  \033[1;33m  ⚠ Please enter a number between 1 and {total}.\033[0m")

# Only the path goes to stdout — captured cleanly by bash $()
print(os.path.join(cls_dir, chosen_file), end="")
PYEOF
)

# ── Run prediction ────────────────────────────────────────────────────────────
echo ""
hr
echo ""
echo -e "  ${BOLD}Running ensemble prediction...${RESET}"
echo -e "  Image : ${CYAN}$(basename "$CHOSEN_IMAGE")${RESET}"
echo ""

mkdir -p "$PRED_DIR"
(cd "$SRC_DIR" && python3 predict.py "$CHOSEN_IMAGE")

# Show where output was saved and open it
LATEST_PRED=$(ls -td "$PRED_DIR"/*/ 2>/dev/null | head -1)
if [[ -n "$LATEST_PRED" ]]; then
    echo ""
    ok "Grad-CAM + report saved to:  $LATEST_PRED"
    info "Opening prediction output ..."
    open "$LATEST_PRED" 2>/dev/null || true
else
    warn "predictions/ folder appears empty — check if predict.py completed."
fi

pause

# ── Section 4: (Optional) Full ensemble evaluation ────────────────────────────
section "4 / 4  Full Ensemble Evaluation (optional)"
echo ""
echo -n "  Run full ensemble evaluation on the test set? (y/N): "
read -r RUN_ENSEMBLE || RUN_ENSEMBLE="N"

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
