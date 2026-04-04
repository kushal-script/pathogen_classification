"""
Generate publication-quality figures for the pathogen classification research paper.

Outputs:
    results/dataset_samples.png      -- 4x4 grid of sample images (4 classes x 4 crops)
    results/architecture_diagram.png -- Ensemble pipeline flowchart
    results/class_distribution.png   -- Bar chart of image counts per class
"""

import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "flat")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Dataset Sample Grid  (4 classes x 4 crops)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_dataset_samples():
    classes = ["bacterial", "fungal", "healthy", "mould"]
    class_labels = ["Bacterial", "Fungal", "Healthy", "Mould"]

    # For each class, pick one image from each of 4 different crops.
    # The crop prefixes vary per class, so we define them explicitly.
    crop_prefixes = {
        "bacterial": ["Cabbage_Black_rot", "Cauliflower_Bacterial",
                      "Spinach", "Lettuce_Bacterial"],
        "fungal":    ["Cabbage_Alternaria", "Cauliflower_Alternaria",
                      "Spinach_Anthracnose", "Septoria_Blight_on_lettuce"],
        "healthy":   ["Cabbage_Healthy", "Cauliflower_Healthy",
                      "Spinach_Healthy", "Lettuce_Healthy"],
        "mould":     ["Cabbage_Downy", "Cauliflower_Downy",
                      "Spinach_Downy", "Lettuce_Downy"],
    }

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=200)
    fig.patch.set_facecolor("white")

    for col_idx, cls in enumerate(classes):
        cls_dir = os.path.join(DATASET_DIR, cls)
        prefixes = crop_prefixes[cls]
        for row_idx, prefix in enumerate(prefixes):
            ax = axes[row_idx, col_idx]
            # Find first image matching this prefix
            pattern = os.path.join(cls_dir, prefix + "_0005.*")
            matches = glob.glob(pattern)
            if not matches:
                # fallback to 0001
                pattern = os.path.join(cls_dir, prefix + "_0001.*")
                matches = glob.glob(pattern)
            if not matches:
                # broader fallback: any file starting with prefix
                pattern = os.path.join(cls_dir, prefix + "*")
                matches = sorted(glob.glob(pattern))
            if matches:
                img = Image.open(matches[0]).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

        # Column title (bold class name)
        axes[0, col_idx].set_title(class_labels[col_idx],
                                   fontweight="bold", fontsize=14, pad=8)

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.93, bottom=0.02,
                        left=0.02, right=0.98)
    out_path = os.path.join(RESULTS_DIR, "dataset_samples.png")
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Architecture Diagram  (Ensemble pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Color palette
    c_input  = "#C8E6C9"   # light green
    c_cnn    = "#BBDEFB"   # light blue
    c_avg    = "#FFE0B2"   # orange
    c_output = "#FFCDD2"   # light red
    c_border = "#37474F"   # dark grey

    def draw_box(x, y, w, h, text, facecolor, fontsize=10, bold=False):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=facecolor,
            edgecolor=c_border,
            linewidth=1.5,
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight=weight, color="#212121")

    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->,head_width=0.15,head_length=0.12",
            color=c_border, linewidth=1.5,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    # ── Input box ─────────────────────────────────────────────────────────
    inp_x, inp_y, inp_w, inp_h = 0.3, 2.8, 2.0, 1.4
    draw_box(inp_x, inp_y, inp_w, inp_h, "Input Image\n(224 x 224 x 3)",
             c_input, fontsize=11, bold=True)

    # ── CNN branches ──────────────────────────────────────────────────────
    cnn_names = [
        "EfficientNet-B3",
        "ResNet-50",
        "VGG-16",
        "DenseNet-121",
        "MobileNet-V3-L",
    ]
    cnn_x = 3.8
    cnn_w, cnn_h = 2.6, 0.85
    gap = 0.35
    total_cnn_height = 5 * cnn_h + 4 * gap
    cnn_start_y = (7 - total_cnn_height) / 2

    cnn_centers = []
    for i, name in enumerate(cnn_names):
        cy = cnn_start_y + i * (cnn_h + gap)
        draw_box(cnn_x, cy, cnn_w, cnn_h,
                 f"{name}\nsoftmax(4)", c_cnn, fontsize=9.5, bold=False)
        cnn_centers.append((cnn_x, cy, cnn_w, cnn_h))

    # ── Arrows: input → CNNs ─────────────────────────────────────────────
    inp_right_x = inp_x + inp_w
    inp_center_y = inp_y + inp_h / 2
    for (cx, cy, cw, ch) in cnn_centers:
        draw_arrow(inp_right_x + 0.05, inp_center_y,
                   cx - 0.05, cy + ch / 2)

    # ── Averaging box ─────────────────────────────────────────────────────
    avg_x, avg_w, avg_h = 7.8, 2.6, 1.4
    avg_y = (7 - avg_h) / 2
    draw_box(avg_x, avg_y, avg_w, avg_h,
             "Probability\nAveraging", c_avg, fontsize=12, bold=True)

    # ── Arrows: CNNs → averaging ──────────────────────────────────────────
    for (cx, cy, cw, ch) in cnn_centers:
        draw_arrow(cx + cw + 0.05, cy + ch / 2,
                   avg_x - 0.05, avg_y + avg_h / 2)

    # ── Output box ────────────────────────────────────────────────────────
    out_x, out_w, out_h = 11.8, 2.0, 1.4
    out_y = (7 - out_h) / 2
    draw_box(out_x, out_y, out_w, out_h,
             "Predicted\nClass", c_output, fontsize=12, bold=True)

    # Arrow: averaging → output
    draw_arrow(avg_x + avg_w + 0.05, avg_y + avg_h / 2,
               out_x - 0.05, out_y + out_h / 2)

    out_path = os.path.join(RESULTS_DIR, "architecture_diagram.png")
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Class Distribution Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def generate_class_distribution():
    classes = ["Bacterial", "Fungal", "Healthy", "Mould"]
    counts  = [440, 423, 440, 439]
    colors  = ["#EF5350", "#42A5F5", "#66BB6A", "#FFA726"]  # red, blue, green, orange

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(classes, counts, color=colors, edgecolor="#37474F",
                  linewidth=1.2, width=0.6)

    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#212121")

    ax.set_ylabel("Number of Images", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", fontsize=13, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylim(0, max(counts) + 40)
    ax.tick_params(axis="both", labelsize=12)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating paper figures...")
    generate_dataset_samples()
    generate_architecture_diagram()
    generate_class_distribution()
    print("All figures generated successfully.")
