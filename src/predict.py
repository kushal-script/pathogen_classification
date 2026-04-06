"""
Single-image ensemble inference with full model breakdown and Grad-CAM.

Usage: python predict.py <image_path>

Displays:
  - Per-model softmax predictions (all 5 CNNs)
  - Ensemble probability-averaged prediction
  - Predicted vs actual class (if image is from dataset)
  - Grad-CAM heatmaps for all 5 models + ensemble average
"""
import os
import sys
import shutil
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import torch
from PIL import Image

from dataset import CLASSES, make_transforms
from model import build_model, ARCHS, ARCH_INPUT_SIZE, gradcam_target_layer, disable_inplace_relu

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR    = os.path.join(BASE_DIR, 'results')

IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])


# ── Grad-CAM ──────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._act  = None
        self._grad = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_act', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, '_grad', go[0].detach()))

    def generate(self, img_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(img_tensor)
        out[0, class_idx].backward()
        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._act).sum(dim=1)).squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


# ── Model loading ─────────────────────────────────────────────────────────

def load_all_models(device):
    models = []
    for arch in ARCHS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{arch}.pt')
        if not os.path.exists(ckpt_path):
            continue
        model = build_model(arch, pretrained=False).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        models.append((arch, model))
    return models


# ── Infer actual class from file path ─────────────────────────────────────

def infer_actual_class(image_path):
    """Try to determine actual class from the directory name."""
    parts = os.path.normpath(image_path).split(os.sep)
    for cls in CLASSES:
        if cls in parts:
            return cls
    return None


# ── Main prediction ───────────────────────────────────────────────────────

def predict(image_path, models, device):
    pil_img = Image.open(image_path).convert('RGB')
    display_size = 300
    img_display = np.array(pil_img.resize((display_size, display_size)))

    # ── Per-model predictions ─────────────────────────────────────────
    all_probs = {}
    all_tensors = {}
    for arch, model in models:
        size = ARCH_INPUT_SIZE.get(arch, 300)
        _, val_tf = make_transforms(size)
        tensor = val_tf(pil_img).unsqueeze(0).to(device)
        all_tensors[arch] = tensor
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        all_probs[arch] = probs

    # ── Ensemble averaging ────────────────────────────────────────────
    prob_stack = np.stack(list(all_probs.values()))
    ensemble_probs = prob_stack.mean(axis=0)
    ensemble_pred = int(ensemble_probs.argmax())
    actual_class = infer_actual_class(image_path)

    # ── Terminal output ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"{'='*65}")

    # Per-model breakdown
    print(f"\n  {'Model':<24} {'Prediction':<12} {'Confidence':>10}")
    print(f"  {'-'*24} {'-'*12} {'-'*10}")
    for arch, _ in models:
        probs = all_probs[arch]
        pred_idx = int(probs.argmax())
        pred_cls = CLASSES[pred_idx]
        conf = probs[pred_idx]
        marker = "  *" if pred_cls != CLASSES[ensemble_pred] else ""
        print(f"  {arch:<24} {pred_cls:<12} {conf:>9.2%}{marker}")

    # Ensemble result
    print(f"\n  {'ENSEMBLE (avg)':<24} {CLASSES[ensemble_pred]:<12} {ensemble_probs[ensemble_pred]:>9.2%}")

    # Actual class
    if actual_class:
        match = "CORRECT" if actual_class == CLASSES[ensemble_pred] else "WRONG"
        print(f"\n  Actual class: {actual_class}  [{match}]")

    # Full probability table
    print(f"\n  Ensemble class probabilities:")
    for cls, p in sorted(zip(CLASSES, ensemble_probs), key=lambda x: -x[1]):
        print(f"    {cls:<12} {p:.4f}")

    print(f"\n{'='*65}")

    # ── Grad-CAM for all models + ensemble ────────────────────────────
    print("\n  Generating Grad-CAM heatmaps...", flush=True)

    grad_cams = {}
    ensemble_cam_sum = None
    ensemble_cam_count = 0

    for arch, model in models:
        disable_inplace_relu(model)
        layer = gradcam_target_layer(model, arch)
        gcam = GradCAM(model, layer)
        tensor = all_tensors[arch].detach().requires_grad_(True)
        with torch.enable_grad():
            cam = gcam.generate(tensor, ensemble_pred)
        cam_resized = cv2.resize(cam, (display_size, display_size))
        grad_cams[arch] = cam_resized

        if ensemble_cam_sum is None:
            ensemble_cam_sum = cam_resized.copy()
        else:
            ensemble_cam_sum += cam_resized
        ensemble_cam_count += 1

    # Ensemble Grad-CAM (average of non-VGG models)
    ensemble_cam = ensemble_cam_sum / ensemble_cam_count
    ensemble_cam = (ensemble_cam - ensemble_cam.min()) / (ensemble_cam.max() - ensemble_cam.min() + 1e-8)

    # ── Build visualisation figure ────────────────────────────────────
    # Layout: Row 1 = Original + 5 model Grad-CAMs + Ensemble Grad-CAM = 7 columns
    n_cols = 2 + len(models)  # original + models + ensemble
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

    img_float = img_display.astype(np.float64) / 255.0

    # ── Row 0: Original image + per-model probability bars ────────────
    # Original
    axes[0, 0].imshow(img_display)
    title_str = f"Input Image"
    if actual_class:
        title_str += f"\nActual: {actual_class}"
    axes[0, 0].set_title(title_str, fontsize=9, fontweight='bold')
    axes[0, 0].axis('off')

    # Per-model probability bars
    for idx, (arch, _) in enumerate(models):
        ax = axes[0, idx + 1]
        probs = all_probs[arch]
        pred_idx = int(probs.argmax())
        colors = ['#4CAF50' if i == pred_idx else '#90CAF9' for i in range(len(CLASSES))]
        bars = ax.barh(CLASSES, probs, color=colors, edgecolor='white', height=0.6)
        ax.set_xlim(0, 1)
        ax.set_title(f"{arch}\n→ {CLASSES[pred_idx]} ({probs[pred_idx]:.1%})",
                     fontsize=8, fontweight='bold')
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=6)
        for bar, p in zip(bars, probs):
            if p > 0.05:
                ax.text(p - 0.02, bar.get_y() + bar.get_height()/2,
                       f'{p:.1%}', ha='right', va='center', fontsize=6, color='white')

    # Ensemble probability bars
    ax_ens = axes[0, -1]
    colors_ens = ['#FF9800' if i == ensemble_pred else '#FFE0B2' for i in range(len(CLASSES))]
    bars_ens = ax_ens.barh(CLASSES, ensemble_probs, color=colors_ens, edgecolor='white', height=0.6)
    ax_ens.set_xlim(0, 1)
    ax_ens.set_title(f"ENSEMBLE\n→ {CLASSES[ensemble_pred]} ({ensemble_probs[ensemble_pred]:.1%})",
                     fontsize=9, fontweight='bold', color='#E65100')
    ax_ens.tick_params(axis='y', labelsize=7)
    ax_ens.tick_params(axis='x', labelsize=6)
    for bar, p in zip(bars_ens, ensemble_probs):
        if p > 0.05:
            ax_ens.text(p - 0.02, bar.get_y() + bar.get_height()/2,
                       f'{p:.1%}', ha='right', va='center', fontsize=6, color='white')

    # ── Row 1: Original + Grad-CAM heatmaps ──────────────────────────
    axes[1, 0].imshow(img_display)
    axes[1, 0].set_title("Original", fontsize=9)
    axes[1, 0].axis('off')

    for idx, (arch, _) in enumerate(models):
        ax = axes[1, idx + 1]
        cam = grad_cams[arch]
        heatmap = mpl_cm.jet(cam)[:, :, :3]
        overlay = np.clip(0.5 * img_float + 0.5 * heatmap, 0, 1)
        ax.imshow(overlay)
        ax.set_title(f"Grad-CAM\n{arch}", fontsize=8)
        ax.axis('off')

    # Ensemble Grad-CAM
    ax_ecam = axes[1, -1]
    heatmap_ens = mpl_cm.jet(ensemble_cam)[:, :, :3]
    overlay_ens = np.clip(0.5 * img_float + 0.5 * heatmap_ens, 0, 1)
    ax_ecam.imshow(overlay_ens)
    ax_ecam.set_title("Grad-CAM\nENSEMBLE (avg)", fontsize=9, fontweight='bold', color='#E65100')
    ax_ecam.axis('off')

    # ── Final layout ──────────────────────────────────────────────────
    match_str = ""
    if actual_class:
        if actual_class == CLASSES[ensemble_pred]:
            match_str = f"  |  Actual: {actual_class} [CORRECT]"
        else:
            match_str = f"  |  Actual: {actual_class} [WRONG]"

    fig.suptitle(
        f"Ensemble Prediction: {CLASSES[ensemble_pred]} ({ensemble_probs[ensemble_pred]:.1%}){match_str}",
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    # ── Save to predictions/<datetime>/ folder ─────────────────────────
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pred_dir = os.path.join(BASE_DIR, 'predictions', timestamp)
    os.makedirs(pred_dir, exist_ok=True)

    # Save visualisation
    vis_path = os.path.join(pred_dir, 'prediction.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Copy original input image
    orig_ext = os.path.splitext(image_path)[1]
    shutil.copy2(image_path, os.path.join(pred_dir, f'input{orig_ext}'))

    # Save prediction summary as text
    summary_path = os.path.join(pred_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write(f"Timestamp: {timestamp}\n")
        if actual_class:
            f.write(f"Actual class: {actual_class}\n")
        f.write(f"\nPer-model predictions:\n")
        f.write(f"{'Model':<24} {'Prediction':<12} {'Confidence':>10}\n")
        f.write(f"{'-'*48}\n")
        for arch, _ in models:
            probs = all_probs[arch]
            pred_idx = int(probs.argmax())
            f.write(f"{arch:<24} {CLASSES[pred_idx]:<12} {probs[pred_idx]:>9.2%}\n")
        f.write(f"\n{'ENSEMBLE (avg)':<24} {CLASSES[ensemble_pred]:<12} {ensemble_probs[ensemble_pred]:>9.2%}\n")
        if actual_class:
            match = "CORRECT" if actual_class == CLASSES[ensemble_pred] else "WRONG"
            f.write(f"\nResult: {match}\n")
        f.write(f"\nEnsemble class probabilities:\n")
        for cls, p in sorted(zip(CLASSES, ensemble_probs), key=lambda x: -x[1]):
            f.write(f"  {cls:<12} {p:.4f}\n")

    print(f"\n  Prediction saved to: {pred_dir}/")
    print(f"    prediction.png  — visualisation (model bars + Grad-CAM)")
    print(f"    input{orig_ext:<11} — original image")
    print(f"    summary.txt     — prediction details")
    print(f"\n  Open:  open \"{pred_dir}\"\n")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"Device: {device}")
    print("Loading models...")
    models = load_all_models(device)
    if not models:
        print("No trained models found. Run train_all.py first.")
        sys.exit(1)
    print(f"{len(models)}/{len(ARCHS)} models loaded.")

    predict(image_path, models, device)
