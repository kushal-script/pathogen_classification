"""
Ensemble inference via probability averaging across all 5 trained models.

Usage:
    # Evaluate on test set
    python ensemble.py

    # Classify a single image
    python ensemble.py --image /path/to/leaf.jpg
"""
import os
import sys
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from dataset import get_dataloaders, CLASSES, VAL_TRANSFORMS, make_transforms
from model import build_model, ARCHS, gradcam_target_layer, ARCH_INPUT_SIZE

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, 'dataset', 'flat')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR    = os.path.join(BASE_DIR, 'results')

IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])


# ── Model loading ───────────────────────────────────────────────────────────

def load_all_models(device):
    """Load all available trained models. Returns list of (arch, model)."""
    models = []
    for arch in ARCHS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{arch}.pt')
        if not os.path.exists(ckpt_path):
            print(f"  [MISSING] {arch} — no checkpoint at {ckpt_path}, skipping.")
            continue
        model = build_model(arch, pretrained=False).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"  [OK] {arch:<22} epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")
        models.append((arch, model))
    return models


# ── Core ensemble prediction ────────────────────────────────────────────────

@torch.no_grad()
def ensemble_probs(models, img_tensor_per_arch):
    """
    Average softmax probabilities from all models.
    img_tensor_per_arch: dict of arch -> (1, C, H, W) tensor on device
    Returns: numpy array shape (num_classes,)
    """
    prob_sum = None
    for arch, model in models:
        tensor = img_tensor_per_arch[arch]
        out    = model(tensor)
        prob   = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
        prob_sum = prob if prob_sum is None else prob_sum + prob
    return prob_sum / len(models)


# ── Grad-CAM (ensemble: average CAMs from all models) ──────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._act  = None
        self._grad = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, '_act', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, '_grad', go[0].detach()))

    def generate(self, img_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(img_tensor)
        out[0, class_idx].backward()
        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._act).sum(dim=1)).squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def ensemble_gradcam(models, img_tensors_per_arch, class_idx, arch_list):
    """Average Grad-CAM heatmaps across models (skips VGG16 — inplace ReLU hook conflict)."""
    cam_sum, n_used = None, 0
    target_h, target_w = None, None
    for arch, model in models:
        if arch == 'vgg16':          # VGG16 inplace ReLU breaks backward hooks
            continue
        layer   = gradcam_target_layer(model, arch)
        gradcam = GradCAM(model, layer)
        img_t   = img_tensors_per_arch[arch].detach().requires_grad_(True)
        with torch.enable_grad():
            cam = gradcam.generate(img_t, class_idx)
        if target_h is None:
            target_h = img_tensors_per_arch[arch].shape[2]
            target_w = img_tensors_per_arch[arch].shape[3]
        cam_r   = cv2.resize(cam, (target_w, target_h))
        cam_sum = cam_r if cam_sum is None else cam_sum + cam_r
        n_used += 1
    avg_cam = cam_sum / n_used
    return (avg_cam - avg_cam.min()) / (avg_cam.max() - avg_cam.min() + 1e-8)


# ── Evaluation on full test set ─────────────────────────────────────────────

def run_evaluation(models, device):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build a separate test loader per arch (different input sizes)
    arch_loaders = {}
    for arch, _ in models:
        size = ARCH_INPUT_SIZE.get(arch, 300)
        _, _, loader = get_dataloaders(DATA_DIR, batch_size=32, num_workers=0,
                                       input_size=size)
        arch_loaders[arch] = loader

    # Use the first arch's loader as reference for labels (same seed → same split)
    ref_loader = arch_loaders[next(iter(arch_loaders))]

    all_preds, all_labels, all_probs = [], [], []
    per_model_correct = {arch: 0 for arch, _ in models}
    total = 0

    # Zip all loaders together (same stratified split → same label order)
    for batches in zip(*[arch_loaders[arch] for arch, _ in models]):
        labels = batches[0][1]   # labels are identical across loaders
        batch_probs = np.zeros((labels.size(0), len(CLASSES)))

        for i, (arch, model) in enumerate(models):
            imgs = batches[i][0].to(device)
            with torch.no_grad():
                out   = model(imgs)
                probs = torch.softmax(out, dim=1).cpu().numpy()
            batch_probs += probs
            per_model_correct[arch] += (probs.argmax(axis=1) == labels.numpy()).sum()

        batch_probs /= len(models)
        all_probs.extend(batch_probs)
        all_preds.extend(batch_probs.argmax(axis=1))
        all_labels.extend(labels.numpy())
        total += labels.size(0)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ── Print results ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Individual Model Accuracies")
    print(f"{'='*60}")
    for arch, _ in models:
        acc = per_model_correct[arch] / total
        print(f"  {arch:<22} {acc:.4f}  ({acc*100:.2f}%)")

    ensemble_acc = (all_preds == all_labels).mean()
    print(f"\n  {'ENSEMBLE':<22} {ensemble_acc:.4f}  ({ensemble_acc*100:.2f}%)")

    print(f"\n{'='*60}")
    print("  Classification Report — Ensemble")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    y_bin = label_binarize(all_labels, classes=list(range(len(CLASSES))))
    auc   = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='macro')
    print(f"  Macro AUC-ROC : {auc:.4f}")

    # ── Save report ────────────────────────────────────────────────────────
    report_path = os.path.join(RESULTS_DIR, 'ensemble_report.txt')
    with open(report_path, 'w') as f:
        f.write("Individual Model Accuracies\n")
        f.write("-"*40 + "\n")
        for arch, _ in models:
            acc = per_model_correct[arch] / total
            f.write(f"{arch:<22} {acc:.4f}\n")
        f.write(f"\n{'ENSEMBLE':<22} {ensemble_acc:.4f}\n\n")
        f.write("Classification Report — Ensemble\n")
        f.write("-"*40 + "\n")
        f.write(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
        f.write(f"\nMacro AUC-ROC: {auc:.4f}\n")
    print(f"\nReport saved: {report_path}")

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Ensemble Confusion Matrix  (acc={ensemble_acc:.3f})')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'ensemble_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"Saved: {cm_path}")

    # ── Model comparison bar chart ─────────────────────────────────────────
    arch_names  = [a for a, _ in models] + ['ENSEMBLE']
    arch_accs   = [per_model_correct[a] / total for a, _ in models] + [ensemble_acc]
    colors      = ['steelblue'] * len(models) + ['darkorange']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(arch_names, [a * 100 for a in arch_accs], color=colors, edgecolor='white')
    ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=10)
    ax.set_ylim(70, 100)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Individual Models vs Ensemble — Test Accuracy')
    ax.axhline(ensemble_acc * 100, color='darkorange', linestyle='--', linewidth=1.2, alpha=0.6)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
    plt.savefig(bar_path, dpi=150); plt.close()
    print(f"Saved: {bar_path}")

    # ── Ensemble Grad-CAM on 8 test samples ───────────────────────────────
    # Use reference arch (first model) for display images
    ref_arch   = models[0][0]
    ref_size   = ARCH_INPUT_SIZE.get(ref_arch, 300)
    _, _, test_loader_small = get_dataloaders(DATA_DIR, batch_size=8, num_workers=0,
                                              input_size=ref_size)
    imgs_batch, labels_batch = next(iter(test_loader_small))
    n = min(8, len(imgs_batch))

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        true_label = labels_batch[i].item()
        # Build per-arch tensors for this single image
        img_tensors = {}
        for arch, _ in models:
            size = ARCH_INPUT_SIZE.get(arch, 300)
            _, val_tf = make_transforms(size)
            # Re-open from reference batch — denormalize then re-transform
            raw = imgs_batch[i].permute(1,2,0).numpy()
            raw = np.clip(raw * IMG_STD + IMG_MEAN, 0, 1)
            pil = Image.fromarray((raw * 255).astype(np.uint8))
            img_tensors[arch] = val_tf(pil).unsqueeze(0).to(device)
        avg_probs  = ensemble_probs(models, img_tensors)
        pred_label = int(avg_probs.argmax())

        cam = ensemble_gradcam(models, img_tensors, pred_label, [a for a, _ in models])
        img_np = imgs_batch[i].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * IMG_STD + IMG_MEAN, 0, 1)
        cam_r  = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        overlay = np.clip(0.5 * img_np + 0.5 * mpl_cm.jet(cam_r)[:, :, :3], 0, 1)

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"True:\n{CLASSES[true_label]}", fontsize=8)
        axes[0, i].axis('off')

        color = 'green' if pred_label == true_label else 'red'
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Pred:\n{CLASSES[pred_label]}\n({avg_probs[pred_label]:.2f})",
                              fontsize=7, color=color)
        axes[1, i].axis('off')

    plt.suptitle('Ensemble Grad-CAM  (green=correct, red=wrong)', fontsize=11)
    plt.tight_layout()
    gcam_path = os.path.join(RESULTS_DIR, 'ensemble_gradcam.png')
    plt.savefig(gcam_path, dpi=150); plt.close()
    print(f"Saved: {gcam_path}")


# ── Single image inference ──────────────────────────────────────────────────

def predict_image(image_path, models, device):
    img = Image.open(image_path).convert('RGB')
    img_tensors = {}
    for arch, _ in models:
        size = ARCH_INPUT_SIZE.get(arch, 300)
        _, val_tf = make_transforms(size)
        img_tensors[arch] = val_tf(img).unsqueeze(0).to(device)

    avg_probs = ensemble_probs(models, img_tensors)
    pred_idx  = int(avg_probs.argmax())

    print(f"\nImage    : {os.path.basename(image_path)}")
    print(f"Predicted: {CLASSES[pred_idx]}  ({avg_probs[pred_idx]:.2%} confidence)\n")
    print("Class probabilities (ensemble average):")
    for cls, p in sorted(zip(CLASSES, avg_probs), key=lambda x: -x[1]):
        bar = '#' * int(p * 40)
        print(f"  {cls:<14} {p:.4f}  {bar}")
    return {'predicted_class': CLASSES[pred_idx],
            'confidence':      float(avg_probs[pred_idx]),
            'probabilities':   dict(zip(CLASSES, avg_probs.tolist()))}


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble inference & evaluation')
    parser.add_argument('--image', default=None,
                        help='Path to single image (omit for full test-set evaluation)')
    args = parser.parse_args()

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"Device: {device}\n")
    print("Loading models...")
    models = load_all_models(device)
    if not models:
        print("No trained models found. Run train_all.py first.")
        sys.exit(1)
    print(f"\n{len(models)}/{len(ARCHS)} models loaded.")

    if args.image:
        predict_image(args.image, models, device)
    else:
        run_evaluation(models, device)
