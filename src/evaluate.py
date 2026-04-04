"""
Run after training to get full metrics + Grad-CAM visualizations.
Usage: python evaluate.py
"""
import os
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from dataset import get_dataloaders, CLASSES
from model import build_model

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, 'dataset', 'flat')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR       = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')

IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])


# ── Grad-CAM ───────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _, __, output):
        self._activations = output.detach()

    def _bwd_hook(self, _, __, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(img_tensor)
        output[0, class_idx].backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1)).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Plotting helpers ────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(log_path, save_path):
    rows = []
    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    p1 = [r for r in rows if r['phase'] == 'phase1']
    p2 = [r for r in rows if r['phase'] == 'phase2']
    offset = len(p1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ['train_acc', 'train_loss'],
                                  ['Accuracy', 'Loss']):
        val_metric = metric.replace('train', 'val')
        if p1:
            eps = [int(r['epoch']) for r in p1]
            ax.plot(eps, [float(r[metric])     for r in p1], 'b--', label='P1 train')
            ax.plot(eps, [float(r[val_metric]) for r in p1], 'b-',  label='P1 val')
        if p2:
            eps = [int(r['epoch']) + offset for r in p2]
            ax.plot(eps, [float(r[metric])     for r in p2], 'r--', label='P2 train')
            ax.plot(eps, [float(r[val_metric]) for r in p2], 'r-',  label='P2 val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Curves', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_gradcam(model, loader, device, save_path, n_samples=8):
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    imgs_batch, labels_batch = next(iter(loader))
    n = min(n_samples, len(imgs_batch))

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        img_tensor = imgs_batch[i:i+1].to(device).requires_grad_(True)
        true_label = labels_batch[i].item()

        with torch.enable_grad():
            pred_label = model(img_tensor).argmax(1).item()
            cam = gradcam.generate(img_tensor, pred_label)

        img_np = imgs_batch[i].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * IMG_STD + IMG_MEAN, 0, 1)

        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = mpl_cm.jet(cam_resized)[:, :, :3]
        overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"True:\n{CLASSES[true_label]}", fontsize=8)
        axes[0, i].axis('off')

        color = 'green' if pred_label == true_label else 'red'
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Pred:\n{CLASSES[pred_label]}", fontsize=8, color=color)
        axes[1, i].axis('off')

    plt.suptitle('Grad-CAM  (green = correct, red = wrong)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def run_evaluation():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))

    _, _, test_loader = get_dataloaders(DATA_DIR, batch_size=32, num_workers=4)

    model = build_model(pretrained=False).to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}\n")

    # Collect predictions
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out  = model(imgs.to(device))
            prob = torch.softmax(out, dim=1).cpu().numpy()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(prob)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Classification report
    print("--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    # AUC-ROC
    y_bin = label_binarize(all_labels, classes=list(range(len(CLASSES))))
    auc   = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='macro')
    print(f"Macro AUC-ROC: {auc:.4f}\n")

    # Save report to txt
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
        f.write(f"\nMacro AUC-ROC: {auc:.4f}\n")
    print(f"Saved: {report_path}")

    # Plots
    plot_confusion_matrix(all_labels, all_preds,
                          os.path.join(RESULTS_DIR, 'confusion_matrix.png'))

    log_path = os.path.join(LOG_DIR, 'training_log.csv')
    if os.path.exists(log_path):
        plot_training_curves(log_path,
                             os.path.join(RESULTS_DIR, 'training_curves.png'))

    plot_gradcam(model, test_loader, device,
                 os.path.join(RESULTS_DIR, 'gradcam.png'))


if __name__ == '__main__':
    run_evaluation()
