import os
import csv
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from model import build_model, freeze_backbone, unfreeze_all, get_param_groups, ARCHS, ARCH_INPUT_SIZE

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, 'dataset', 'flat')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR        = os.path.join(BASE_DIR, 'logs')


def get_device():
    if torch.backends.mps.is_available():  return torch.device('mps')
    if torch.cuda.is_available():          return torch.device('cuda')
    return torch.device('cpu')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total


def run_training(args):
    arch = args.arch
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{arch}.pt')
    if os.path.exists(ckpt_path) and not args.force:
        print(f"[{arch}] Checkpoint already exists — skipping. Use --force to retrain.")
        return

    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Training : {arch.upper()}")
    print(f"  Device   : {device}")
    print(f"{'='*60}")

    input_size = ARCH_INPUT_SIZE.get(arch, 300)
    print(f"  Input size : {input_size}×{input_size}")
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_DIR, batch_size=args.batch_size, num_workers=args.num_workers,
        input_size=input_size
    )

    model     = build_model(arch, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    log_path = os.path.join(LOG_DIR, f'training_log_{arch}.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['phase', 'epoch', 'train_loss', 'train_acc',
                                 'val_loss', 'val_acc', 'lr'])

    best_val_acc = 0.0

    # ── Phase 1: classifier head only ──────────────────────────────────────
    print(f"\n--- Phase 1: head only ({args.phase1_epochs} epochs) ---")
    freeze_backbone(model, arch)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = optim.Adam(head_params, lr=1e-3)
    scheduler   = CosineAnnealingLR(optimizer, T_max=args.phase1_epochs, eta_min=1e-5)

    for epoch in range(1, args.phase1_epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"[P1 {epoch:02d}/{args.phase1_epochs}] "
              f"loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} val_acc={va:.4f} | "
              f"lr={lr:.2e}  ({time.time()-t0:.1f}s)", flush=True)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(['phase1', epoch, tl, ta, vl, va, lr])
        if va > best_val_acc:
            best_val_acc = va
            torch.save({'arch': arch, 'epoch': epoch,
                        'model_state': model.state_dict(), 'val_acc': va}, ckpt_path)

    # ── Phase 2: fine-tune all layers ──────────────────────────────────────
    print(f"\n--- Phase 2: full fine-tune ({args.phase2_epochs} epochs) ---")
    unfreeze_all(model)
    optimizer = optim.Adam(
        get_param_groups(model, arch, args.finetune_lr, args.finetune_lr * 10)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs, eta_min=1e-6)

    for epoch in range(1, args.phase2_epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"[P2 {epoch:02d}/{args.phase2_epochs}] "
              f"loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} val_acc={va:.4f} | "
              f"lr={lr:.2e}  ({time.time()-t0:.1f}s)", flush=True)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(['phase2', epoch, tl, ta, vl, va, lr])
        if va > best_val_acc:
            best_val_acc = va
            torch.save({'arch': arch, 'epoch': epoch,
                        'model_state': model.state_dict(), 'val_acc': va}, ckpt_path)

    # ── Test evaluation ─────────────────────────────────────────────────────
    print(f"\n--- Test Evaluation [{arch}] ---")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    tl, ta = evaluate(model, test_loader, criterion, device)
    print(f"Best epoch {ckpt['epoch']} | test_loss={tl:.4f}  test_acc={ta:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',           default='efficientnet_b3', choices=ARCHS)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--num_workers',    type=int,   default=0)
    parser.add_argument('--phase1_epochs',  type=int,   default=10)
    parser.add_argument('--phase2_epochs',  type=int,   default=30)
    parser.add_argument('--finetune_lr',    type=float, default=1e-4)
    parser.add_argument('--force',          action='store_true',
                        help='Retrain even if checkpoint exists')
    args = parser.parse_args()
    run_training(args)
