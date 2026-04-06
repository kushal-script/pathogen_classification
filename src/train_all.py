"""
Train all 5 architectures sequentially.
Already-trained models (checkpoint exists) are automatically skipped.

Usage:
    python train_all.py
    python train_all.py --force          # retrain everything
    python train_all.py --skip           # skip specific arch
"""
import os
import sys
import argparse
import subprocess
import time

from model import ARCHS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(args):
    skip = set(args.skip or [])
    results = {}

    for arch in ARCHS:
        if arch in skip:
            print(f"\n[SKIP] {arch}")
            continue

        ckpt = os.path.join(BASE_DIR, 'checkpoints', f'best_{arch}.pt')
        if os.path.exists(ckpt) and not args.force:
            print(f"\n[SKIP] {arch} — checkpoint exists ({ckpt})")
            results[arch] = 'skipped (checkpoint exists)'
            continue

        print(f"\n{'#'*60}")
        print(f"# Starting: {arch}")
        print(f"{'#'*60}")
        t0 = time.time()

        cmd = [
            sys.executable, '-u', 'train.py',
            '--arch',          arch,
            '--batch_size',    str(args.batch_size),
            '--num_workers',   str(args.num_workers),
            '--phase1_epochs', str(args.phase1_epochs),
            '--phase2_epochs', str(args.phase2_epochs),
            '--finetune_lr',   str(args.finetune_lr),
        ]
        if args.force:
            cmd.append('--force')

        ret = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        elapsed = (time.time() - t0) / 60
        status = 'OK' if ret.returncode == 0 else f'FAILED (exit {ret.returncode})'
        results[arch] = f"{status}  ({elapsed:.1f} min)"
        print(f"\n[{arch}] {status} in {elapsed:.1f} min")

    print(f"\n{'='*60}")
    print("  Training Summary")
    print(f"{'='*60}")
    for arch, status in results.items():
        print(f"  {arch:<22} {status}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--num_workers',    type=int,   default=0)
    parser.add_argument('--phase1_epochs',  type=int,   default=10)
    parser.add_argument('--phase2_epochs',  type=int,   default=30)
    parser.add_argument('--finetune_lr',    type=float, default=1e-4)
    parser.add_argument('--force',          action='store_true')
    parser.add_argument('--skip',           nargs='+',  default=[],
                        help='Arch names to skip')
    args = parser.parse_args()
    main(args)
