"""
Single-image inference.
Usage: python predict.py <image_path>
"""
import os
import sys

import torch
from PIL import Image

from dataset import CLASSES, VAL_TRANSFORMS
from model import build_model

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')


def predict(image_path: str, checkpoint_path: str = None) -> dict:
    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))

    ckpt_path = checkpoint_path or os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    model = build_model(pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    img    = Image.open(image_path).convert('RGB')
    tensor = VAL_TRANSFORMS(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    return {
        'predicted_class': CLASSES[pred_idx],
        'confidence':      float(probs[pred_idx]),
        'probabilities':   {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(f"\nPredicted : {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}\n")
    print("Class probabilities:")
    for cls, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar = '#' * int(prob * 40)
        print(f"  {cls:<12} {prob:.4f}  {bar}")
