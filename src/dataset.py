import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES = ['bacterial', 'fungal', 'healthy', 'mould']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def make_transforms(input_size=300):
    """Return (train_transforms, val_transforms) for a given input image size."""
    resize = int(input_size * 1.067)   # ~7% larger before crop (same ratio as 320→300)
    train = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train, val

# Default transforms (300px) — used by ensemble.py / evaluate.py for non-VGG models
TRAIN_TRANSFORMS, VAL_TRANSFORMS = make_transforms(300)


class PathogenDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def _load_samples(data_dir):
    samples = []
    skipped = 0
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(cls_dir, fname)
            try:
                with Image.open(path) as img:
                    img.verify()
                samples.append((path, CLASS_TO_IDX[cls]))
            except Exception:
                skipped += 1
    if skipped:
        print(f"Skipped {skipped} unreadable image(s).")
    return samples


def get_dataloaders(data_dir, batch_size=32, num_workers=4,
                    val_split=0.15, test_split=0.15, seed=42, input_size=300):
    samples = _load_samples(data_dir)
    labels = [s[1] for s in samples]
    indices = list(range(len(samples)))

    train_idx, rest_idx = train_test_split(
        indices, test_size=val_split + test_split,
        stratify=labels, random_state=seed
    )
    rest_labels = [labels[i] for i in rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_split / (val_split + test_split),
        stratify=rest_labels, random_state=seed
    )

    train_tf, val_tf = make_transforms(input_size)
    train_ds = PathogenDataset([samples[i] for i in train_idx], train_tf)
    val_ds   = PathogenDataset([samples[i] for i in val_idx],   val_tf)
    test_ds  = PathogenDataset([samples[i] for i in test_idx],  val_tf)

    print(f"Splits  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    import torch
    pin = torch.cuda.is_available()   # pin_memory only useful for CUDA, not MPS

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader
