# Dataset Setup Guide

The full dataset is **not tracked in Git** due to size. Follow the instructions below to set it up locally.

## Dataset Overview

| Class      | Images | Diseases Included |
|------------|--------|-------------------|
| bacterial  | 440    | Cabbage Black Rot, Cauliflower Bacterial, Cauliflower Black Rot, Lettuce Bacterial, Spinach Bacterial |
| fungal     | 423    | Cabbage Alternaria, Cauliflower Alternaria, Cauliflower Ring Spot, Lettuce Septoria Blight, Lettuce Wilt & Leaf Blight, Spinach Anthracnose |
| healthy    | 440    | Healthy Cabbage, Cauliflower, Lettuce, Spinach |
| mould      | 439    | Cabbage Downy Mildew, Cauliflower Downy Mildew, Lettuce Downy Mildew, Lettuce Powdery Mildew, Spinach Downy Mildew |
| **Total**  | **1,742** | |

## Required Folder Structure

Place your images in the following flat structure under `dataset/flat/`:

```
dataset/
└── flat/
    ├── bacterial/
    │   ├── Cabbage_Black_rot_0001.jpg
    │   ├── Cauliflower_Bacterial_0001.jpg
    │   └── ...
    ├── fungal/
    │   ├── Cabbage_Alternaria_0001.jpg
    │   └── ...
    ├── healthy/
    │   ├── Cabbage_Healthy_0001.jpg
    │   └── ...
    └── mould/
        ├── Cabbage_Downy_0001.jpg
        └── ...
```

## Raw Dataset Structure (optional)

If you have the raw hierarchical dataset, run the flattening script to generate the above structure:

```bash
cd src
python rename_and_flatten.py
```

The raw structure expected:
```
dataset/raw/
├── Healthy/
│   ├── Lettuce Healthy/
│   ├── Cabbage Healthy/
│   ├── Cauliflower Healthy/
│   └── Spinach Healthy/
├── Bacterial/
│   ├── Lettuce Bacterial/
│   ├── Spinach/
│   ├── Cauliflower Bacterial/
│   ├── Cauliflower Black Rot/
│   └── Cabbage Black Rot/
├── Fungal/
│   ├── Cabbage Alternaria/
│   ├── Cauliflower Alternaria/
│   ├── Cauliflower Ring Spot/
│   ├── Septoria Blight Lettuce/
│   ├── Spinach Anthracnose/
│   └── Wilt and Leaf Blight Lettuce/
└── Mould/
    ├── Cabbage Downy/
    ├── Cauliflower Downy/
    ├── Lettuce Downy/
    ├── Lettuce Powdery/
    └── Spinach Downy Mildew/
```

## Image Specifications

- Format: JPEG
- Resolution: Variable (typically 1000×1000 to 2000×2000 px)
- Colour: RGB
- Resized to: 300×300 (or 224×224 for VGG-16) during preprocessing
