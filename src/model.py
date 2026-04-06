import torch.nn as nn
from torchvision.models import (
    efficientnet_b3,      EfficientNet_B3_Weights,
    resnet50,             ResNet50_Weights,
    vgg16,                VGG16_Weights,
    densenet121,          DenseNet121_Weights,
    mobilenet_v3_large,   MobileNet_V3_Large_Weights,
)

NUM_CLASSES = 4
ARCHS = ['efficientnet_b3', 'resnet50', 'vgg16', 'densenet121', 'mobilenet_v3_large']

# Input image size per arch (MPS adaptive_avg_pool2d requires sizes divisible by pool output)
# VGG16 pool output is 7x7 → needs input divisible by 32 → 224 is standard
ARCH_INPUT_SIZE = {
    'efficientnet_b3':    300,
    'resnet50':           300,
    'vgg16':              224,   # MPS constraint: 224/32=7, divisible
    'densenet121':        300,
    'mobilenet_v3_large': 300,
}


def build_model(arch='efficientnet_b3', num_classes=NUM_CLASSES, pretrained=True):
    if arch == 'efficientnet_b3':
        m = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif arch == 'resnet50':
        m = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == 'vgg16':
        m = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)

    elif arch == 'densenet121':
        m = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    elif arch == 'mobilenet_v3_large':
        m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)

    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {ARCHS}")
    return m


def freeze_backbone(model, arch):
    """Freeze entire backbone; only the final classification layer stays trainable."""
    for p in model.parameters():
        p.requires_grad = False
    # For VGG16 the full classifier is huge (119M) — only train the last linear layer
    if arch == 'vgg16':
        model.classifier[6].requires_grad_(True)
    else:
        _head(model, arch).requires_grad_(True)


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def get_param_groups(model, arch, backbone_lr, head_lr):
    """Differential LRs: lower for backbone, higher for head."""
    head_ids = {id(p) for p in _head(model, arch).parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head     = [p for p in model.parameters() if id(p) in head_ids]
    return [{'params': backbone, 'lr': backbone_lr},
            {'params': head,     'lr': head_lr}]


def disable_inplace_relu(model):
    """Set inplace=False on all ReLU layers so backward hooks work for Grad-CAM."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def gradcam_target_layer(model, arch):
    """Return the last conv/feature layer suitable for Grad-CAM."""
    if arch == 'efficientnet_b3':   return model.features[-1]
    if arch == 'resnet50':          return model.layer4[-1]
    if arch == 'vgg16':             return model.features[-3]   # last conv before pool
    if arch == 'densenet121':       return model.features.denseblock4
    if arch == 'mobilenet_v3_large': return model.features[-1]


# ── internal helper ────────────────────────────────────────────────────────
def _head(model, arch):
    if arch == 'resnet50':  return model.fc
    return model.classifier
