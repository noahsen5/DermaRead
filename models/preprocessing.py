"""
Single source of truth for image preprocessing.

All training scripts, inference helpers, Grad-CAM, and the Gradio app
must import from here — never define transforms inline elsewhere.

If this file changes, re-train ALL models from scratch.
"""

import torchvision.transforms as T

# ImageNet statistics — used for all ResNet variants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
RESIZE_SIZE   = 256
INPUT_SIZE    = 224

# ── Inference transform (no augmentation) ─────────────────────────────────────
# Used by: predict_pil, Grad-CAM, evaluation, debug scripts, Gradio app.
inference_transform = T.Compose([
    T.Resize(RESIZE_SIZE),
    T.CenterCrop(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Training transform (light augmentation) ───────────────────────────────────
# Used only inside ManifestDataset when split == "train".
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.Resize(RESIZE_SIZE),
    T.CenterCrop(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
