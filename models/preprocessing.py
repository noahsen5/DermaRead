"""
Single source of truth for image preprocessing.

All training scripts, inference helpers, Grad-CAM, and the Gradio app
must import from here — never define transforms inline elsewhere.

IMPORTANT: If inference_transform changes, re-train ALL models from scratch
           (it defines how pixels reach the model at test time).
           train_transform changes only require re-training the affected model.
"""

import torchvision.transforms as T

# ImageNet statistics — used for all ResNet variants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
RESIZE_SIZE   = 256
INPUT_SIZE    = 224

# ── Inference transform (no augmentation) ─────────────────────────────────────
# Used by: predict_pil, Grad-CAM, evaluation, debug scripts, Gradio app.
# Must NOT change without retraining all models.
inference_transform = T.Compose([
    T.Resize(RESIZE_SIZE),
    T.CenterCrop(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Training transform (augmented for real-world generalisation) ──────────────
# Used only inside ManifestDataset / ExternalDataset when split == "train".
# Augmentations chosen to match real-world variation without distorting lesion structure:
#   ColorJitter   — different cameras, lighting conditions, skin tones
#   RandomRotation — clinical photos arrive at any orientation
#   RandomVerticalFlip — lesions appear on any body surface
train_transform = T.Compose([
    T.Resize(RESIZE_SIZE),
    T.CenterCrop(INPUT_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(p=0.15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
