"""
Single source of truth for image preprocessing.

IMPORTANT: If inference_transform changes, re-train ALL models from scratch.
"""

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
RESIZE_SIZE   = 256
INPUT_SIZE    = 224

inference_transform = T.Compose([
    T.Resize(RESIZE_SIZE),
    T.CenterCrop(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Augmentations simulate real-world photographic variation (lighting, orientation).
# ColorJitter range is conservative to avoid washing out lesion colour information.
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
