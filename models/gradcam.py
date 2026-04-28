"""Minimal Grad-CAM helpers for the ResNet18 baseline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = _ROOT / "docs/mpl-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
MPL_CACHE.mkdir(parents=True, exist_ok=True)

import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from models.preprocessing import inference_transform as _PREPROCESS


def _default_layer(model):
    # ResNet50 uses Bottleneck (conv1/conv2/conv3); ResNet18 uses BasicBlock (conv1/conv2).
    # Hook the final spatial conv in the last residual block.
    try:
        from torchvision.models.resnet import Bottleneck
        if isinstance(model.layer4[-1], Bottleneck):
            return model.layer4[-1].conv3
    except ImportError:
        pass
    return model.layer4[-1].conv2


def compute_gradcam(model, pil_img: Image.Image, target_class: Optional[int] = None, target_layer=None):
    model.eval()
    target_layer = target_layer or _default_layer(model)
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)

    device = next(model.parameters()).device
    x = _PREPROCESS(pil_img).unsqueeze(0).to(device)

    try:
        logits = model(x)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1))
        loss = logits[:, target_class]
        model.zero_grad()
        loss.backward()

        acts = activations[-1]
        grads = gradients[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1)).squeeze().cpu().numpy()
        if cam.size == 0:
            raise RuntimeError("Grad-CAM produced an empty map.")
        cam -= cam.min()
        cam /= cam.max() + 1e-8
    finally:
        h_forward.remove()
        h_backward.remove()

    heatmap = np.uint8(cam * 255)
    heatmap = Image.fromarray(heatmap).resize(pil_img.size, resample=Image.BILINEAR)
    colormap = matplotlib.colormaps["jet"]
    heatmap = Image.fromarray(np.uint8(colormap(np.array(heatmap) / 255.0) * 255))
    return heatmap


def overlay_heatmap(pil_img: Image.Image, heatmap: Image.Image, alpha: float = 0.4) -> Image.Image:
    heat_rgba = heatmap.convert("RGBA")
    alpha_channel = heat_rgba.split()[-1].point(lambda v: int(alpha * v))
    heat_rgba.putalpha(alpha_channel)
    base = pil_img.convert("RGBA")
    return Image.alpha_composite(base, heat_rgba).convert("RGB")


def save_gradcam(model, pil_img: Image.Image, out_path: Path, target_class: Optional[int] = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = compute_gradcam(model, pil_img, target_class=target_class)
    overlay = overlay_heatmap(pil_img, heatmap)
    overlay.save(out_path)
    return out_path
