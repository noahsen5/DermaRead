"""ResNet50 model definition — V2 (transfer), V3 (balanced), V4 (external fine-tune) checkpoints."""

from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet50
from models.preprocessing import inference_transform

CLASS_NAMES = ["non-psoriasis", "psoriasis"]
_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_V2 = _ROOT / "models/checkpoints/resnet50_v2.pt"
DEFAULT_CHECKPOINT_V3 = _ROOT / "models/checkpoints/resnet50_v3_balanced.pt"
DEFAULT_CHECKPOINT_V4 = _ROOT / "models/checkpoints/resnet50_v4_external.pt"

def build_resnet50(weights="IMAGENET1K_V2", checkpoint_path=None, device="cpu"):
    m = resnet50(weights=None if weights is None else "IMAGENET1K_V2")
    m.fc = torch.nn.Linear(m.fc.in_features, len(CLASS_NAMES))
    if checkpoint_path:
        _load_checkpoint(m, checkpoint_path, map_location=device)
    m.eval()
    return m.to(device)


def _load_checkpoint(model, checkpoint_path, map_location="cpu"):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = {k.split("model.", 1)[-1] if k.startswith("model.") else k: v for k, v in state.items()}
    model.load_state_dict(clean)
    return model


def load_v2_model(device="cpu"):
    return build_resnet50(weights=None, checkpoint_path=DEFAULT_CHECKPOINT_V2, device=device)


def load_v3_model(device="cpu"):
    return build_resnet50(weights=None, checkpoint_path=DEFAULT_CHECKPOINT_V3, device=device)


def load_v4_model(device="cpu"):
    return build_resnet50(weights=None, checkpoint_path=DEFAULT_CHECKPOINT_V4, device=device)


def predict_pil(model, pil: Image.Image):
    x = inference_transform(pil).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).cpu()[0].tolist()
    return {k: float(v) for k, v in zip(CLASS_NAMES, p)}
