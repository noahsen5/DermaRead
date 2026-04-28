from pathlib import Path

from PIL import Image
import torch
from torchvision.models import resnet18
from models.preprocessing import inference_transform

CLASS_NAMES = ["non-psoriasis", "psoriasis"]
_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = _ROOT / "models/checkpoints/resnet18_baseline.pt"


def build_model(weights=None, checkpoint_path: Path | str | None = None, device: str = "cpu"):
    m = resnet18(weights=None if weights is None else "IMAGENET1K_V1")
    m.fc = torch.nn.Linear(m.fc.in_features, len(CLASS_NAMES))
    if checkpoint_path:
        load_checkpoint(m, checkpoint_path, map_location=device)
    m.eval()
    return m.to(device)


def load_checkpoint(model, checkpoint_path, map_location="cpu"):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean_state = {k.split("model.", 1)[-1] if k.startswith("model.") else k: v for k, v in state.items()}
    model.load_state_dict(clean_state)
    return model


def load_trained_model(device: str = "cpu"):
    return build_model(weights=None, checkpoint_path=DEFAULT_CHECKPOINT, device=device)

def predict_pil(model, pil: Image.Image):
    x = inference_transform(pil).unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).cpu()[0].tolist()
    return {k: float(v) for k,v in zip(CLASS_NAMES, p)}
