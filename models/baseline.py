from pathlib import Path

from PIL import Image
import torch, torchvision.transforms as T
from torchvision.models import resnet18

CLASS_NAMES = ["non-psoriasis", "psoriasis"]
DEFAULT_CHECKPOINT = Path("models/checkpoints/resnet18_baseline.pt")


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
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean_state = {k.split("model.", 1)[-1] if k.startswith("model.") else k: v for k, v in state.items()}
    model.load_state_dict(clean_state)
    return model


def load_trained_model(device: str = "cpu"):
    return build_model(weights=None, checkpoint_path=DEFAULT_CHECKPOINT, device=device)

_pre = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_pil(model, pil: Image.Image):
    x = _pre(pil).unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).cpu()[0].tolist()
    return {k: float(v) for k,v in zip(CLASS_NAMES, p)}
