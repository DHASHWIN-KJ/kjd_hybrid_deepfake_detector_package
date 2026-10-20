"""
Phase 3 / Phase 6 -- Model Architectures
========================================
Implements:
1. ResNet-18 baseline: ImageNet-pretrained backbone with custom dropout + linear head.
2. Checkpoint loader for existing weights (e.g., best_resnet18_universal.pth).
3. Forward helper returning probabilities P(Fake).

Usage:
    python -m src.models
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Detector(nn.Module):
    """
    ResNet-18 Deepfake Detector.
    Backbone: ImageNet-pretrained ResNet-18.
    Head: Sequential(Dropout(p), Linear(512, 2)).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, 2) where class 0 = Real, class 1 = Fake."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probability of being Fake (B,)."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]


class CLIPLinearProbe(nn.Module):
    """
    Linear Probe head on top of frozen CLIP embeddings.
    """

    def __init__(self, in_features: int = 768, dropout: float = 0.0):
        super().__init__()
        if dropout > 0.0:
            self.net = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1),
            )
        else:
            self.net = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B,)."""
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability of being Fake (B,) in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


class CLIPDetector(nn.Module):
    """
    End-to-end CLIP Deepfake Detector:
    Frozen CLIP visual backbone + Linear Probe head.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        probe_head: nn.Module = None,
        device: str = "cpu",
    ):
        super().__init__()
        clip_path = os.path.expanduser("~/.cache/clip/ViT-L-14.pt")
        clip_model = torch.jit.load(clip_path, map_location=device)
        self.visual = clip_model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.visual.eval()

        self.probe = probe_head if probe_head is not None else CLIPLinearProbe(in_features=768)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "probe_state_dict" in state_dict:
                state_dict = state_dict["probe_state_dict"]
            self.probe.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_in = x.half() if x.is_cuda else x.float()
            feats = self.visual(x_in).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return self.probe(feats)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


def load_model(
    model_type: str = "resnet18_baseline",
    checkpoint_path: str = None,
    device: str = None,
) -> nn.Module:
    """
    Builds model and loads weights from checkpoint if provided.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "resnet18_baseline":
        model = ResNet18Detector(num_classes=2, pretrained=False)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.backbone.load_state_dict(state_dict)
            print(f"[models] Loaded weights from {checkpoint_path}")
        elif checkpoint_path:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    elif model_type in ("clip_linear", "clip_vit_l_14"):
        model = CLIPDetector(checkpoint_path=checkpoint_path, device=device)
        print(f"[models] Loaded CLIP ViT-L/14 model on {device}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[models] Device: {dev}")
    ckpt = "best_resnet18_universal.pth"
    if os.path.isfile(ckpt):
        m = load_model("resnet18_baseline", checkpoint_path=ckpt, device=dev)
        dummy = torch.randn(2, 3, 224, 224).to(dev)
        with torch.no_grad():
            out = m(dummy)
            probs = m.predict_proba(dummy)
        print(f"[models] Test forward pass successful: logits {out.shape}, probs {probs.shape}")
