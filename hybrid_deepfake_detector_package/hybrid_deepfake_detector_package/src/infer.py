"""
Phase 8 -- Inference Module
===========================
Implements single-image and batch deepfake detection:
1. Applies exact standardized preprocessing & CLIP normalization.
2. Loads temperature calibration T and decision threshold from calibration.json.
3. Performs calibrated inference with out-of-range quality warnings.
4. Renders probability bar-chart visualization.

Usage:
    python -m src.infer --image path/to/face.jpg
"""

import os
import json
import argparse
from typing import Dict, Any, Union
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.models import CLIPDetector, load_model
from src.transforms import get_eval_transform


class DeepfakePredictor:
    """
    Production-ready Inference Engine for Deepfake Detection.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_clip_linear.pth",
        calibration_path: str = "results/calibration.json",
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load calibration parameters
        self.temperature = 1.0
        self.threshold = 0.5

        if os.path.isfile(calibration_path):
            with open(calibration_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
                self.temperature = float(calib.get("temperature", 1.0))
                self.threshold = float(calib.get("threshold", 0.5))
                print(f"[infer] Loaded calibration from json: T = {self.temperature:.4f}, Threshold = {self.threshold:.3f}")
        elif os.path.isfile(checkpoint_path):
            try:
                pkg = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(pkg, dict) and "temperature" in pkg and "threshold" in pkg:
                    self.temperature = float(pkg["temperature"])
                    self.threshold = float(pkg["threshold"])
                    print(f"[infer] Loaded calibration from package: T = {self.temperature:.4f}, Threshold = {self.threshold:.3f}")
            except Exception:
                print("[infer] Warning: Using default T=1.0, threshold=0.5")
        else:
            print("[infer] Warning: calibration.json not found, using default T=1.0, threshold=0.5")

        # Load CLIP ViT-L/14 model
        print(f"[infer] Loading CLIP ViT-L/14 model on {self.device}...")
        self.model = load_model(
            model_type="clip_linear",
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        self.model.eval()

        # Deterministic evaluation transform with CLIP normalization
        self.transform = get_eval_transform(img_size=224, backbone="clip")

    def predict(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Runs inference on a single image path or PIL Image.
        Returns a dictionary with prediction label, calibrated probabilities, and warnings.
        """
        warnings = []

        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            try:
                img = Image.open(image_input).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to open image {image_input}: {e}")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError("image_input must be a filepath (str) or PIL.Image.Image")

        # Quality check: resolution warning
        w, h = img.size
        if w < 64 or h < 64:
            warnings.append(f"Image resolution is very small ({w}x{h} px). Predictions may be unreliable.")

        # Transform
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            x_in = img_tensor.half() if self.device == "cuda" else img_tensor.float()
            feats = self.model.visual(x_in).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            raw_logit = self.model.probe(feats).squeeze()

            # Calibrate logit with temperature
            cal_logit = (raw_logit / self.temperature).cpu().item()
            p_fake = 1.0 / (1.0 + np.exp(-cal_logit))
            p_real = 1.0 - p_fake

        # Decision rule
        label = "Fake" if p_fake >= self.threshold else "Real"
        confidence = p_fake if label == "Fake" else p_real

        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "p_fake": round(float(p_fake), 4),
            "p_real": round(float(p_real), 4),
            "raw_logit": round(float(raw_logit), 4),
            "calibrated_logit": round(float(cal_logit), 4),
            "threshold_used": self.threshold,
            "temperature_used": self.temperature,
            "image_size": f"{w}x{h}",
            "warnings": warnings,
        }

    @staticmethod
    def render_bar(result: Dict[str, Any], bar_width: int = 35) -> str:
        """Renders an ASCII probability bar chart."""
        p_fake = result["p_fake"]
        p_real = result["p_real"]
        label = result["label"]
        thresh = result["threshold_used"]

        fake_fill = int(round(p_fake * bar_width))
        real_fill = bar_width - fake_fill

        bar = "#" * fake_fill + "-" * real_fill
        threshold_pos = int(round(thresh * bar_width))
        marker_line = " " * threshold_pos + "^ [Threshold: " + f"{thresh:.2f}]"

        output = [
            f"\nPrediction:  [{label.upper()}] (Confidence: {result['confidence'] * 100:.1f}%)",
            f"Image Size:  {result['image_size']}",
            f"Probability: [Real: {p_real * 100:.1f}%]  |  [Fake: {p_fake * 100:.1f}%]",
            f"Bar Chart:   |{bar}|",
            f"             {marker_line}",
        ]
        if result["warnings"]:
            output.append("\nWarnings:")
            for w in result["warnings"]:
                output.append(f"  - {w}")

        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Deepfake Single-Image Inference")
    parser.add_argument("--image", required=True, help="Path to input face image")
    parser.add_argument("--checkpoint", default="checkpoints/best_clip_linear.pth", help="Checkpoint path")
    parser.add_argument("--calibration", default="results/calibration.json", help="Calibration JSON")
    args = parser.parse_args()

    predictor = DeepfakePredictor(checkpoint_path=args.checkpoint, calibration_path=args.calibration)
    res = predictor.predict(args.image)
    print(predictor.render_bar(res))


if __name__ == "__main__":
    main()
