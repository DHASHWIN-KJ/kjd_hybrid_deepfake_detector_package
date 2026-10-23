"""
Phase 2: Hybrid Multi-Scale Forensic Prediction Engine
======================================================
Unifies the three complementary analytical streams:
1. CLIP ViT-L/14 Foundation Stream: Primary semantic decision anchor (cross-generator generalization).
2. ResNet-18 Stream: Convolutional micro-texture and frequency sensor-noise stream.
3. Forensic U-Net Stream: Spatial boundary seam localization and visual heatmap generation.
"""

import os
import sys
import argparse
from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.infer import DeepfakePredictor
from src.models import load_model
from src.transforms import get_eval_transform
from src.unet import ForensicHeatmapGenerator


class FaceRoIExtractor:
    """
    Automatic Face Region-of-Interest (RoI) Extractor.
    Standardizes real-world / uncropped images into face crops with boundary margins.
    Bypasses cropping if the input is already a tight benchmark face crop (>= 70% area).
    """

    def __init__(
        self,
        margin: float = 0.25,
        min_face_ratio: float = 0.50,
        device: str = "cpu",
    ):
        self.margin = float(margin)
        self.min_face_ratio = float(min_face_ratio)
        self.device = device
        self.mtcnn = None
        self._init_detector()

    def _init_detector(self):
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                keep_all=False,
                select_largest=True,
                post_process=False,
                device=self.device,
            )
            print(f"[roi] Initialized MTCNN Face RoI Detector on {str(self.device).upper()}.")
        except Exception as e:
            print(f"[roi] Notice: MTCNN initialization fallback to OpenCV ({e})")
            self.mtcnn = None

    def extract_face(
        self,
        img: Image.Image,
    ) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]], bool]:
        """
        Extracts face RoI with expanded margin.
        Returns: (processed_image, bounding_box, was_cropped)
        """
        w, h = img.size
        img_area = float(w * h)
        bbox = None

        # 1. Primary: MTCNN detection
        if self.mtcnn is not None:
            try:
                boxes, probs = self.mtcnn.detect(img)
                if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.80:
                    box = boxes[0]
                    bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            except Exception:
                bbox = None

        # 2. Secondary Fallback: OpenCV Haar Cascade
        if bbox is None:
            try:
                import cv2
                np_img = np.array(img)
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = faces[0]
                    bbox = (int(fx), int(fy), int(fx + fw), int(fy + fh))
            except Exception:
                bbox = None

        # If no face found, return original image safely
        if bbox is None:
            return img, None, False

        # 3. Check Smart Passthrough Ratio
        x1, y1, x2, y2 = bbox
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        face_ratio = (face_w * face_h) / img_area

        # If face already occupies >= 70% of canvas, it's already a tight benchmark crop!
        if face_ratio >= self.min_face_ratio:
            return img, bbox, False

        # 4. Expand Margin symmetrically to form a SQUARE 1:1 canvas
        # This preserves natural facial aspect ratio and prevents geometry stretching when resized
        face_cx = (x1 + x2) / 2.0
        face_cy = (y1 + y2) / 2.0
        side = max(face_w, face_h) * (1.0 + 2.0 * self.margin)

        half_side = side / 2.0
        cx1 = max(0, int(round(face_cx - half_side)))
        cy1 = max(0, int(round(face_cy - half_side)))
        cx2 = min(w, int(round(face_cx + half_side)))
        cy2 = min(h, int(round(face_cy + half_side)))

        cropped_img = img.crop((cx1, cy1, cx2, cy2))
        return cropped_img, (cx1, cy1, cx2, cy2), True


class HybridForensicPredictor:
    """
    Hybrid Multi-Scale Forensic System unifying CLIP, ResNet, and Forensic U-Net.
    """

    def __init__(
        self,
        clip_checkpoint: str = "checkpoints/full_detector_package.pth",
        resnet_checkpoint: str = "best_resnet18_universal.pth",
        unet_checkpoint: str = "checkpoints/forensic_unet.pth",
        calibration_path: str = "results/calibration.json",
        alpha: float = 0.85,
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.alpha = float(alpha)
        print("=" * 65)
        print(f"[hybrid] Initializing Hybrid Forensic System on {self.device.upper()}...")
        print("=" * 65)

        # 1. Primary Semantic Stream: CLIP ViT-L/14
        clip_path = os.path.join(PROJECT_ROOT, clip_checkpoint) if not os.path.isabs(clip_checkpoint) else clip_checkpoint
        if not os.path.isfile(clip_path):
            clip_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_clip_linear.pth")

        calib_path = os.path.join(PROJECT_ROOT, calibration_path) if not os.path.isabs(calibration_path) else calibration_path
        self.clip_predictor = DeepfakePredictor(
            checkpoint_path=clip_path,
            calibration_path=calib_path,
            device=self.device,
        )
        self.threshold = self.clip_predictor.threshold
        self.temperature = self.clip_predictor.temperature

        # 2. Texture Stream: ResNet-18 Baseline
        resnet_path = os.path.join(PROJECT_ROOT, resnet_checkpoint) if not os.path.isabs(resnet_checkpoint) else resnet_checkpoint
        self.has_resnet = False
        if os.path.isfile(resnet_path):
            try:
                print(f"[hybrid] Loading ResNet-18 texture model from: {resnet_path}")
                self.resnet_model = load_model("resnet18_baseline", checkpoint_path=resnet_path, device=self.device)
                self.resnet_model.eval()
                self.resnet_transform = get_eval_transform(img_size=224, backbone="resnet")
                self.has_resnet = True
            except Exception as e:
                print(f"[hybrid] Notice: ResNet-18 loading skipped ({e}). Relying solely on CLIP foundation anchor.")
        else:
            print(f"[hybrid] Notice: '{resnet_checkpoint}' not found. Using CLIP primary stream.")

        # 3. Spatial Localization Stream: Forensic U-Net
        self.heatmap_generator = ForensicHeatmapGenerator(device=self.device)

        # 4. Face RoI Localization Head (Adaptive Boundary Normalization)
        self.roi_extractor = FaceRoIExtractor(
            margin=0.25,
            min_face_ratio=0.50,
            device=self.device,
        )

        print("[hybrid] [SUCCESS] All 3 Forensic Streams Initialized Successfully!")
        print("=" * 65)

    def predict(
        self,
        image_input: Union[str, Image.Image],
        generate_heatmap: bool = True,
        auto_crop_face: bool = True,
        threshold: Optional[float] = None,
        forensic_mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Runs hybrid prediction on an image:
        - Automatically standardizes uncropped/real-world portraits to Face RoI (with 25% boundary margin)
        - Evaluates CLIP semantic stream, ResNet micro-texture stream, and Forensic U-Net spatial residuals
        - Autonomously detects and classifies BOTH full-canvas generative deepfakes (Midjourney, StyleGAN)
          AND localized face-swaps / live stream screenshots (SwapFace, OBS DeepFaceLive, Roop) without requiring
          manual mode switches or threshold modifications.
        - Calculates multi-model consensus, confirmation percentage, and visual heatmap
        """
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            img_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert("RGB")
        else:
            raise TypeError("image_input must be a filepath (str) or PIL.Image.Image")

        orig_w, orig_h = img_pil.size
        img_area = float(orig_w * orig_h)
        warnings = []
        if orig_w < 160 or orig_h < 160:
            warnings.append(f"Image resolution ({orig_w}x{orig_h}) is below 160px; detection quality may be reduced.")

        # 0. Face RoI Extraction (Bypasses tight benchmark crops, standardizes uncropped photos)
        if auto_crop_face:
            eval_img, face_bbox, crop_applied = self.roi_extractor.extract_face(img_pil)
        else:
            eval_img, face_bbox, crop_applied = img_pil, None, False

        # Calculate Face Area Ratio relative to original canvas
        face_ratio = 1.0
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            face_ratio = ((fx2 - fx1) * (fy2 - fy1)) / max(1.0, img_area)

        # 1. Stream 1: CLIP ViT-L/14 Semantic Evaluation
        clip_res = self.clip_predictor.predict(eval_img)
        p_fake_clip = float(clip_res["p_fake"])
        p_real_clip = float(clip_res["p_real"])

        # 2. Stream 2: ResNet-18 Texture Evaluation (Full Face RoI)
        p_fake_resnet = None
        if self.has_resnet:
            with torch.no_grad():
                res_tensor = self.resnet_transform(eval_img).unsqueeze(0).to(self.device)
                res_logits = self.resnet_model(res_tensor)
                res_probs = F.softmax(res_logits, dim=1).squeeze().cpu().numpy()
                p_fake_resnet = float(res_probs[1])

        # Multi-Scale Inner-Face Micro-Texture Inspection (tight MTCNN crop isolating eyes/nose/mouth)
        p_fake_resnet_inner = None
        inner_crop = None
        if face_bbox is not None and self.has_resnet:
            try:
                if hasattr(self.roi_extractor, 'mtcnn') and self.roi_extractor.mtcnn is not None:
                    boxes, probs = self.roi_extractor.mtcnn.detect(img_pil)
                    if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.80:
                        ibox = boxes[0]
                        ix1 = max(0, int(ibox[0]))
                        iy1 = max(0, int(ibox[1]))
                        ix2 = min(orig_w, int(ibox[2]))
                        iy2 = min(orig_h, int(ibox[3]))
                        if (ix2 - ix1) >= 48 and (iy2 - iy1) >= 48:
                            inner_crop = img_pil.crop((ix1, iy1, ix2, iy2))
                if inner_crop is not None:
                    with torch.no_grad():
                        in_tensor = self.resnet_transform(inner_crop).unsqueeze(0).to(self.device)
                        in_logits = self.resnet_model(in_tensor)
                        in_probs = F.softmax(in_logits, dim=1).squeeze().cpu().numpy()
                        p_fake_resnet_inner = float(in_probs[1])
            except Exception:
                pass

        eff_resnet = max(p_fake_resnet if p_fake_resnet is not None else 0.0,
                         p_fake_resnet_inner if p_fake_resnet_inner is not None else 0.0)

        # Active decision threshold
        active_thresh = float(threshold) if threshold is not None else self.threshold

        # Foundation anchor: calibrated 75/25 CLIP-dominant base probability
        if self.has_resnet and p_fake_resnet is not None:
            base_prob = 0.75 * p_fake_clip + 0.25 * p_fake_resnet
        else:
            base_prob = p_fake_clip

        # 3. Stream 3: Forensic U-Net Spatial Heatmap & Anomaly Residuals
        heatmap_img = None
        anomalous_regions = []
        anomaly_score = 0.0
        seam_detected = False

        if generate_heatmap:
            u_res = self.heatmap_generator.generate_heatmap(eval_img, classifier_prob=base_prob)
            heatmap_img = u_res["heatmap_image"]
            anomalous_regions = u_res["anomalous_regions"]
            anomaly_score = float(u_res["anomaly_score"])
            seam_detected = bool(u_res["seam_detected"])

        # 4. Multi-Stream Calibration & Weighted Fusion
        spatial_boost = 0.0
        if generate_heatmap:
            p_fake_ensemble = 0.80 * base_prob + 0.20 * anomaly_score
            # Adaptive Spatial Tie-Breaker for subtle generative deepfakes
            if anomaly_score >= 0.65 and 0.35 <= base_prob <= 0.53:
                spatial_boost = (anomaly_score - 0.50) * 0.22
                p_fake_ensemble = min(1.0, p_fake_ensemble + spatial_boost)
        else:
            p_fake_ensemble = base_prob

        # -------------------------------------------------------------
        # AUTONOMOUS LOCALIZED FACE-SWAP DETECTION (Zero-Config Abstraction)
        # -------------------------------------------------------------
        is_faceswap = False
        eval_arr = np.array(eval_img)
        gray = cv2.cvtColor(eval_arr, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Condition A: Uncropped / Contextual Live Stream Face-Swap (DeepFaceLive, OBS live, Roop in uncropped video/photo)
        # The inner face contains neural synthesis artifacts (r_in >= 0.68) with positive divergence over outer camera canvas (>= +0.10)
        # and Forensic U-Net spatial residual confirming manipulation (u >= 0.35)
        if (orig_w > 256 or orig_h > 256) and face_ratio < 0.65:
            if p_fake_resnet_inner is not None and p_fake_resnet_inner >= 0.68:
                r_diff = p_fake_resnet_inner - (p_fake_resnet if p_fake_resnet is not None else 0.0)
                if r_diff >= 0.10 and anomaly_score >= 0.35:
                    is_faceswap = True

        # Condition B: Tightly Cropped Screen Capture Face-Swap (SwapFace, web app screenshot)
        # Both outer and inner ResNet detect synthetic texture (ResNet >= 0.70), U-Net anomaly is high (>= 0.42),
        # CLIP is fooled by human facial features (<= 0.25), and neural-smoothing lack of sensor grain is confirmed (lap_var < 100)
        if not is_faceswap and p_fake_resnet is not None and p_fake_resnet >= 0.70:
            if anomaly_score >= 0.42 and p_fake_clip <= 0.25 and lap_var < 100.0:
                is_faceswap = True

        # Manual override support if explicitly requested
        if forensic_mode == "faceswap":
            is_faceswap = True

        # When localized face-swap is autonomously detected, self-balance fusion towards texture & spatial streams
        if is_faceswap and self.has_resnet:
            fs_prob = 0.50 * eff_resnet + 0.30 * anomaly_score + 0.20 * p_fake_clip
            if fs_prob > p_fake_ensemble:
                p_fake_ensemble = fs_prob

        p_real_ensemble = 1.0 - p_fake_ensemble

        # Decision rule with calibrated threshold
        is_fake = p_fake_ensemble >= active_thresh
        verdict = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC REAL FACE"
        conf_pct = (p_fake_ensemble if is_fake else p_real_ensemble) * 100.0

        # Consensus determination
        if is_faceswap and is_fake:
            consensus = "LOCALIZED FACE-SWAP (Inner-Face Micro-Texture Discrepancy Confirmed)"
        elif self.has_resnet and p_fake_resnet is not None:
            clip_vote = p_fake_clip >= active_thresh
            resnet_vote = p_fake_resnet >= active_thresh
            unet_vote = anomaly_score >= active_thresh

            if clip_vote == resnet_vote == unet_vote:
                consensus = "STRONG TRI-STREAM CONSENSUS (All 3 Models in Firm Agreement)"
            elif spatial_boost > 0:
                consensus = "SPATIAL LOCALIZATION (U-Net Anomaly Confirms Manipulation)"
            elif clip_vote == resnet_vote:
                consensus = "GLOBAL CONSENSUS (CLIP & ResNet Agreement)"
            elif clip_vote:
                consensus = "SEMANTIC ANCHOR (CLIP Foundation Evaluation)"
            else:
                consensus = "TEXTURE ANCHOR (ResNet Micro-Texture Alert)"
        else:
            consensus = "SINGLE ANCHOR (CLIP Foundation Evaluation)"

        return {
            "verdict": verdict,
            "confidence_pct": round(float(conf_pct), 2),
            "p_fake": round(float(p_fake_ensemble), 4),
            "p_real": round(float(p_real_ensemble), 4),
            "consensus": consensus,
            "is_faceswap": is_faceswap,
            "laplacian_variance": round(lap_var, 2),
            "forensic_mode": forensic_mode,
            "model_breakdown": {
                "clip_prob_fake": round(float(p_fake_clip), 4),
                "resnet_prob_fake": round(float(p_fake_resnet), 4) if p_fake_resnet is not None else None,
                "resnet_inner_prob_fake": round(float(p_fake_resnet_inner), 4) if p_fake_resnet_inner is not None else None,
                "unet_anomaly_score": round(float(anomaly_score), 4),
            },
            "heatmap_image": heatmap_img,
            "anomalous_regions": anomalous_regions,
            "seam_detected": seam_detected,
            "roi_info": {
                "face_detected": face_bbox is not None,
                "crop_applied": crop_applied,
                "bbox": face_bbox,
                "cropped_image": eval_img if crop_applied else None,
            },
            "image_size": f"{orig_w}x{orig_h}",
            "warnings": warnings,
            "threshold_used": active_thresh,
            "temperature_used": self.temperature,
        }

    def render_summary(self, res: Dict[str, Any]) -> str:
        """Renders formatted ASCII summary."""
        lines = [
            "=" * 70,
            "       HYBRID MULTI-SCALE FORENSIC REPORT (CLIP + ResNet + U-Net)",
            "=" * 70,
            f"VERDICT:                 [{res['verdict']}]",
            f"CONFIRMATION PERCENTAGE: {res['confidence_pct']:.1f}%",
            f"CONSENSUS STATUS:        {res['consensus']}",
            "-" * 70,
            "MULTI-MODEL PROBABILITY BREAKDOWN:",
            f"* CLIP ViT-L/14 Semantic Stream: {res['model_breakdown']['clip_prob_fake']*100:.1f}% Fake",
        ]
        if res["model_breakdown"]["resnet_prob_fake"] is not None:
            lines.append(f"* ResNet-18 Texture Stream:      {res['model_breakdown']['resnet_prob_fake']*100:.1f}% Fake")
        lines.extend([
            f"* U-Net Anomaly Residual Score:  {res['model_breakdown']['unet_anomaly_score']*100:.1f}%",
            f"* Detected Anomalous Regions:    {', '.join(res['anomalous_regions'])}",
            f"* Boundary Seam Detected:        {'YES' if res['seam_detected'] else 'NO'}",
        ])
        if res.get("roi_info", {}).get("crop_applied"):
            bbox = res["roi_info"]["bbox"]
            lines.append(f"* Auto-Face RoI Extracted:       YES (Bounding Box: {bbox})")
        else:
            lines.append(f"* Auto-Face RoI Extracted:       NO (Direct Benchmark Face Evaluation)")

        lines.extend([
            f"* Calibrated Threshold:          {res['threshold_used']:.3f} (T = {res['temperature_used']:.4f})",
            "-" * 70,
        ])

        # Meter
        bar_width = 35
        p_fake = res["p_fake"]
        fake_fill = int(round(p_fake * bar_width))
        bar_str = "#" * fake_fill + "-" * (bar_width - fake_fill)
        thresh_pos = int(round(res["threshold_used"] * bar_width))
        pointer = " " * thresh_pos + "^ [Threshold]"
        lines.append(f"Ensemble Meter:          |{bar_str}|")
        lines.append(f"                          {pointer}")
        lines.append("=" * 70)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Forensic Multi-Scale Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to face image.")
    parser.add_argument("--save_heatmap", type=str, default="results/hybrid_heatmap_output.jpg", help="Path to save heatmap overlay.")
    args = parser.parse_args()

    predictor = HybridForensicPredictor()
    res = predictor.predict(args.image)
    print(predictor.render_summary(res))

    if res["heatmap_image"] and args.save_heatmap:
        os.makedirs(os.path.dirname(args.save_heatmap), exist_ok=True)
        res["heatmap_image"].save(args.save_heatmap)
        print(f"[hybrid] Saved visual heatmap overlay to: {args.save_heatmap}\n")


if __name__ == "__main__":
    main()
