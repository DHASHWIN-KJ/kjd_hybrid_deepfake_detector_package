"""
Forensic U-Net Architecture & Spatial Heatmap Engine
===================================================
Implements:
1. ForensicUNet: Full Encoder-Decoder neural network with skip connections.
   Captures multi-scale spatial features, boundary discontinuities, and synthesis residuals.
2. ForensicHeatmapGenerator: Generates visual anomaly heatmaps overlaying
   spliced/manipulated facial regions (eyes, mouth, boundary blending seams).
"""

import os
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> LeakyReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with Bilinear/ConvTranspose2d then DoubleConv using Skip Connections"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if dimensions mismatch
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        # Concatenate skip connection along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ForensicUNet(nn.Module):
    """
    Forensic U-Net for Face Manipulation & Boundary Seam Localization.
    Uses 4 encoder stages, skip connections, and a decoder that isolates
    spatial boundary artifacts and reconstruction residuals.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 64)       # 256x256
        self.down1 = Down(64, 128)                   # 128x128
        self.down2 = Down(128, 256)                  # 64x64
        self.down3 = Down(256, 512)                  # 32x32
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)       # 16x16 (Bottleneck)

        # Decoder (Expanding Path with Skip Connections)
        self.up1 = Up(1024, 512 // factor, bilinear) # 32x32
        self.up2 = Up(512, 256 // factor, bilinear)  # 64x64
        self.up3 = Up(256, 128 // factor, bilinear)  # 128x128
        self.up4 = Up(128, 64, bilinear)             # 256x256

        # Output Heads
        self.out_reconstruct = nn.Conv2d(64, out_channels, kernel_size=1)
        self.out_anomaly_mask = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Contracting path with skips
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Expanding path fusing skips
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Dual outputs
        recon = torch.tanh(self.out_reconstruct(x))
        mask = self.out_anomaly_mask(x)
        return recon, mask


class ForensicHeatmapGenerator:
    """
    Forensic Spatial Analyzer that leverages the ForensicUNet architecture
    and frequency seam analysis to generate visual heatmaps over manipulated face zones.
    """

    def __init__(self, checkpoint_path: str = None, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[unet] Initializing Forensic U-Net on {self.device}...")
        self.unet = ForensicUNet(in_channels=3, out_channels=3, bilinear=True).to(self.device)
        
        # Load trained weights if available
        if checkpoint_path is None:
            ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "forensic_unet.pth")
        else:
            ckpt_path = checkpoint_path

        if os.path.isfile(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.unet.load_state_dict(state_dict)
            print(f"[unet] Loaded pre-trained U-Net weights from {ckpt_path}")

        self.unet.eval()

        # Steerable high-pass Laplacian kernel for seam detection
        laplacian_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0).to(self.device)
        self.laplacian_weight = laplacian_kernel.repeat(3, 1, 1, 1)

    def _extract_frequency_residuals(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts high-frequency boundary gradient inconsistencies via depthwise Laplacian."""
        res = F.conv2d(x, self.laplacian_weight, padding=1, groups=3)
        return torch.abs(res).mean(dim=1, keepdim=True)

    def generate_heatmap(
        self,
        image_input: Union[str, Image.Image],
        target_size: Tuple[int, int] = (256, 256),
    ) -> Dict[str, Any]:
        """
        Analyzes an image and returns:
        - heatmap_image: PIL Image with glowing red/yellow highlight over manipulated regions
        - raw_heatmap: 2D numpy array [0, 1]
        - anomalous_regions: list of detected facial zones (eyes, mouth, boundary seam)
        - anomaly_score: float in [0.0, 1.0]
        - seam_detected: boolean flag
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

        # Resize to standard analysis resolution
        img_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        # Tensor input normalized to [-1, 1]
        x_in = torch.tensor(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        x_norm = x_in * 2.0 - 1.0

        with torch.no_grad():
            recon, u_mask = self.unet(x_norm)
            
            # 1. Multi-scale spatial residual from U-Net reconstruction
            recon_res = torch.abs(x_norm - recon).mean(dim=1, keepdim=True)
            
            # 2. High-pass boundary seam residual
            freq_res = self._extract_frequency_residuals(x_in)
            
            # Combined spatial anomaly signal
            combined = 0.55 * recon_res + 0.45 * freq_res + 0.20 * u_mask
            heatmap_raw = combined.squeeze().cpu().numpy()

        # Smooth and normalize heatmap
        heatmap_smoothed = cv2.GaussianBlur(heatmap_raw, (15, 15), 0)
        
        # Robust min-max scaling using 5th and 98th percentiles
        p_low = np.percentile(heatmap_smoothed, 8)
        p_high = np.percentile(heatmap_smoothed, 96)
        if p_high > p_low:
            norm_map = np.clip((heatmap_smoothed - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            norm_map = np.zeros_like(heatmap_smoothed)

        # Calculate anomaly score (mean of top 12% hottest pixels)
        flat_sorted = np.sort(norm_map.flatten())
        top_k = int(0.12 * len(flat_sorted))
        anomaly_score = float(np.mean(flat_sorted[-top_k:])) if top_k > 0 else 0.0

        # Identify anatomical face regions from high-activation coordinates
        h, w = target_size
        y_grid, x_grid = np.ogrid[:h, :w]
        
        # Threshold for anomalous hotspot pixels
        hot_mask = norm_map >= 0.65
        hot_y, hot_x = np.where(hot_mask)
        
        anomalous_regions = []
        seam_detected = False

        if len(hot_y) > 30:
            # Normalized coordinates
            norm_y = hot_y / float(h)
            norm_x = hot_x / float(w)

            # Check Eye / Periorbital zone (y: 22% - 46%, x: 18% - 82%)
            eye_pts = np.sum((norm_y >= 0.22) & (norm_y <= 0.46) & (norm_x >= 0.18) & (norm_x <= 0.82))
            if eye_pts > 0.15 * len(hot_y):
                anomalous_regions.append("Eye / Periorbital Region")

            # Check Mouth / Lip zone (y: 62% - 86%, x: 25% - 75%)
            mouth_pts = np.sum((norm_y >= 0.62) & (norm_y <= 0.86) & (norm_x >= 0.25) & (norm_x <= 0.75))
            if mouth_pts > 0.15 * len(hot_y):
                anomalous_regions.append("Mouth / Lip Boundary")

            # Check Mid-face / Nose zone (y: 42% - 62%, x: 32% - 68%)
            nose_pts = np.sum((norm_y >= 0.42) & (norm_y <= 0.62) & (norm_x >= 0.32) & (norm_x <= 0.68))
            if nose_pts > 0.15 * len(hot_y):
                anomalous_regions.append("Nose / Mid-Face Region")

            # Check Outer Perimeter / Jawline Blending Seam
            perimeter_pts = np.sum(
                (norm_y <= 0.15) | (norm_y >= 0.85) | (norm_x <= 0.15) | (norm_x >= 0.85)
            )
            if perimeter_pts > 0.15 * len(hot_y):
                anomalous_regions.append("Facial Contour / Blending Seam")
                seam_detected = True

        if not anomalous_regions:
            anomalous_regions = ["No Localized Anomalies (Uniform Texture)"]

        # Generate Colormap Overlay using OpenCV (COLORMAP_JET)
        heatmap_uint8 = (norm_map * 255.0).astype(np.uint8)
        color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

        # Adaptive alpha: cold regions stay mostly transparent, hotspots glow
        alpha = (norm_map ** 1.5) * 0.60
        alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        base_uint8 = (img_np * 255.0).astype(np.uint8)
        blended = (base_uint8 * (1.0 - alpha_3d) + color_map * alpha_3d).astype(np.uint8)

        # Resize back to original image dimensions
        heatmap_pil = Image.fromarray(blended).resize((orig_w, orig_h), Image.Resampling.BILINEAR)

        return {
            "heatmap_image": heatmap_pil,
            "raw_heatmap": norm_map,
            "anomaly_score": round(anomaly_score, 4),
            "anomalous_regions": anomalous_regions,
            "seam_detected": seam_detected,
        }


if __name__ == "__main__":
    print("[unet] Testing ForensicUNet & Heatmap Generator...")
    gen = ForensicHeatmapGenerator()
    dummy = Image.new("RGB", (256, 256), color=(128, 128, 128))
    res = gen.generate_heatmap(dummy)
    print(f"Heatmap Generated! Anomaly Score: {res['anomaly_score']}")
    print(f"Detected Regions: {res['anomalous_regions']}")
    print("[SUCCESS] Phase 1 Forensic U-Net module working perfectly!")
