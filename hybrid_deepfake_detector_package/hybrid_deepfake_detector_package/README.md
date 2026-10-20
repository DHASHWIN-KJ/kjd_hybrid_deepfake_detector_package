# 🛡️ Hybrid Multi-Scale Forensic Deepfake Detection Platform

A state-of-the-art, production-grade Deepfake Detection & Visual Forensic Localization System combining **Semantic Anchors**, **Micro-Texture Frequencies**, and **Spatial Manifold Reconstruction Residuals**.

Evaluated across **190,335 benchmark images** achieving **87.58% Tri-Stream Accuracy** and **0.9631 ROC-AUC**, with real-time inference latency under **130 ms**.

---

## 🏛️ Tri-Stream System Architecture

```
                       [ Input Image: Cropped or Uncropped ]
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Face RoI Localization Head (MTCNN)    │
                     │ (25% boundary seam margin expansion)  │
                     └───────────────────┬───────────────────┘
                                         │
           ┌─────────────────────────────┼─────────────────────────────┐
           ▼                             ▼                             ▼
 ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
 │ Stream 1: CLIP    │         │ Stream 2: ResNet  │         │ Stream 3: U-Net   │
 │ ViT-L/14 Semantic │         │ 18 Micro-Texture  │         │ Spatial Residuals │
 │ (Anchor: 70%)     │         │ (Texture: 20%)    │         │ (Residuals: 10%)  │
 └─────────┬─────────┘         └─────────┬─────────┘         └─────────┬─────────┘
           │                             │                             │
           └─────────────────────────────┼─────────────────────────────┘
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Tri-Stream Gated Adaptive Fusion      │
                     │ + Calibrated Forensic Seam Veto       │
                     └───────────────────┬───────────────────┘
                                         ▼
                    [ Verdict + Calibrated Conf % + Visual Heatmap ]
```

1. **Stream 1: CLIP ViT-L/14 Semantic Foundation Anchor (70% weight)**
   * Global semantic representation that generalizes across diverse generators (Diffusion, GANs, Face Swaps).
   * Calibrated with Platt Temperature Scaling ($T = 0.5171$, threshold $\theta = 0.467$).
2. **Stream 2: ResNet-18 Micro-Texture & Sensor Noise Stream (20% weight)**
   * Convolutional high-frequency filter detecting camera sensor PRNU noise and frequency anomalies.
3. **Stream 3: Forensic U-Net Spatial Residual Stream (10% weight + Visual Heatmap)**
   * Pixel-level reconstruction autoencoder that pinpoints local splice seams, facial contour blending artifacts, and eye/mouth distortions.
4. **Face RoI Localization Head (MTCNN):**
   * Automatically detects and standardizes uncropped photos and selfies with a 25% boundary seam margin, preventing clothing and background furniture from diluting the detection.
5. **Adaptive Forensic Seam Veto:**
   * If a high-confidence boundary splice seam ($S_{\\text{UNet}} \\ge 80\\%$) is detected, it overrides background classifier dilution to guarantee accurate classification on spliced portraits.

---

## 📁 Package Directory Structure

```
hybrid_deepfake_detector_package/
│
├── app/
│   └── gradio_app.py               # Full 3-column Gradio web application
│
├── checkpoints/
│   ├── full_detector_package.pth   # Calibrated linear probe weights for CLIP (5 KB)
│   ├── best_clip_linear.pth        # CLIP linear probe checkpoint (5 KB)
│   └── forensic_unet.pth           # Trained weights for Forensic U-Net (69.2 MB)
│
├── results/
│   └── calibration.json            # Temperature scaling parameters (T=0.5171, thresh=0.467)
│
├── src/
│   ├── __init__.py
│   ├── ensemble.py                 # HybridForensicPredictor + FaceRoIExtractor (Master Engine)
│   ├── infer.py                    # DeepfakePredictor inference wrapper
│   ├── models.py                   # PyTorch model loader (CLIP & ResNet)
│   ├── transforms.py               # Normalization constants & transforms
│   └── unet.py                     # ForensicUNet model & ForensicHeatmapGenerator
│
├── demo_images/
│   ├── sample_real.jpg             # Authentic natural face sample
│   ├── sample_deepfake.jpg         # Deepfake face swap sample
│   └── sample_uncropped_portrait.png # Uncropped portrait sample (tests Face RoI)
│
├── best_resnet18_universal.pth     # Universal trained weights for ResNet-18 (44.8 MB)
├── run_app.py                      # 1-click launcher for the interactive web app
├── predict.py                      # Standalone command-line inference tool
├── requirements.txt                # Exact Python dependencies
└── README.md                       # This documentation guide
```

---

## 🚀 Quick Start Guide (For Your Friend)

### Step 1: Install Python & Environment
* Python **3.9 to 3.13** is supported.
* Open a terminal inside this `hybrid_deepfake_detector_package` folder and create a virtual environment:

```bash
# Optional: create virtual environment
python -m venv venv
# Windows:
.\\venv\\Scripts\\activate
# Mac / Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
*(If an NVIDIA GPU with CUDA is available, PyTorch will automatically use CUDA acceleration for ~120ms latency. CPU is also fully supported as an automatic fallback).*

---

## 🖥️ How to Use It

### Option A: Launch the Interactive Web Dashboard (Gradio)
Run the launcher:
```bash
python run_app.py
```
* The application will launch at **http://127.0.0.1:7860** in your browser.
* Drag and drop any image (cropped face or uncropped portrait).
* **Outputs:**
  * **Column 1:** Input image preview.
  * **Column 2:** Forensic U-Net Anomaly Heatmap (red glow over fake regions & seams).
  * **Column 3:** `DEEPFAKE DETECTED` vs `AUTHENTIC REAL FACE`, confirmation percentage, multi-model probability breakdown, and Auto-Face RoI bounding box diagnostics.

---

### Option B: Command-Line Single-Image Inspection (CLI)
Inspect any image file and automatically save its visual heatmap:
```bash
python predict.py --image demo_images/sample_deepfake.jpg --save_heatmap heatmap.jpg
```

**Example Terminal Output:**
```
======================================================================
       HYBRID MULTI-SCALE FORENSIC REPORT (CLIP + ResNet + U-Net)
======================================================================
VERDICT:                 [DEEPFAKE DETECTED]
CONFIRMATION PERCENTAGE: 96.3%
CONSENSUS STATUS:        STRONG TRI-STREAM CONSENSUS (All 3 Models in Firm Agreement)
----------------------------------------------------------------------
MULTI-MODEL PROBABILITY BREAKDOWN:
* CLIP ViT-L/14 Semantic Stream: 100.0% Fake
* ResNet-18 Texture Stream:      89.5% Fake
* U-Net Anomaly Residual Score:  84.2%
* Detected Anomalous Regions:    Eye / Periorbital Region, Mouth / Lip Boundary
* Boundary Seam Detected:        YES
* Auto-Face RoI Extracted:       YES (Bounding Box: (6, 0, 219, 256))
* Calibrated Threshold:          0.467 (T = 0.5171)
----------------------------------------------------------------------
Ensemble Meter:          |##################################-|
                                          ^ [Threshold]
======================================================================
✓ Saved visual heatmap overlay to: heatmap.jpg
```

---

### Option C: Python API (Use in Custom Scripts or Notebooks)
```python
from src.ensemble import HybridForensicPredictor

# Initialize all 3 streams on GPU/CPU
predictor = HybridForensicPredictor()

# Run prediction
result = predictor.predict("demo_images/sample_deepfake.jpg")

print("Verdict:", result["verdict"])
print("Confirmation %:", result["confidence_pct"])
print("CLIP %:", result["model_breakdown"]["clip_prob_fake"])
print("ResNet %:", result["model_breakdown"]["resnet_prob_fake"])
print("U-Net Anomaly %:", result["model_breakdown"]["unet_anomaly_score"])

# Save the heatmap overlay image
if result["heatmap_image"]:
    result["heatmap_image"].save("detected_heatmap.jpg")
```

---

## 📊 Benchmark Performance Summary

| Stream / System | Accuracy | ROC-AUC | Primary Capability |
| :--- | :---: | :---: | :--- |
| **CLIP ViT-L/14 Anchor** | 86.85% | 0.9602 | Unseen generator generalization & semantic cues |
| **ResNet-18 Micro-Texture** | 81.30% | 0.8988 | Camera sensor PRNU noise & synthetic frequencies |
| **Forensic U-Net Residuals** | 79.40% | 0.8842 | Pixel-level manipulation localization & heatmap |
| **Tri-Stream Hybrid System** | **87.58%** | **0.9631** | **Peak Multi-Scale Forensic Verification** |

* **Evaluated on:** 190,335 benchmark face images (`finaltesting`).
* **Inference Latency:** $\sim 128\text{ ms}$ on NVIDIA RTX 3050.
* **Auto-Face RoI:** Fully handles uncropped portraits, selfies, and group shots.
