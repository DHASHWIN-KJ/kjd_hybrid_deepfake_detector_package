# 🛡️ Hybrid Multi-Scale Forensic Deepfake Detection Platform

A state-of-the-art, production-grade Deepfake Detection & Visual Forensic Localization System combining **Semantic Anchors (CLIP ViT-L/14)**, **Micro-Texture Frequencies (ResNet-18)**, and **Spatial Manifold Reconstruction Residuals (Forensic U-Net)**.

Evaluated across **190,335 benchmark images** achieving **87.58% Hybrid Accuracy** and **0.9631 ROC-AUC**, with real-time inference latency under **130 ms**.

---

## 🏛️ Tri-Stream System Architecture

```
                       [ Input Image: Casual / Web / Benchmark ]
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │ Stage 0: Adaptive Face RoI Extractor  │
                      │ - MTCNN with 25% boundary margin      │
                      │ - Smart Benchmark Passthrough         │
                      │ - Symmetrical 1:1 Zero-Distortion Crop│
                      └───────────────────┬───────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
  ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
  │ Stream 1: CLIP    │         │ Stream 2: ResNet  │         │ Stream 3: U-Net   │
  │ ViT-L/14 Semantic │         │ 18 Micro-Texture  │         │ Spatial Residuals │
  │ 768-D Embedding   │         │ 512-D Embedding   │         │ Multi-Scale L1    │
  │ Calibrated Probe  │         │ Sensor / PRNU     │         │ Elliptical Mask   │
  └─────────┬─────────┘         └─────────┬─────────┘         └─────────┬─────────┘
            │                             │                             │
            │ P_fake(CLIP)                │ P_fake(ResNet)              │ Anomaly Heatmap
            └──────────────┬──────────────┘                             │ & Region Tags
                           ▼                                            │
        ┌───────────────────────────────────────┐                       │
        │ Continuous Calibrated Blend (80 / 20) │                       │
        │ P_base = 0.80*P_CLIP + 0.20*P_ResNet  │                       │
        │ + Live-Stream & SwapFace Guard        │                       │
        └──────────────────┬────────────────────┘                       │
                           ▼                                            │
        ┌───────────────────────────────────────────────────────────────┴──┐
        │            Decision Engine with Uncertainty Margin               │
        │  * Crisp Fake: P_ens >= theta + 0.06 -> [DEEPFAKE DETECTED]      │
        │  * Borderline: |P_ens - theta| < 0.06 -> [BORDERLINE / REVIEW]   │
        │  * Crisp Real: P_ens <= theta - 0.06 -> [AUTHENTIC REAL FACE]    │
        └──────────────────┬───────────────────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                       Final Forensic Output                      │
        │  * Authenticity Verdict & Confirmation Percentage (%)            │
        │  * Multi-Model Stream Probability Breakdown                      │
        │  * Pixel-Level Jet Colormap Anomaly Heatmap Overlay              │
        │  * Anatomical Region Tags: Eyes, Mouth, Nose, Blending Seam      │
        └──────────────────────────────────────────────────────────────────┘
```

### The Three Forensic Perspectives:
1. **Stream 1: Foundation Semantic Stream (CLIP ViT-L/14 — 80% Decision Weight)**
   * Global semantic representation that generalizes across unseen generators (StyleGAN, Diffusion, Face Swaps).
   * Calibrated with Temperature Scaling ($T = 0.6433$, decision threshold $\theta = 0.467$).
2. **Stream 2: Micro-Texture & Sensor Noise Stream (ResNet-18 — 20% Decision Weight)**
   * Deep convolutional filter detecting sub-pixel synthesis fingerprints and high-frequency anomalies.
3. **Stream 3: Spatial Residual Engine & Heatmaps (Forensic U-Net)**
   * Full encoder-decoder autoencoder with skip connections and a smooth anatomical elliptical face mask.
   * Renders pixel-level Jet-colormap overlays highlighting manipulated eyes, mouth, and boundary blending seams.
4. **Adaptive Face RoI Extractor (MTCNN):**
   * Automatically detects and standardizes uncropped casual photos, selfies, and group shots with a 25% boundary seam margin, while passing through pre-cropped benchmark portraits without resampling degradation.
5. **Non-Destructive Live-Stream & SwapFace Guard:**
   * Specifically detects subtle live-stream face swaps (OBS DeepFaceLive, SwapFace) exhibiting low Laplacian variance ($< 35.0$) coupled with high texture activations ($\ge 0.65$).

---

## 📁 Package Directory Structure

```
hybrid_deepfake_detector_package/
│
├── app/
│   └── gradio_app.py               # Interactive 3-column Gradio web dashboard
│
├── checkpoints/
│   ├── full_detector_package.pth   # Calibrated linear probe weights for CLIP (5 KB)
│   ├── best_clip_linear.pth        # CLIP linear probe checkpoint (5 KB)
│   └── forensic_unet.pth           # Trained weights for Forensic U-Net (69.2 MB)
│
├── demo_images/                    # Quick test images for instant verification
│   ├── sample_real.jpg             # Authentic natural face sample
│   ├── sample_deepfake.jpg         # Full synthetic deepfake face sample
│   ├── sample_faceswap.png         # Latent face-swap sample (RoOP / Inswapper)
│   └── sample_obs_live.png         # Live-stream face swap capture (OBS DeepFaceLive)
│
├── features/                       # 11,960 pre-extracted latent embeddings (32.3 MB)
│   ├── ciplab.npz                  # 1,960 embeddings (CIPLAB component edits)
│   ├── ds1.npz                     # 2,000 embeddings (StyleGAN synthesis)
│   ├── ds2.npz                     # 2,000 embeddings (Diffusion synthesis)
│   ├── ds3.npz                     # 2,000 embeddings (Blend-based edits)
│   ├── ds4.npz                     # 2,000 embeddings (Re-enactment / warping)
│   └── original_inswapper.npz      # 2,000 embeddings (Paired Inswapper swaps)
│
├── results/
│   └── calibration.json            # Calibrated parameters (T=0.6433, theta=0.467)
│
├── src/
│   ├── __init__.py
│   ├── ensemble.py                 # Master HybridForensicPredictor & FaceRoIExtractor
│   ├── infer.py                    # Calibrated inference engine (CLIP ViT-L/14)
│   ├── models.py                   # PyTorch models with auto-download for CLIP
│   ├── transforms.py               # Normalization constants & transforms
│   └── unet.py                     # ForensicUNet autoencoder & heatmap generator
│
├── best_resnet18_universal.pth     # Universal trained weights for ResNet-18 (44.8 MB)
├── run_app.py                      # 1-click launcher for the web application
├── predict.py                      # Standalone command-line inference tool
├── requirements.txt                # Exact Python dependencies
├── EMBEDDINGS_SIZE_AND_SPECIFICATIONS.md # Full embedding dimensions & memory specs
└── README.md                       # This user documentation manual
```

---

## 🚀 Quick Start Guide

### Step 1: Install Python & Environment
* Compatible with Python **3.9 to 3.13** (Windows, Linux, or macOS).
* Open a terminal inside this `hybrid_deepfake_detector_package` folder:

```bash
# Optional: create a virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚡ **Hardware Acceleration:** If an NVIDIA GPU with CUDA is present, PyTorch will automatically use it for ultra-fast $\sim 120\text{ ms}$ latency. If not, CPU is automatically supported as a seamless fallback.  
> 🌐 **Auto-Download:** On the very first run, if the OpenAI CLIP ViT-L/14 visual backbone is not yet cached on your system, it will automatically download the official checkpoint from OpenAI's CDN directly into `~/.cache/clip/`.

---

## 🖥️ How to Run the System

### Option A: Interactive Web Dashboard (Gradio) — Recommended
Run the launcher:
```bash
python run_app.py
```
*(Or `python app/gradio_app.py`)*

* The dashboard will open in your browser at: **`http://127.0.0.1:7860`**
* **Features:**
  * **Column 1:** Input face (Drag & drop any photo or click the built-in sample test cards).
  * **Column 2:** Real-time Forensic U-Net Heatmap overlay (glowing red over anomalous regions).
  * **Column 3:** `DEEPFAKE DETECTED` vs `AUTHENTIC REAL FACE` verdict card, calibrated confirmation percentage, multi-model probability breakdown, anatomical region tags (Eyes, Mouth, Nose, Seams), and Auto-Face RoI diagnostics.
  * **Sensitivity Slider:** Real-time calibration slider to adjust operational decision threshold $\theta$.

---

### Option B: Standalone Command-Line Tool (CLI)
Inspect any image file directly from the command line and optionally save its visual heatmap overlay:

```bash
python predict.py --image demo_images/sample_deepfake.jpg --save_heatmap output_heatmap.jpg
```

**Real Output Example:**
```
======================================================================
       HYBRID MULTI-SCALE FORENSIC REPORT (CLIP + ResNet + U-Net)
======================================================================
VERDICT:                 [DEEPFAKE DETECTED]
CONFIRMATION PERCENTAGE: 93.0%
CONSENSUS STATUS:        STRONG TRI-STREAM CONSENSUS (All 3 Models in Firm Agreement)
----------------------------------------------------------------------
MULTI-MODEL PROBABILITY BREAKDOWN:
* CLIP ViT-L/14 Semantic Stream: 100.0% Fake
* ResNet-18 Texture Stream:      71.8% Fake
* U-Net Anomaly Residual Score:  93.6%
* Detected Anomalous Regions:    Eye / Periorbital Region, Mouth / Lip Boundary, Nose / Mid-Face Region
* Boundary Seam Detected:        NO
* Auto-Face RoI Extracted:       YES (Bounding Box: (0, 0, 256, 256))
* Calibrated Threshold:          0.467 (T = 0.6433)
----------------------------------------------------------------------
Ensemble Meter:          |#################################--|
                                          ^ [Threshold]
======================================================================
[OK] Saved visual heatmap overlay to: output_heatmap.jpg
```

---

### Option C: Python API Integration
You can import the predictor directly into your own scripts or notebooks:

```python
from src.ensemble import HybridForensicPredictor

# Initialize all 3 forensic streams on CUDA (or CPU fallback)
predictor = HybridForensicPredictor()

# Run inference
result = predictor.predict(
    image_input="demo_images/sample_deepfake.jpg",
    generate_heatmap=True,
    auto_crop_face=True,
    threshold=0.467  # Optional custom threshold
)

print(f"Verdict: {result['verdict']}")
print(f"Confirmation: {result['confidence_pct']:.1f}%")
print(f"Consensus: {result['consensus']}")
print(f"Anomalous Regions: {result['anomalous_regions']}")

# Save the visual heatmap overlay image
if result["heatmap_image"]:
    result["heatmap_image"].save("heatmap_result.jpg")
```

---

## 📊 Benchmark Verification

| Metric | Stream 1: CLIP ViT-L/14 Alone | Stream 2: ResNet-18 Alone | Tri-Stream Hybrid System |
| :--- | :---: | :---: | :---: |
| **In-Distribution AUC (`finaltesting`)** | 0.8698 | 0.9993 | **0.9631** |
| **Unseen Out-of-Domain AUC (`ciplab`)** | 0.8875 | 0.6608 *(Overfit)* | **0.9011** |
| **Real Face Specificity (TNR on Real)** | **98.00%** | 70.60% | **98.00%** |
| **Overall Accuracy** | 80.00% | 61.10% | **87.58%** |
| **Inference Latency (RTX 3050)** | ~85 ms | ~15 ms | **~128 ms** |

* **Zero Metadata Shortcut Risk:** Audit classifier AUC is **0.5054** (equivalent to random chance), proving detection is strictly driven by authentic facial representations rather than container shortcuts.

---

## 🧠 Embedding Specifications
For detailed documentation on the **768-D** CLIP latent vectors, memory footprints (3 KB per face in FP32), and dataset storage details, see:
📄 **[`EMBEDDINGS_SIZE_AND_SPECIFICATIONS.md`](EMBEDDINGS_SIZE_AND_SPECIFICATIONS.md)**
