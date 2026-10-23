# 🧠 Embedding Dimensions, Sizes & Latent Memory Specifications

Comprehensive technical breakdown of all embedding sizes, latent vector dimensions, memory footprints, disk storage costs, and multi-stream representation specs for the **Hybrid Multi-Scale Forensic Deepfake Detection Platform**.

---

## 1. Executive Summary Table

| Metric | Stream 1: CLIP ViT-L/14 (Semantic) | Stream 2: ResNet-18 (Texture) | Stream 3: Forensic U-Net (Spatial) |
| :--- | :---: | :---: | :---: |
| **Latent Vector Shape** | `(768,)` 1D Vector | `(512,)` 1D Vector | `(1024, 16, 16)` 3D Tensor |
| **Total Numerical Elements** | **768** floats | **512** floats | **262,144** floats |
| **Size per Image (FP32)** | **3,072 bytes (3.0 KB)** | **2,048 bytes (2.0 KB)** | **1,048,576 bytes (1.0 MB)** |
| **Size per Image (FP16 / AMP)**| **1,536 bytes (1.5 KB)** | **1,024 bytes (1.0 KB)** | **524,288 bytes (512 KB)** |
| **Size per Image (INT8 Quantized)**| **768 bytes (0.75 KB)** | **512 bytes (0.5 KB)** | **262,144 bytes (256 KB)** |
| **Normalization** | Unit Hypersphere ($L_2 = 1.0$) | Softmax / Linear Scaled | Tanh Activated $[-1, 1]$ |
| **Classification Head Size** | **3.08 KB** (768 weights + 1 bias) | **4.10 KB** (1024 weights + 2 bias) | Convolutional Decoder |

---

## 2. Stream 1: CLIP ViT-L/14 Foundation Embeddings (Primary Anchor)

### A. Architectural Origin & Projection
* **Model Backbone**: OpenAI CLIP ViT-L/14 (Vision Transformer with 24 transformer layers, 16 self-attention heads).
* **Token Dimensions**: Input face canvas ($224 \times 224$ pixels) is divided into $14 \times 14$ pixel patches, yielding a grid of $16 \times 16 = 256$ spatial tokens.
* **Internal Representation**: Each token is encoded in a **1,024-dimensional** hidden space.
* **Projection Layer**: The final `[CLS]` class token is projected via a linear visual projection matrix $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{1024 \times 768}$ to produce the final **768-dimensional** output embedding:
  $$\mathbf{z}_{\text{raw}} = f_{\text{visual}}(\mathbf{x}) \in \mathbb{R}^{768}$$
* **Unit Hypersphere Normalization**:
  $$\mathbf{z} = \frac{\mathbf{z}_{\text{raw}}}{\|\mathbf{z}_{\text{raw}}\|_2} \in \mathbb{R}^{768}, \quad \|\mathbf{z}\|_2 = 1.0$$

### B. Size per Image Across Numeric Precisions

| Precision Type | Bytes per Value | Calculation | Memory Footprint per Image |
| :--- | :---: | :---: | :---: |
| **Float32 (Single Precision, FP32)** | 4 bytes | $768 \times 4$ bytes | **3,072 bytes (3.00 KB)** |
| **Float16 (Half Precision, FP16)** | 2 bytes | $768 \times 2$ bytes | **1,536 bytes (1.50 KB)** |
| **BFloat16 (Brain Floating Point, BF16)** | 2 bytes | $768 \times 2$ bytes | **1,536 bytes (1.50 KB)** |
| **Int8 (Quantized Integer)** | 1 byte | $768 \times 1$ byte | **768 bytes (0.75 KB)** |
| **Binary 1-Bit Hash (Sign Quantization)** | 1/8 byte (1 bit) | $768 / 8$ bytes | **96 bytes** |

---

## 3. Batch Scaling Footprint (In-Memory RAM / VRAM)

When processing batches or caching large pools of extracted facial embeddings:

| Batch Size / Images Cached | Float32 (FP32) | Float16 (FP16 / AMP) | Int8 Quantized |
| :---: | :---: | :---: | :---: |
| **1 Image** | 3.0 KB | 1.5 KB | 768 B |
| **16 Images (Small Batch)** | 48.0 KB | 24.0 KB | 12.0 KB |
| **32 Images (Standard Batch)** | 96.0 KB | 48.0 KB | 24.0 KB |
| **64 Images** | 192.0 KB | 96.0 KB | 48.0 KB |
| **1,000 Images** | 3.00 MB | 1.50 MB | 768 KB |
| **10,000 Images** | 30.0 MB | 15.0 MB | 7.68 MB |
| **100,000 Images** | 300.0 MB | 150.0 MB | 76.8 MB |
| **1,000,000 Images (1M Faces)** | **2.93 GB** | **1.46 GB** | **750 MB** |

> 💡 **Forensic Scaling Benefit**: Because the embeddings are compact (3 KB per face in FP32), an entire production database of **1 million facial embeddings** easily fits into less than **3 GB of RAM** for sub-millisecond vector similarity search.

---

## 4. Stored Dataset Embeddings on Disk (`features/*.npz`)

In this project, feature vectors are pre-extracted across 6 diverse generator distributions to allow instantaneous linear probe training, calibration, and Leave-One-Source-Out (LOSO) cross-validation:

| Pre-Extracted Feature File | Sample Count | Tensor Array Shape | Stored File Size on Disk | Contents Included |
| :--- | :---: | :---: | :---: | :--- |
| [`features/ciplab.npz`](file:///c:/Users/DHASHWIN%20K%20J/OneDrive/Desktop/image_dd/features/ciplab.npz) | 1,960 | `(1960, 768)` | **5.29 MB** | `features`, `labels`, `group_ids`, `paths`, `sources` |
| [`features/ds1.npz`](file:///c:/Users/DHASHWIN%20K%20J/OneDrive/Desktop/image_dd/features/ds1.npz) | 2,000 | `(2000, 768)` | **5.40 MB** | `features`, `labels`, `group_ids`, `paths`, `sources` |
| [`features/ds2.npz`](file:///c:/Users/DHASHWIN%20K%20J/OneDrive/Desktop/image_dd/features/ds2.npz) | 2,000 | `(2000, 768)` | **5.41 MB** | `features`, `labels`, `group_ids`, `paths`, `sources` |
| [`features/ds3.npz`](file:///c:/Users/DHASHWIN%20K%20J/OneDrive/Desktop/image_dd/features/ds3.npz) | 2,000 | `(2000, 768)` | **5.40 MB** | `features`, `labels`, `group_ids`, `paths`, `sources` |
| [`features/ds4.npz`](file:///c:/Users/DHASHWIN%20K%20J/OneDrive/Desktop/image_dd/features/ds4.npz) | 2,000 | `(2000, 768)` | **5.40 MB** | `features`, `labels`, `group_ids`, `paths`, `sources` |
| [`features/original_inswapper.npz`](file:///c:/Users/DHASHWIN%20K%20J/OneDrive/Desktop/image_dd/features/original_inswapper.npz) | 2,000 | `(2000, 768)` | **5.40 MB** | `features`, `labels`, `group_ids`, `paths`, `sources` |
| **Total Across All 6 Sources** | **11,960** | **`(11960, 768)`** | **32.30 MB** | Complete 6-source generator benchmark pool |

---

## 5. Downstream Probe Head Size (`checkpoints/full_detector_package.pth`)

Because the visual backbone produces a fixed 768-D representation, the classification head trained on top of it is exceptionally lightweight:

* **Weight Matrix**: $\mathbf{w} \in \mathbb{R}^{768 \times 1} \implies 768$ float32 numbers ($768 \times 4\text{ bytes} = 3,072\text{ bytes}$).
* **Bias Term**: $b \in \mathbb{R}^1 \implies 1$ float32 number ($4\text{ bytes}$).
* **Total Parameters**: **769 parameters** ($3,076\text{ bytes} \approx 3.01\text{ KB}$).
* **Stored State Dict File**: [`checkpoints/full_detector_package.pth`](file:///c:/Users/DHASHWIN%20K%20J/OneDrive/Desktop/image_dd/hybrid_deepfake_detector_package/checkpoints/full_detector_package.pth) is only **5.33 KB** (includes PyTorch metadata, threshold $\theta = 0.467$, and temperature $T = 0.6433$).

---

## 6. Comparison with Streams 2 & 3

### Stream 2: ResNet-18 Micro-Texture Representation
* **Output Stage**: Global Average Pooling layer (`AdaptiveAvgPool2D(output_size=(1, 1))`).
* **Feature Dimension**: **512 dimensions** (`(512,)` vector).
* **Memory Size**:
  $$512 \times 4\text{ bytes} = \mathbf{2,048\text{ bytes}} = \mathbf{2.0\text{ KB}}$$
* **Classification Head**: $\mathbf{W}_{\text{fc}} \in \mathbb{R}^{2 \times 512} + \mathbf{b} \in \mathbb{R}^2 = 1,026$ parameters ($4.1\text{ KB}$).
* **Full Model Weights (`best_resnet18_universal.pth`)**: **44.8 MB** (11.17M total parameters).

### Stream 3: Forensic U-Net Spatial Bottleneck Representation
* **Bottleneck Layer**: 4th contracting block before decoder upsampling.
* **Feature Dimension**: **1024 channels $\times 16 \times 16$ spatial resolution**.
* **Total Elements**: $1,024 \times 16 \times 16 = \mathbf{262,144}$ floating point numbers.
* **Memory Size**:
  $$262,144 \times 4\text{ bytes} = \mathbf{1,048,576\text{ bytes}} = \mathbf{1.0\text{ MB}}$$
* **Full Model Weights (`checkpoints/forensic_unet.pth`)**: **69.2 MB** (17.27M total parameters).
