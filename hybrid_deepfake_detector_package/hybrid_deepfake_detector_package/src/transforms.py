"""
Phase 4 -- Transforms and Dataset
=================================
Implements:
1. Deterministic evaluation transforms (ImageNet or CLIP normalization).
2. Advanced data augmentations (JPEG recompression, down-up scaling, Gaussian noise, etc.)
   with individual toggles from config.yaml.
3. Robust PyTorch Dataset class for image loading.

Usage:
    python -m src.transforms
"""

import io
import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLIP_MEAN = [0.4815, 0.4578, 0.4082]
CLIP_STD = [0.2686, 0.2613, 0.2758]


class RandomJPEGCompression:
    """Randomly compress image as JPEG with quality in [q_min, q_max]."""

    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img


class RandomDownUpScale:
    """Randomly downscale image by scale factor, then upscale back."""

    def __init__(self, scale_range=(0.5, 1.0), p=0.3):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            w, h = img.size
            scale = random.uniform(*self.scale_range)
            new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
            down = img.resize((new_w, new_h), Image.BILINEAR)
            return down.resize((w, h), Image.BILINEAR)
        return img


class AddGaussianNoise:
    """Add zero-mean Gaussian noise to a tensor in [0, 1]."""

    def __init__(self, sigma_max=0.03, p=0.2):
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            sigma = random.uniform(0.0, self.sigma_max)
            noise = torch.randn_like(tensor) * sigma
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


def get_eval_transform(img_size: int = 224, backbone: str = "resnet") -> T.Compose:
    """
    Deterministic evaluation transform.
    Center crop / resize to img_size, ToTensor, and model-appropriate normalization.
    """
    mean = CLIP_MEAN if backbone == "clip" else IMAGENET_MEAN
    std = CLIP_STD if backbone == "clip" else IMAGENET_STD

    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_aug_config(config_path: str = "config.yaml") -> dict:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("augmentations", {})
    except Exception:
        return {}


def get_train_transform(
    img_size: int = 224,
    backbone: str = "resnet",
    aug_cfg: dict = None,
) -> T.Compose:
    """
    Training transform with forensic augmentations.
    Each augmentation can be toggled via config.
    """
    if aug_cfg is None:
        aug_cfg = load_aug_config()

    mean = CLIP_MEAN if backbone == "clip" else IMAGENET_MEAN
    std = CLIP_STD if backbone == "clip" else IMAGENET_STD

    transforms = []

    # 1. Random Resized Crop (falls back to plain Resize if disabled)
    rrc_cfg = aug_cfg.get("random_resized_crop", {})
    if rrc_cfg.get("enabled", True):
        scale = tuple(rrc_cfg.get("scale", [0.7, 1.0]))
        transforms.append(T.RandomResizedCrop(img_size, scale=scale))
    else:
        transforms.append(T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR))

    # 2. Horizontal Flip
    hf_cfg = aug_cfg.get("horizontal_flip", {})
    if hf_cfg.get("enabled", True):
        transforms.append(T.RandomHorizontalFlip(p=hf_cfg.get("p", 0.5)))

    # 3. JPEG compression
    jpeg_cfg = aug_cfg.get("jpeg_compression", {})
    if jpeg_cfg.get("enabled", True):
        quality_range = tuple(jpeg_cfg.get("quality_range", [30, 95]))
        transforms.append(RandomJPEGCompression(quality_range=quality_range, p=jpeg_cfg.get("p", 0.5)))

    # 4. Down-up scaling
    downup_cfg = aug_cfg.get("downup_scale", {})
    if downup_cfg.get("enabled", True):
        scale_range = tuple(downup_cfg.get("scale_range", [0.5, 1.0]))
        transforms.append(RandomDownUpScale(scale_range=scale_range, p=downup_cfg.get("p", 0.3)))

    # 5. Gaussian blur
    blur_cfg = aug_cfg.get("gaussian_blur", {})
    if blur_cfg.get("enabled", True):
        kernel = blur_cfg.get("kernel_size", [3, 5])
        k = kernel[0] if isinstance(kernel, list) else kernel
        if k % 2 == 0:
            k += 1
        sigma = tuple(blur_cfg.get("sigma", [0.1, 2.0]))
        transforms.append(T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=sigma)], p=blur_cfg.get("p", 0.4)))

    # 6. ColorJitter
    cj_cfg = aug_cfg.get("color_jitter", {})
    if cj_cfg.get("enabled", True):
        transforms.append(T.ColorJitter(
            brightness=cj_cfg.get("brightness", 0.1),
            contrast=cj_cfg.get("contrast", 0.1),
            saturation=cj_cfg.get("saturation", 0.1),
        ))

    # 7. Rotation
    rot_cfg = aug_cfg.get("rotation", {})
    if rot_cfg.get("enabled", True):
        transforms.append(T.RandomRotation(degrees=rot_cfg.get("degrees", 10)))

    # 8. ToTensor
    transforms.append(T.ToTensor())

    # 9. Gaussian noise
    gn_cfg = aug_cfg.get("gaussian_noise", {})
    if gn_cfg.get("enabled", True):
        transforms.append(AddGaussianNoise(
            sigma_max=gn_cfg.get("sigma_max", 0.03),
            p=gn_cfg.get("p", 0.2),
        ))

    # 10. Normalization
    transforms.append(T.Normalize(mean=mean, std=std))

    # 11. Random Erasing
    re_cfg = aug_cfg.get("random_erasing", {})
    if re_cfg.get("enabled", True):
        scale = tuple(re_cfg.get("scale", [0.02, 0.2]))
        transforms.append(T.RandomErasing(p=re_cfg.get("p", 0.3), scale=scale))

    return T.Compose(transforms)


class DeepfakeDataset(Dataset):
    """
    Standard PyTorch Dataset for loading images from index dataframe.
    """

    def __init__(self, df, transform=None):
        self.paths = df["path"].values
        self.labels = df["label"].values.astype(np.int64)
        self.sources = df["source"].values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback black image if corrupt
            img = Image.new("RGB", (256, 256), color=(0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label, self.sources[idx]


if __name__ == "__main__":
    t_eval = get_eval_transform(224, "resnet")
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    t1 = t_eval(img)
    t2 = t_eval(img)
    assert torch.allclose(t1, t2), "Eval transform must be strictly deterministic!"
    print(f"[transforms] Determinism test passed: {t1.shape}")
