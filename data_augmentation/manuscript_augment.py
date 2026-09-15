#!/usr/bin/env python3
"""
manuscript_augment.py
======================

Data augmentation pipeline for cropped handwritten manuscript line images
(e.g. line-segmented images used for Handwritten Text Recognition / HTR).

Each augmentation algorithm is implemented as its own function and saved
into its own output folder, so the final structure looks like:

    output/
        rotation/
            manuscript1.png
        shear/
            manuscript1.png
        scaling/
            manuscript1.png
        translation/
            manuscript1.png
        elastic_distortion/
            manuscript1.png
        perspective/
            manuscript1.png
        brightness/
            manuscript1.png
        contrast/
            manuscript1.png
        gaussian_noise/
            manuscript1.png
        gaussian_blur/
            manuscript1.png
        binarization/
            manuscript1.png
        bleed_through/
            manuscript1.png
        salt_pepper/
            manuscript1.png
        dilation/
            manuscript1.png
        erosion/
            manuscript1.png
        stain/
            manuscript1.png
        cutout/
            manuscript1.png
        ink_color_jitter/
            manuscript1.png
        crop_pad_boundary/
            manuscript1.png

Usage
-----
    python manuscript_augment.py --input path/to/line.png --output output/
    python manuscript_augment.py --input path/to/folder_of_lines/ --output output/

Requirements
------------
    numpy, opencv-python (or opencv-python-headless), Pillow, scipy
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# --------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------

def load_image(path):
    """Load image as BGR (OpenCV) with an alpha-safe fallback to white bg."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:  # grayscale -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:  # has alpha -> flatten onto white
        bgr, alpha = img[:, :, :3], img[:, :, 3] / 255.0
        white = np.ones_like(bgr) * 255
        img = (bgr * alpha[..., None] + white * (1 - alpha[..., None])).astype(np.uint8)
    return img


def save_image(img, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def border_value_from_image(img):
    """Estimate background color (assume manuscript background is light)."""
    corners = np.concatenate([
        img[0, :, :].reshape(-1, 3),
        img[-1, :, :].reshape(-1, 3),
        img[:, 0, :].reshape(-1, 3),
        img[:, -1, :].reshape(-1, 3),
    ])
    return tuple(int(v) for v in np.median(corners, axis=0))


# --------------------------------------------------------------------------
# 1. Geometric transformations
# --------------------------------------------------------------------------

def augment_rotation(img, angle_range=(-3, 3)):
    """Rotate the line image by a small random angle.
    Simulates skewed scanning or slightly tilted line segmentation."""
    angle = random.uniform(*angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    bg = border_value_from_image(img)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def augment_shear(img, shear_range=(-0.2, 0.2)):
    """Apply horizontal shear to simulate handwriting slant variability."""
    shear = random.uniform(*shear_range)
    h, w = img.shape[:2]
    M = np.array([[1, shear, -shear * h / 2 if shear > 0 else 0],
                  [0, 1, 0]], dtype=np.float32)
    new_w = w + int(abs(shear) * h)
    bg = border_value_from_image(img)
    return cv2.warpAffine(img, M, (new_w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def augment_scaling(img, scale_range=(0.9, 1.1)):
    """Resize the image slightly to simulate variation in pen size / hand size."""
    sx = random.uniform(*scale_range)
    sy = random.uniform(*scale_range)
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(w * sx)), max(1, int(h * sy))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def augment_translation(img, max_frac=0.05):
    """Shift image content within a padded canvas.
    Makes the model robust to imperfect line-cropping / segmentation."""
    h, w = img.shape[:2]
    tx = int(random.uniform(-max_frac, max_frac) * w)
    ty = int(random.uniform(-max_frac, max_frac) * h)
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    bg = border_value_from_image(img)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def augment_elastic_distortion(img, alpha=34, sigma=4):
    """Elastic deformation (Simard et al.) - locally warps strokes to
    simulate natural handwriting variability. A staple of HTR augmentation."""
    h, w = img.shape[:2]
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    bg = border_value_from_image(img)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def augment_perspective(img, jitter_frac=0.05):
    """Slight perspective warp to simulate camera-angle distortion or
    curved manuscript pages photographed from a book."""
    h, w = img.shape[:2]
    j = jitter_frac
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [random.uniform(0, j) * w, random.uniform(0, j) * h],
        [w - random.uniform(0, j) * w, random.uniform(0, j) * h],
        [random.uniform(0, j) * w, h - random.uniform(0, j) * h],
        [w - random.uniform(0, j) * w, h - random.uniform(0, j) * h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    bg = border_value_from_image(img)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


# --------------------------------------------------------------------------
# 2. Photometric / intensity transformations
# --------------------------------------------------------------------------

def augment_brightness(img, delta_range=(-40, 40)):
    """Randomly brighten/darken the image (different lighting/scan settings)."""
    delta = random.uniform(*delta_range)
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def augment_contrast(img, factor_range=(0.7, 1.3)):
    """Increase/decrease contrast to simulate faded ink or paper contrast."""
    factor = random.uniform(*factor_range)
    mean = img.mean()
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def augment_gaussian_noise(img, sigma=12):
    """Add Gaussian pixel noise to simulate scanner/camera sensor noise."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_gaussian_blur(img, ksize_range=(3, 5)):
    """Slight blur to simulate out-of-focus scans or low-resolution manuscripts."""
    k = random.choice(range(ksize_range[0], ksize_range[1] + 1, 2))
    return cv2.GaussianBlur(img, (k, k), 0)


def augment_binarization(img):
    """Apply adaptive/Otsu thresholding to simulate different
    preprocessing pipelines commonly used on manuscripts."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = random.choice(["otsu", "adaptive_mean", "adaptive_gaussian"])
    if method == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_mean":
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 25, 10)
    else:
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 25, 10)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def augment_salt_pepper(img, amount=0.01):
    """Randomly flip pixels to black/white (dust, ink specks, scan artifacts)."""
    out = img.copy()
    h, w = img.shape[:2]
    n_salt = int(amount * h * w * 0.5)
    n_pepper = int(amount * h * w * 0.5)

    ys = np.random.randint(0, h, n_salt)
    xs = np.random.randint(0, w, n_salt)
    out[ys, xs] = 255

    ys = np.random.randint(0, h, n_pepper)
    xs = np.random.randint(0, w, n_pepper)
    out[ys, xs] = 0
    return out


# --------------------------------------------------------------------------
# 3. Morphological transformations
# --------------------------------------------------------------------------

def augment_dilation(img, ksize=2):
    """Thicken strokes slightly - simulates heavier pen pressure/thicker ink."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # assume dark strokes on light bg -> invert so strokes are foreground (white)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    out = np.full_like(img, 255)
    out[mask > 0] = (0, 0, 0)
    return out


def augment_erosion(img, ksize=2):
    """Thin strokes slightly - simulates lighter pen pressure / faded ink."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    out = np.full_like(img, 255)
    out[mask > 0] = (0, 0, 0)
    return out


# --------------------------------------------------------------------------
# 4. Manuscript / document-specific augmentations
# --------------------------------------------------------------------------

def augment_bleed_through(img, alpha=0.15):
    """Overlay a faint horizontally-mirrored copy of the text to simulate
    ink bleeding through from the reverse side of the manuscript page."""
    mirrored = cv2.flip(img, 1)
    out = cv2.addWeighted(img, 1 - alpha, mirrored, alpha, 0)
    return out


def augment_stain(img, n_stains_range=(1, 3), max_radius_frac=0.15):
    """Add synthetic stains / degraded blotches typical of aged manuscripts."""
    out = img.copy()
    h, w = img.shape[:2]
    n = random.randint(*n_stains_range)
    overlay = out.copy()
    for _ in range(n):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        r = int(random.uniform(0.05, max_radius_frac) * min(h, w))
        color = tuple(int(c) for c in np.random.uniform(150, 210, 3))
        cv2.circle(overlay, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)
    alpha = random.uniform(0.2, 0.45)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def augment_crop_pad_boundary(img, max_px=4):
    """Randomly trim/pad a few pixels off the top and bottom edges to
    simulate imperfect line-segmentation boundaries."""
    h, w = img.shape[:2]
    bg = border_value_from_image(img)
    top = random.randint(0, max_px)
    bottom = random.randint(0, max_px)
    cropped = img[top:h - bottom if bottom > 0 else h, :]
    pad_top = random.randint(0, max_px)
    pad_bottom = random.randint(0, max_px)
    out = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, 0, 0,
                              cv2.BORDER_CONSTANT, value=bg)
    return out


def augment_ink_color_jitter(img, hue_shift=8, sat_shift=15):
    """Slightly shift ink tone/hue (e.g. iron-gall brown vs carbon black)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = np.clip(hsv[..., 0] + random.randint(-hue_shift, hue_shift), 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-sat_shift, sat_shift), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# --------------------------------------------------------------------------
# 5. Advanced / regularization-style augmentations
# --------------------------------------------------------------------------

def augment_cutout(img, n_holes=2, size_frac=0.12):
    """Randomly mask small rectangular regions (Cutout) so the model
    does not over-rely on any single stroke/region."""
    out = img.copy()
    h, w = img.shape[:2]
    bg = border_value_from_image(img)
    for _ in range(n_holes):
        hw, hh = int(size_frac * w), int(size_frac * h)
        x = random.randint(0, max(0, w - hw))
        y = random.randint(0, max(0, h - hh))
        out[y:y + hh, x:x + hw] = bg
    return out


# --------------------------------------------------------------------------
# Registry: name -> function  (folder name = key)
# --------------------------------------------------------------------------

AUGMENTATIONS = {
    "rotation": augment_rotation,
    "shear": augment_shear,
    "scaling": augment_scaling,
    "translation": augment_translation,
    "elastic_distortion": augment_elastic_distortion,
    "perspective": augment_perspective,
    "brightness": augment_brightness,
    "contrast": augment_contrast,
    "gaussian_noise": augment_gaussian_noise,
    "gaussian_blur": augment_gaussian_blur,
    "binarization": augment_binarization,
    "salt_pepper": augment_salt_pepper,
    "dilation": augment_dilation,
    "erosion": augment_erosion,
    "bleed_through": augment_bleed_through,
    "stain": augment_stain,
    "crop_pad_boundary": augment_crop_pad_boundary,
    "ink_color_jitter": augment_ink_color_jitter,
    "cutout": augment_cutout,
}


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def process_image(img_path, output_root, n_variants=1, seed=None):
    """Run every algorithm in AUGMENTATIONS on one image and save results as:
        output_root/<algorithm_name>/<manuscript_stem>[_v{i}].png
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = load_image(img_path)
    stem = Path(img_path).stem
    output_root = Path(output_root)

    for algo_name, fn in AUGMENTATIONS.items():
        algo_dir = output_root / algo_name
        for v in range(n_variants):
            try:
                result = fn(img)
            except Exception as e:
                print(f"  [!] {algo_name} failed on {img_path.name}: {e}")
                continue
            suffix = f"_v{v+1}" if n_variants > 1 else ""
            out_path = algo_dir / f"{stem}{suffix}.png"
            save_image(result, out_path)
        print(f"  [+] {algo_name}: saved to {algo_dir}/")


def gather_input_images(input_path):
    input_path = Path(input_path)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
    return [input_path]


def main():
    parser = argparse.ArgumentParser(
        description="Augment cropped manuscript line images. "
                     "Output structure: output/<algorithm>/<manuscript_name>.png"
    )
    parser.add_argument("--input", "-i", required=True,
                         help="Path to a single line image OR a folder of images.")
    parser.add_argument("--output", "-o", default="output",
                         help="Root output folder (default: output/).")
    parser.add_argument("--variants", "-n", type=int, default=1,
                         help="Number of augmented variants per algorithm per image.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility.")
    args = parser.parse_args()

    images = gather_input_images(args.input)
    if not images:
        print(f"No images found at {args.input}")
        return

    print(f"Found {len(images)} image(s). Applying {len(AUGMENTATIONS)} augmentation "
          f"algorithms -> {args.output}/")

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        process_image(img_path, args.output, n_variants=args.variants, seed=args.seed)

    print("\nDone. Folder structure created under:", Path(args.output).resolve())


if __name__ == "__main__":
    main()