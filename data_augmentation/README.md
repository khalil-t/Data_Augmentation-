# Manuscript Line Image Augmentation

`manuscript_augment.py` applies a set of data augmentation techniques to
cropped handwritten manuscript **line images** (e.g. line-segmented images
used for Handwritten Text Recognition / HTR).

Each method is implemented as an independent function and results are saved
in their own folder:

```
output/
  rotation/
  shear/
  scaling/
  translation/
  elastic_distortion/
  perspective/
  brightness/
  contrast/
  gaussian_noise/
  gaussian_blur/
  binarization/
  salt_pepper/
  dilation/
  erosion/
  bleed_through/
  stain/
  crop_pad_boundary/
  ink_color_jitter/
  cutout/
```

## Usage

```bash
pip install opencv-python numpy pillow scipy

# single image
python manuscript_augment.py --input line.png --output output/

# folder of line-cropped images
python manuscript_augment.py --input lines_folder/ --output output/

# multiple randomized variants per method
python manuscript_augment.py --input lines_folder/ --output output/ --variants 3
```

## Methods

### 1. Geometric Transformations

- **Rotation** — Rotates the image by a small random angle (typically ±2°–±5°,
  since text lines are horizontal and large rotations destroy readability).
  Simulates skewed scanning or inconsistent line cropping.

- **Shearing (Slant)** — Applies horizontal shear to simulate different
  handwriting slant styles. Very relevant for manuscripts since slant varies
  a lot between writers.

- **Scaling** — Resizes the image slightly (e.g. 90%–110%) to simulate
  variation in pen size, writer hand size, or camera/scanner distance.

- **Translation** — Shifts the image content within the canvas (with
  padding) to make the model robust to imperfect line-cropping and
  segmentation.

- **Elastic Distortion** — Applies a random elastic deformation field
  (Gaussian-smoothed random displacement) to locally warp strokes,
  simulating natural handwriting variability. A classic augmentation in HTR
  pipelines (originally popularized by Simard et al.).

- **Perspective / Affine Warp** — Simulates slight perspective distortion
  from a camera angle or a curved manuscript page, common with photographed
  old manuscripts or books.

### 2. Photometric / Intensity Transformations

- **Brightness Adjustment** — Randomly brightens or darkens the image to
  simulate different lighting or scanning conditions.

- **Contrast Adjustment** — Increases or decreases contrast to simulate
  faded ink or varying paper contrast.

- **Gaussian Noise Addition** — Adds random pixel noise to simulate scanner
  or camera sensor noise and paper texture artifacts.

- **Gaussian Blur** — Slightly blurs the image to simulate out-of-focus
  scans or low-resolution manuscripts.

- **Binarization Variation** — Applies different thresholding algorithms
  (Otsu, adaptive mean, adaptive Gaussian) to simulate different
  preprocessing pipelines, since manuscripts are often binarized
  inconsistently.

- **Salt-and-Pepper Noise** — Randomly sets isolated pixels to black or
  white, simulating dust, ink specks, or scanning artifacts.

### 3. Morphological Transformations

- **Dilation** — Thickens strokes slightly, simulating heavier pen pressure
  or thicker ink.

- **Erosion** — Thins strokes slightly, simulating lighter pen pressure or
  faded ink.

  Dilation and erosion are especially useful for manuscripts since ink
  thickness varies a lot across historical documents.

### 4. Manuscript / Document-Specific Augmentations

- **Ink Bleed-Through Simulation** — Overlays a faint, horizontally mirrored
  copy of the text to simulate ink bleeding through from the reverse side of
  the page — a very manuscript-specific augmentation.

- **Stain / Degradation Simulation** — Adds synthetic stains, blotches, or
  degraded regions typical of old manuscripts, making the model robust to
  document damage.

- **Random Crop/Pad of Line Boundaries** — Since automatic line
  segmentation is imperfect, randomly trims or pads a few pixels at the
  top/bottom of the line crop to simulate segmentation errors.

- **Ink Color Jitter** — Slightly shifts the color/hue and saturation of the
  ink, simulating different ink types (e.g. iron-gall brown vs.
  carbon-based black).

### 5. Advanced / Regularization-Style Augmentations

- **Cutout (Random Erasing)** — Randomly masks small rectangular regions of
  the image, forcing the model to not over-rely on any single stroke or
  local region.

### 6. Additional Advanced Methods (not included in the script — require external models/data)

- **GAN-based Style Transfer** — Uses a GAN trained on different handwriting
  styles to transform the writing style of a line image while preserving
  text content — effectively generating synthetic writers.

- **Synthetic Handwriting Generation** — Generates entirely new handwritten
  line images from text using a generative handwriting model trained on the
  target script, augmenting the dataset with fully synthetic samples.

- **CutMix** — Combines patches from two different line images (and blends
  their labels), pushing the model to learn from partial/combined visual
  evidence. Less common in sequential text recognition than in
  classification tasks.

- **Mixup** — Linearly blends two line images and their labels. More
  experimental for text recognition and generally less effective than for
  classification tasks, but used in some HTR pipelines.

- **Character/Word Spacing Jitter** — If word/character bounding boxes are
  available, randomly adjusts the spacing between them to simulate
  different writer spacing habits.

## Recommended Libraries

| Library | Notes |
|---|---|
| [Albumentations](https://albumentations.ai/) | General-purpose, fast; supports most geometric/photometric ops above. |
| [imgaug](https://github.com/aleju/imgaug) | Popular in HTR research; built-in elastic transform, dropout, etc. |
| [OCRodeg](https://github.com/NVlabs/ocrodeg) | Designed specifically for OCR/HTR degradation (bleed-through, blur, warping). |
| [Augraphy](https://github.com/sparkfish/augraphy) | Document-image-specific augmentation (ink bleed, folding, stains, scanner noise). |

## Practical Note

For handwritten manuscript lines, the augmentations with the best track
record in HTR literature (e.g. PyLaia, Kraken, Transformer-based HTR
systems) are **elastic distortion, slant/shear, dilation/erosion, random
noise/blur, and bleed-through simulation** — combined moderately so
legibility is preserved.