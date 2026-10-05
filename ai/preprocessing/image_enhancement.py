#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Image Preprocessing & Enhancement Module
Implements noise reduction and contrast enhancement protocols specified in Senmozhi Work II:
1. Gaussian Denoising: Attenuates high-frequency camera sensor noise.
2. Median Filtering: Removes impulsive/salt-and-pepper noise while preserving sharp crack edges.
3. CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhances micro-crack visibility
   and biological growth boundaries under non-uniform sunlight/shade on stone surfaces.
4. LAB/YCrCb Illumination Normalization: Stabilizes luminance across ancient masonry facades.
"""

import os
import cv2
import numpy as np
from PIL import Image

def apply_gaussian_denoising(image_bgr, kernel_size=(5, 5), sigma=1.0):
    """
    Applies Gaussian blurring to attenuate high-frequency sensor grain.
    """
    return cv2.GaussianBlur(image_bgr, kernel_size, sigma)

def apply_median_denoising(image_bgr, kernel_size=3):
    """
    Applies median filtering to eliminate impulse noise while preserving sharp crack boundaries.
    """
    return cv2.medianBlur(image_bgr, kernel_size)

def apply_clahe(image_bgr, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel (LAB space)
    to boost contrast in shadows and carved relief crevices without color distortion.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def apply_unsharp_mask(image_bgr, sigma=1.0, strength=1.5):
    """
    Accentuates fine hairline fractures and subtle stone texture using unsharp masking.
    """
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def preprocess_monument_image(image_input, enable_clahe=True, enable_denoise=True, enable_sharpen=False):
    """
    Comprehensive preprocessing pipeline for monument decay inspection.
    Accepts PIL Image, file path, or numpy BGR array.
    Returns:
        enhanced_bgr: np.ndarray (BGR format for OpenCV processing)
        enhanced_pil: PIL.Image (RGB format for PyTorch / visualization)
        info_dict: dict of applied operations and quality metrics
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at: {image_input}")
        image_bgr = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        image_rgb = np.array(image_input.convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            image_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    orig_h, orig_w = image_bgr.shape[:2]
    processed_bgr = image_bgr.copy()
    ops_applied = []

    # 1. Edge-preserving noise filtering
    if enable_denoise:
        processed_bgr = apply_median_denoising(processed_bgr, kernel_size=3)
        ops_applied.append("MedianDenoise(k=3)")

    # 2. Local contrast enhancement via CLAHE
    if enable_clahe:
        processed_bgr = apply_clahe(processed_bgr, clip_limit=2.5, tile_grid_size=(8, 8))
        ops_applied.append("CLAHE(clip=2.5, grid=8x8)")

    # 3. Optional unsharp masking for crack accentuation
    if enable_sharpen:
        processed_bgr = apply_unsharp_mask(processed_bgr, sigma=1.0, strength=0.8)
        ops_applied.append("UnsharpMask(sigma=1.0, strength=0.8)")

    # Convert to RGB PIL
    rgb_arr = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    enhanced_pil = Image.fromarray(rgb_arr)

    # Compute quick quality metrics
    gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) # Sharpness metric
    mean_brightness = float(np.mean(gray))

    info_dict = {
        "original_dimensions": [orig_w, orig_h],
        "operations_applied": ops_applied,
        "sharpness_laplacian_var": round(laplacian_var, 2),
        "mean_brightness": round(mean_brightness, 2),
        "is_underexposed": mean_brightness < 45.0,
        "is_overexposed": mean_brightness > 210.0
    }

    return processed_bgr, enhanced_pil, info_dict

if __name__ == "__main__":
    import sys
    print("Testing monument image enhancement pipeline...")
    # Test on a dummy black/white patch
    test_img = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
    bgr, pil_img, info = preprocess_monument_image(test_img)
    print("Preprocessed successfully:", info)
