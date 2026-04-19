#!/usr/bin/env python3
"""
Sonar image denoiser — offline equivalent of the ROS SonarDenoiser node.
Applies bilateral filter, optional hard threshold, and optional center column mask.
"""
import cv2
import numpy as np


def denoise_sonar_image(
    img: np.ndarray,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 60.0,
    bilateral_sigma_space: float = 60.0,
    threshold: int = 40,
    center_mask_half_width: int = 0,
) -> np.ndarray:
    """
    Denoise a 2D sonar image (H x W, float32 or uint8).
    Returns a uint8 denoised image of the same shape.

    Args:
        img: raw sonar image
        bilateral_d: pixel neighbourhood diameter for bilateral filter
        bilateral_sigma_color: range sigma (higher = blends more intensity levels)
        bilateral_sigma_space: spatial sigma (higher = blends farther pixels)
        threshold: zero out pixels below this value (0 = disabled)
        center_mask_half_width: mask this many pixels either side of centre column (0 = disabled)
    """
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    result = cv2.bilateralFilter(img_u8, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    if threshold > 0:
        result = np.where(result > threshold, result, 0).astype(np.uint8)

    if center_mask_half_width > 0:
        center = result.shape[1] // 2
        result[:, center - center_mask_half_width: center + center_mask_half_width] = 0

    return result
