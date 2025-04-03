import cv2
import numpy as np
import pandas as pd
import torch
from skimage.draw import line
from skimage.morphology import skeletonize


def unpatch(tensor: torch.Tensor, h: int, w: int, channels: int, patch_size: int) -> torch.Tensor:
    # the next line of code was thoroughly thought and tested, never to be touched again
    return tensor.permute(0, 2, 3, 1).reshape(
        -1, h, w, channels, patch_size, patch_size
    ).permute(
        0, 3, 1, 4, 2, 5
    ).reshape(
        -1, channels, patch_size * h, patch_size * w
    )


def pad_to_square(img: np.ndarray, fill_value=-1, size=None) -> np.ndarray:
    """Pads an image to make it square."""
    if size is not None and max(img.shape) <= size:
        pic_size = size
    else:
        pic_size = max(img.shape)
    
    pad_h = (pic_size - img.shape[0]) // 2
    pad_w = (pic_size - img.shape[1]) // 2
    
    padded_img = np.full(
        (pic_size, pic_size, *img.shape[2:]), fill_value=fill_value, dtype=img.dtype
    )
    padded_img[pad_h:pad_h + img.shape[0], pad_w:pad_w + img.shape[1]] = img
    return padded_img


def draw_radiation_pattern(radiation_pattern_csv_path, input_img, azimuth):
    df = pd.read_csv(radiation_pattern_csv_path, header=None)
    height, width = input_img.shape[:2]
    min_intensity_y = np.argmin(input_img[..., 2]) // input_img[..., 2].shape[1]
    min_intensity_x = np.argmin(input_img[..., 2]) % input_img[..., 2].shape[1]
    antenna_location = [min_intensity_x, min_intensity_y]  # Convert to list for uniformity
    values = df[0]
    rp_img = np.zeros((height, width), dtype=float)
    length = (input_img.shape[0] ** 2 + input_img.shape[1] ** 2) ** 0.5
    for i in range(360):
        angle_1_rad = np.radians(i - 0.5 + azimuth)
        angle_2_rad = np.radians(i + 0.5 + azimuth)
        x1 = int(antenna_location[0] + length * np.cos(angle_1_rad))
        y1 = int(antenna_location[1] - length * np.sin(angle_1_rad))
        x2 = int(antenna_location[0] + length * np.cos(angle_2_rad))
        y2 = int(antenna_location[1] - length * np.sin(angle_2_rad))
        triangle_cnt = np.array([antenna_location, [x1, y1], [x2, y2]], dtype=np.int32)
        color = float(values.iloc[i])
        cv2.drawContours(rp_img, [triangle_cnt], 0, color, -1)
    return rp_img


def get_pl_estimation(img, freq):
    # PL_0 = compute_pathloss_clamped(img, freq_MHz=freq / 1000)
    L_o = 1
    T = img[:, :, 1]  # Transmittance
    d = img[:, :, 2]  # Distance
    
    d = np.maximum(d, 0.1)
    
    wall_map = T > 0
    thin_wall_map = skeletonize(wall_map)
    obstruction_map = thin_wall_map * T
    tx_x, tx_y = d.argmin() // d.shape[1], d.argmin() % d.shape[1]
    
    O = np.zeros_like(d, dtype=int)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            rr, cc = line(tx_x, tx_y, i, j)
            O[i, j] = np.sum(obstruction_map[rr, cc])
    
    PL = O * L_o  # + PL_0  # - rp + 10 * n * np.log10(d)
    return PL[..., np.newaxis]
