import cv2
import numpy as np
import pandas as pd


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
