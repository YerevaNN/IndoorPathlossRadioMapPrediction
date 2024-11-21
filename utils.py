import torch
import numpy as np


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