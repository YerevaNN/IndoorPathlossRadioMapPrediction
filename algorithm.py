import logging
from collections import defaultdict

import imageio.v3 as iio
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from skimage.transform import resize

from utils import pad_to_square

log = logging.getLogger(__name__)


class ICASSP(pl.LightningModule):
    
    def __init__(
        self,
        out_norm: float,
        fixed_scale: bool,
        network: nn.Module = None,
        gpu: int = None,
        *args, **kwargs
    ):
        super().__init__()
        self.out_norm = out_norm
        self.fixed_scale = fixed_scale
        self.training_step_outputs = []
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.mse = nn.MSELoss()
        self._network = network
        self._gpu = gpu
    
    @property
    def network(self) -> nn.Module:
        return self._network
    
    def pred(self, batch):
        input_image, supervision_image, orig_out_path, mask = batch
        pred_image = self._network(input_image.unsqueeze(0).cuda(self._gpu))
        orig_out = iio.imread(orig_out_path)
        pred_image = self.get_pred_image(pred_image[0], orig_out, mask)
        return {
            "pred_image": torch.from_numpy(pred_image).unsqueeze(0).unsqueeze(0)
        }
    
    def get_pred_image(
        self, pred_image: torch.Tensor, orig_out: np.ndarray, mask: torch.Tensor
    ) -> np.ndarray:
        pred_image = pred_image.squeeze(0).detach().cpu().numpy()
        mask = mask.detach().cpu().numpy().astype(bool)
        if True or max(orig_out.shape) > max(mask.shape) or self._network.training or not self.fixed_scale:
            mask = np.ones_like(orig_out)
            mask = pad_to_square(mask, fill_value=0).astype(bool)
            pred_image = resize(pred_image, mask.shape)
        pred_image = pred_image[mask].reshape(orig_out.shape)
        return pred_image
