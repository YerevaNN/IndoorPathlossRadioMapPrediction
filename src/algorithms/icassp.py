import logging
from collections import defaultdict
from typing import Any, List

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from skimage.transform import resize
from sklearn.metrics import mean_squared_error

from src.algorithms.algorithm_base import AlgorithmBase
from src.utils import CompileParams, pad_to_square

log = logging.getLogger(__name__)


class ICASSP(AlgorithmBase):
    
    def __init__(
        self,
        out_norm: float,
        fixed_scale: bool,
        compiled: CompileParams,
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        *args, **kwargs
    ):
        super().__init__(
            compiled=compiled,
            optimizer_conf=optimizer_conf,
            scheduler_conf=scheduler_conf,
            network=network,
            network_conf=network_conf,
            gpu=gpu
        )
        
        self.out_norm = out_norm
        self.fixed_scale = fixed_scale
        self.training_step_outputs = []
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.mse = nn.MSELoss()
    
    def pred(self, batch):
        input_image, supervision_image, orig_out_path, mask = batch
        pred_image = self._network(input_image.unsqueeze(0).cuda(self._gpu))
        orig_out = iio.imread(orig_out_path)
        pred_image = self.get_pred_image(pred_image[0], orig_out, mask)
        return {
            "pred_image": torch.from_numpy(pred_image).unsqueeze(0).unsqueeze(0),
            "orig_out": orig_out,
        }
    
    def _step(self, batch, *args, **kwargs):
        input_image, supervision_image, orig_out_path, mask = batch
        pred_image = self._network(input_image)
        return self.get_metrics(pred_image, supervision_image, orig_out_path, mask)
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.training_step_outputs.append(outputs)
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.validation_step_outputs[dataloader_idx].append(outputs)
    
    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.test_step_outputs[dataloader_idx].append(outputs)
    
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
    
    def get_rmse(self, pred_images, orig_out_paths, masks):
        rmses = []
        # Note: these calculations are generally wrong for the training set due to augmentations
        for i, orig_out_path in enumerate(orig_out_paths):
            orig_out = iio.imread(orig_out_path)
            pred_image = self.get_pred_image(pred_images[i], orig_out, masks[i]) * self.out_norm
            rmses.append(mean_squared_error(orig_out, pred_image, squared=False))
        return np.mean(rmses)
    
    def get_metrics(self, pred_image, supervision_image, orig_out_path, mask):
        pred_image = torch.sigmoid(pred_image)
        loss = self.mse(supervision_image, pred_image)
        rmse = torch.Tensor([self.get_rmse(pred_image, orig_out_path, mask)])
        metrics = {
            "loss": loss,
            "rmse": rmse
        }
        return metrics
    
    def _calculate_epoch_metrics(self, outputs: List[Any]) -> dict:
        # init combined metrics with zero values
        combined_general_metrics = {k: 0 for k in outputs[0].keys()}
        
        # add all output values to combined_group_metrics
        for o in outputs:
            for k in o.keys():
                combined_general_metrics[k] += o[k]
        
        # compute means of metrics
        for k in outputs[0].keys():
            combined_general_metrics[k] /= len(outputs)
        
        # merge all
        epoch_metrics_sep = combined_general_metrics
        
        epoch_metrics_shared = {
            "learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"]
        }
        
        if self.logger:
            self.logger.log_metrics(epoch_metrics_shared, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics_shared}\n""")
        
        return epoch_metrics_sep
