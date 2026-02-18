import logging
from collections import defaultdict
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.flop_counter import FlopCounterMode

from src.utils import CompileParams

log = logging.getLogger(__name__)


class AlgorithmBase(pl.LightningModule):
    
    def __init__(
        self,
        compiled: CompileParams,
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        self._compile = compiled
        self._optimizer_conf = optimizer_conf
        self._scheduler_conf = scheduler_conf
        
        if network is None:
            self._network: nn.Module = hydra.utils.instantiate(
                OmegaConf.create(network_conf)
            )
        else:
            self._network: nn.Module = network
        
        self._gpu = gpu
        if self._gpu is not None:
            self._network.cuda(gpu)
        
        self.training_step_outputs = defaultdict(list)
        self.validation_step_outputs = defaultdict(lambda: defaultdict(list))
        self.test_step_outputs = defaultdict(lambda: defaultdict(list))
        
        self.__num_flop = None
        self.__first_step = True
        
        self.__flop_counter = FlopCounterMode(display=False, depth=1)
        self.__start = torch.cuda.Event(enable_timing=True)
        self.__end = torch.cuda.Event(enable_timing=True)
    
    @property
    def network(self) -> nn.Module:
        return self._network
    
    def forward(self, *args, **kwargs):
        outputs = self._network(*args, **kwargs)
        return outputs
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            OmegaConf.create(self._optimizer_conf),
            params=filter(lambda p: p.requires_grad, self.parameters()),
        )
        
        ret_opt = {"optimizer": optimizer}
        if self._scheduler_conf is not None:
            scheduler_conf = OmegaConf.create(self._scheduler_conf)
            # Get monitor if exists, else None
            monitor = scheduler_conf.get("monitor", None)
            if "monitor" in scheduler_conf:
                del scheduler_conf.monitor
            
            scheduler: LRScheduler = hydra.utils.instantiate(
                scheduler_conf, optimizer=optimizer
            )
            sch_opt = {"scheduler": scheduler}
            
            if monitor:
                sch_opt["monitor"] = monitor
            
            ret_opt.update({"lr_scheduler": sch_opt})
        
        return ret_opt
    
    def pred(self, batch):
        raise NotImplementedError
    
    def _step(self, batch, *args, **kwargs):
        raise NotImplementedError
    
    def __step(self, batch, split_name):
        if self.__first_step:
            with self.__flop_counter:
                self.__start.record()
                output = self._step(batch)
                self.__end.record()
                torch.cuda.synchronize()
            
            if self.__num_flop is None:
                self.__num_flop = self.__flop_counter.get_total_flops()
            
            if not self._compile.disable:
                log.info("Compiling the model.")
                self._network = torch.compile(
                    self._network,
                    fullgraph=self._compile.fullgraph,
                    dynamic=self._compile.dynamic,
                    backend=self._compile.backend,
                    mode=self._compile.mode,
                    options=self._compile.options,
                    disable=self._compile.disable,
                )
            
            self.__first_step = False
        else:
            self.__start.record()
            output = self._step(batch)
            self.__end.record()
            torch.cuda.synchronize()
        
        # In test step we get 0 FLOP, so we use the previous known value
        
        flops = self.__num_flop / (self.__start.elapsed_time(self.__end) / 1000)
        output["flops"] = flops
        progress_bar_dict = dict(flops=flops)
        
        progress_bar_dict["loss"] = output["loss"].item()
        self.trainer.progress_bar_metrics.update(progress_bar_dict)
        return output
    
    @staticmethod
    def convert_to_numpy(output_dict: dict[str, torch.Tensor | Any]):
        for key in output_dict:
            if isinstance(output_dict[key], torch.Tensor):
                output_dict[key] = output_dict[key].detach().cpu()
        
        return output_dict
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        for key, value in outputs.items():
            self.training_step_outputs[key].append(value)
    
    def on_validation_batch_end(
        self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        for key, value in outputs.items():
            self.validation_step_outputs[dataloader_idx][key].append(value)
    
    def on_test_batch_end(
        self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        for key, value in outputs.items():
            self.test_step_outputs[dataloader_idx][key].append(value)
    
    def training_step(self, batch, *args, **kwargs):
        output = self.__step(batch, split_name="train")
        return output
    
    def validation_step(self, batch, *args, **kwargs):
        output = self.__step(batch, split_name="val")
        return output
    
    def test_step(self, batch, *args, **kwargs):
        output = self.__step(batch, split_name="test")
        return output
    
    def _epoch_end(self, outputs: dict[str, list], split_name):
        epoch_metrics = self._calculate_epoch_metrics(outputs)
        epoch_metrics = {f"{split_name}_{k}": v for k, v in epoch_metrics.items()}
        for checkpoint in self.trainer.checkpoint_callbacks:
            if checkpoint.monitor in epoch_metrics:
                epoch_metrics[checkpoint.monitor] = torch.Tensor(
                    epoch_metrics[checkpoint.monitor]
                )
        
        self.trainer.callback_metrics.update(epoch_metrics)
        if self.logger:
            self.logger.log_metrics(epoch_metrics, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics}\n""")
    
    def on_train_epoch_end(self) -> None:
        outputs = self.training_step_outputs
        num_dataloaders = 1
        if isinstance(self.trainer.val_dataloaders, (list, tuple)):
            num_dataloaders = max(len(self.trainer.val_dataloaders), 1)
        for i in range(num_dataloaders):
            self._epoch_end(outputs, split_name=f"train_{i}")
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        for validation_num in self.validation_step_outputs:
            outputs = self.validation_step_outputs[validation_num]
            self._epoch_end(outputs, split_name=f"val_{validation_num}")
        
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self) -> None:
        for test_num in self.test_step_outputs:
            outputs = self.test_step_outputs[test_num]
            self._epoch_end(outputs, split_name=f"test_{test_num}")
        
        self.test_step_outputs.clear()
    
    def _calculate_epoch_metrics(self, outputs: dict[str, list]) -> dict:
        epoch_metrics_sep = {}
        
        # add all output values to combined_group_metrics
        for metric_name, metric_values in outputs.items():
            epoch_metrics_sep[metric_name] = torch.tensor(
                sum(metric_values) / len(metric_values)
            )
        
        epoch_metrics_shared = {
            "learning_rate": torch.tensor(
                self.trainer.optimizers[0].param_groups[0]["lr"]
            )
        }
        
        if self.logger:
            self.logger.log_metrics(epoch_metrics_shared, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics_shared}\n""")
        
        return epoch_metrics_sep
