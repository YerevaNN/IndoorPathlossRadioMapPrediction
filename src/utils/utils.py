import os
import struct
from dataclasses import dataclass
from typing import Any, Sequence

import fcntl
import numpy as np
import psutil
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import termios
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str],
    resolve: bool = True,
):
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    
    style = "green bold"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    
    rich.print(tree)
    
    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)
    
    OmegaConf.save(config, "config_tree.yaml")


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    algorithm: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """ This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally, saves:
        - number of trainable model parameters
    """
    if trainer.logger is None:
        return
    
    hparams = dict()
    
    # choose which parts of hydra config will be saved to loggers
    hparams["run_dir"] = os.getcwd()
    hparams["trainer"] = config["trainer"]
    hparams["algorithm"] = config["algorithm"]
    hparams["network"] = config["network"]
    hparams["optimizer"] = config["optimizer"]
    hparams["datamodule"] = config["datamodule"]
    hparams["ckpt_path"] = config["ckpt_path"]
    if "strategy" in config:
        hparams["strategy"] = config["strategy"]
    if "scheduler" in config:
        hparams["scheduler"] = config["scheduler"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]
    
    # save number of model parameters
    hparams["algorithm/params_total"] = sum(p.numel() for p in algorithm.parameters())
    hparams["algorithm/params_trainable"] = sum(
        p.numel() for p in algorithm.parameters() if p.requires_grad
    )
    hparams["algorithm/params_not_trainable"] = sum(
        p.numel() for p in algorithm.parameters() if not p.requires_grad
    )
    
    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
    
    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def pad_to_square(img, fill_value=-1, size=None):
    if size is not None and max(img.shape) <= size:
        pic_size = size
        pic = np.full(shape=(pic_size, pic_size, *img.shape[2:]), fill_value=fill_value, dtype=img.dtype)
        left = (size - img.shape[0]) // 2
        top = (size - img.shape[1]) // 2
        pic[left:left + img.shape[0], top:top + img.shape[1]] = img
        return pic
    else:
        pic_size = max(img.shape)
        if pic_size > img.shape[0]:
            pad_size = (pic_size - img.shape[0]) // 2
            pad = np.full(shape=(pad_size, pic_size, *img.shape[2:]), fill_value=fill_value, dtype=img.dtype)
            img = np.concatenate((pad, img, pad), axis=0)
        
        elif pic_size > img.shape[1]:
            pad_size = (pic_size - img.shape[1]) // 2
            pad = np.full(shape=(pic_size, pad_size, *img.shape[2:]), fill_value=fill_value, dtype=img.dtype)
            img = np.concatenate((pad, img, pad), axis=1)
    
    return img


class EpochCounter:
    count = 0


def set_winsize(fd, col, row, xpix=0, ypix=0):
    winsize = struct.pack("HHHH", row, col, xpix, ypix)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


class ProgressBarTheme(RichProgressBarTheme):
    # Hydra doesn't work with RichProgressBarTheme's Union type annotations
    description: str = "white"
    progress_bar: str = "#6206E0"
    progress_bar_finished: str = "#6206E0"
    progress_bar_pulse: str = "#6206E0"
    batch_progress: str = "white"
    time: str = "grey54"
    processing_speed: str = "grey70"
    metrics: str = "white"
    metrics_text_delimiter: str = " "
    metrics_format: str = ".3f"


@dataclass
class CompileParams:
    fullgraph: bool
    dynamic: bool
    backend: str
    mode: str
    options: dict[str, Any]
    disable: bool


def unpatch(tensor: torch.Tensor, h: int, w: int, channels: int, patch_size: int) -> torch.Tensor:
    """
    Args:
        tensor: tokens of shape (batch size, token size, h, w)
        h: image height / patch size
        w: image width / patch size
        channels: number of channels in the image
        patch_size: patch size

    Returns: Unpactched image of shape (batch size, number of channels, image height, image width)
    """
    # the next line of code was thoroughly thought and tested, never to be touched again
    return tensor.permute(0, 2, 3, 1).reshape(
        -1, h, w, channels, patch_size, patch_size
    ).permute(
        0, 3, 1, 4, 2, 5
    ).reshape(
        -1, channels, patch_size * h, patch_size * w
    )


def worker_initializer(cpu_list, index_counter):
    """Initializer function for pool workers."""
    # Get the process ID of the current worker
    pid = os.getpid()
    p = psutil.Process(pid)
    
    with index_counter.get_lock():
        index = index_counter.value % len(cpu_list)
        index_counter.value += 1
    
    # Assign CPU affinity based on the process index
    cpu_index = cpu_list[index]
    p.cpu_affinity([cpu_index])


class BuildingShiftDistribution:
    """
    Randomly sampling from a distribution of building shifts with a given ECDF.
    We interpolate the CDF from ECDF with linear interpolation, resulting constant interpolation in the PDF.
    """
    
    def __init__(self, ecdf: dict[float, float]):
        self.x_values = np.array(sorted(ecdf.keys()))
        self.ecdf_values = np.array(sorted(ecdf.values()))
    
    def sample(self, size=1):
        pdf_values = np.diff(self.ecdf_values)
        # the index segment of the CDF to sample from
        segment = np.random.choice(list(range(pdf_values.shape[0])), size, p=pdf_values / pdf_values.sum())
        u = np.random.random(size)
        # left and right endpoints of a segment
        left = self.x_values[segment]
        right = self.x_values[segment + 1]
        # constant interpolation
        return u * (right - left) + left
