import logging
import os
from collections.abc import Iterable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from src.algorithms.algorithm_base import AlgorithmBase
from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.utils import EpochCounter

log = logging.getLogger(__name__)


def pred(config: DictConfig) -> None:
    epoch_counter = EpochCounter()
    log.info(f"Instantiating datamodule <{config['datamodule']['_target_']}>")
    datamodule: WAIRDBaseDatamodule = hydra.utils.instantiate(
        config["datamodule"],
        epoch_counter=epoch_counter, drop_last=False
    )
    datamodule.prepare_data()
    
    log.info(f"Instantiating algorithm {config['algorithm']['_target_']} with checkpoint {config['checkpoint_path']}")
    algorithm: AlgorithmBase = hydra.utils.get_class(config["algorithm"]["_target_"]).load_from_checkpoint(
        config["checkpoint_path"], **config["algorithm"],
        network_conf=(OmegaConf.to_yaml(config["network"]) if "network" in config else None),
        gpu=config["gpu"],
        epoch_counter=epoch_counter,
        map_location=f"cuda:{config['gpu']}"
    )
    algorithm.network.eval()
    algorithm.network.cuda(config["gpu"])
    
    pred_path = os.path.dirname(os.path.dirname(config["checkpoint_path"]))
    pred_path = os.path.join(config["prediction_dir"], os.path.basename(pred_path))
    os.makedirs(pred_path, exist_ok=True)
    
    dataset = None
    if config["split"] == "test":
        dataset = datamodule.test_set
    elif config["split"] == "val":
        dataset = datamodule.val_set
    elif config["split"] == "train":
        dataset = datamodule.train_set
    
    if isinstance(dataset, Iterable):
        for data_idx, data_set in enumerate(dataset):
            log.info(f"{data_idx=}")
            os.makedirs(os.path.join(pred_path, str(data_idx)), exist_ok=True)
            for i in tqdm(range(len(data_set)), total=len(data_set)):
                batch = data_set[i]
                alg_out = algorithm.pred(batch)
                out = alg_out.pop("pred_image")
                out = nn.functional.sigmoid(out)
                out = out.detach().cpu().numpy()[0, 0]
                np.savez(os.path.join(pred_path, str(data_idx), str(i)), out=out, **alg_out)
    else:
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            batch = dataset[i]
            alg_out = algorithm.pred(batch)
            out = alg_out.pop("pred_image")
            out = nn.functional.sigmoid(out)
            out = out.detach().cpu().numpy()[0, 0]
            np.savez(os.path.join(pred_path, str(i)), out=out, **alg_out)
