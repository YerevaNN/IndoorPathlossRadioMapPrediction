import logging
from typing import Iterable, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.strategies import ParallelStrategy

from src.algorithms.algorithm_base import AlgorithmBase
from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.utils import EpochCounter, log_hyperparameters

log = logging.getLogger(__name__)


def train(config: DictConfig) -> None:
    epoch_counter = EpochCounter()
    gpus = config.trainer.devices
    multi_gpu = gpus == -1 or (isinstance(gpus, Iterable) and len(gpus) > 1) or (isinstance(gpus, int) and gpus > 1)
    
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: WAIRDBaseDatamodule = hydra.utils.instantiate(
        config.datamodule,
        epoch_counter=epoch_counter, multi_gpu=multi_gpu, drop_last=not config.algorithm.compiled.disable
    )
    
    log.info(f"Instantiating algorithm {config.algorithm._target_}")
    algorithm: AlgorithmBase = hydra.utils.instantiate(
        config.algorithm,
        epoch_counter=epoch_counter,
        network=None,  # instead, we give network_conf
        network_conf=(OmegaConf.to_yaml(config.network) if "network" in config else None),
        optimizer_conf=(OmegaConf.to_yaml(config.optimizer) if "optimizer" in config else None),
        scheduler_conf=(OmegaConf.to_yaml(config.scheduler) if "scheduler" in config else None)
    )
    
    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    
    # Init lightning loggers
    loggers: list[Logger] = []
    if "loggers" in config:
        for name, lg_conf in config.loggers.items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = hydra.utils.instantiate(lg_conf)
            loggers.append(logger)
    
    if "strategy" in config:
        log.info(f"Instantiating strategy <{config.strategy}>")
        strategy: Optional[ParallelStrategy] = hydra.utils.instantiate(config.strategy)
    else:
        if multi_gpu:
            log.error("In case of using multiple GPUs, you must provide a strategy")
        strategy = None
    
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, strategy=strategy or "auto", _convert_="partial"
    )
    
    log_hyperparameters(config=config, algorithm=algorithm, trainer=trainer)
    
    # Train the model
    log.info("Starting training!")
    trainer.fit(algorithm, datamodule=datamodule, ckpt_path=config.ckpt_path)
    
    trainer.test(dataloaders=datamodule.test_dataloader())
