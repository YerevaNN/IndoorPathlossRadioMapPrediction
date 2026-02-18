import logging
import os
import random
import sys
import warnings

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

log = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

hydra.core.global_hydra.GlobalHydra.instance().clear()


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(config: DictConfig) -> None:
    from src import utils, train, pred, evaluate
    
    warnings.filterwarnings("ignore", ".*beta state*")
    
    terminal_col = config.get("terminal_col")
    if terminal_col:
        terminal_row = config.get("terminal_row", 24)
        utils.set_winsize(sys.stdin, terminal_col, terminal_row)
        utils.set_winsize(sys.stderr, terminal_col, terminal_row)
        utils.set_winsize(sys.stdout, terminal_col, terminal_row)
    
    if config.seed == -1:
        config.seed = random.randint(0, 10 ** 8)
    
    seed_everything(config.seed)
    log.info(f"Run dir: {os.path.realpath('./')}")
    
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    if config.get("print_config"):
        utils.print_config(config, fields=tuple(config.keys()), resolve=True)
    
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")
    
    if config.name == "train":
        return train(config)
    
    if config.name == "inference":
        return pred(config)
    
    if config.name == "evaluation":
        return evaluate(config)


if __name__ == "__main__":
    main()
