import logging
import os

import numpy as np
from omegaconf import DictConfig
from sklearn import metrics
from tqdm import tqdm

log = logging.getLogger(__name__)


def evaluate(config: DictConfig) -> None:
    log.info(f"Evaluating predictions from {config['prediction_path']}")
    pred_path = config["prediction_path"]
    multiple = all(os.path.isdir(os.path.join(pred_path, path)) for path in os.listdir(pred_path))
    if multiple:
        pred_paths = [os.path.join(pred_path, path) for path in sorted(os.listdir(pred_path), key=int)]
    else:
        pred_paths = [pred_path]
    for pred_path in pred_paths:
        log.info(f"data_idx={os.path.basename(os.path.normpath(pred_path))}")
        gts = []
        preds = []
        for i in tqdm(sorted(os.listdir(pred_path), key=lambda p: int(p.split(".")[0]))):
            pred_data = np.load(os.path.join(pred_path, i))
            pred = pred_data["out"] * 160
            gt = pred_data["orig_out"]
            preds.extend(pred.flatten().tolist())
            gts.extend(gt.flatten().tolist())
        mse = metrics.mean_squared_error(gts, preds)
        log.info(f"MSE: {mse}")
