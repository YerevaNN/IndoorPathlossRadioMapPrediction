# Indoor Pathloss Radio Map Prediction

Implementation for indoor radio map prediction using Vision Transformers.

**Paper:** [Vision Transformers for Efficient Indoor Pathloss Radio Map Prediction](https://doi.org/10.3390/electronics14101905)

## Environment Setup

Create and activate the Conda environment:

```bash
conda env create
conda activate indoor_pathloss
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set the required paths:
   - `PREDICTIONS_PATH` — directory for inference outputs
   - `OUTPUT_DIR` — directory for training outputs and run logs
   - `AIM_REPO` — path for Aim experiment logging
   - `ICASSP_ORIG_PATH` — path to original ICASSP dataset
   - `ICASSP_TASK1_PATH`, `ICASSP_TASK2_PATH`, `ICASSP_TASK3_PATH` — paths to preprocessed task data

3. Customize `configs/` as needed (e.g., checkpoint path in `configs/inference.yaml`, prediction path in `configs/evaluation.yaml`).

## Running the Code

Scripts in the [run/](run/) directory should be executed from the project root.

| Script | Purpose |
|--------|---------|
| `run/train_task1.sh` | Train model for ICASSP Task 1 |
| `run/train_task2.sh` | Train model for ICASSP Task 2 |
| `run/train_task3.sh` | Train model for ICASSP Task 3 |
| `run/inference.sh` | Run inference (uses checkpoint and data paths from configs) |
| `run/evaluation.sh` | Evaluate predictions (MSE) |

## Project Structure

- `src/` — algorithms, datamodules, networks
- `configs/` — Hydra configuration files
- `run/` — executable scripts for training, inference, and evaluation
- `jupyter/` — notebooks for inference
