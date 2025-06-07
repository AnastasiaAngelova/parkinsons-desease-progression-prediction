from pathlib import Path

import hydra
from omegaconf import DictConfig

from parkinsons_prediction.data.loader import load_data
from parkinsons_prediction.data.preprocessing import preprocess_data
from parkinsons_prediction.inference.inference import run_inference
from parkinsons_prediction.training.train_lgbm import train_lgbm_model
from parkinsons_prediction.training.train_lightning import train_lightning_model


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    Path(hydra.utils.get_original_cwd()).resolve().chdir()

    task = cfg.task.name

    if task == "train_lightning":
        train_lightning_model(cfg)
    elif task == "load_data":
        load_data(cfg)
    elif task == "preprocess_data":
        preprocess_data(cfg)
    elif task == "train_lgbm":
        train_lgbm_model(cfg)
    elif task == "infer":
        run_inference(cfg)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
