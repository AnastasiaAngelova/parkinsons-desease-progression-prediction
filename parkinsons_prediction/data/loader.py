import pandas as pd
import dvc.api
from pathlib import Path
from typing import Union

def load_data(cfg: dict) -> Union[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame]
]:
    """
    Загружает данные из DVC или с локального пути.
    
    Параметры:
        cfg (dict): Конфигурация, содержащая:
            - use_dvc (bool): Использовать ли DVC
            - mode (str): 'train', 'inference' или 'all'
            - Пути к нужным CSV-файлам (строки)
    
    Возвращает:
        Кортеж с соответствующими DataFrame в зависимости от режима.
    """
    base_path = Path(__file__).resolve().parents[2]

    def read_csv(path_str: str) -> pd.DataFrame:
        local_path = Path(path_str)
        if cfg.data.get("use_dvc", True) and not local_path.exists():
            with dvc.api.open(
                path=path_str,
                repo=str(base_path),
                mode='r'
            ) as fd:
                df = pd.read_csv(fd)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(local_path, index=False)
                return df
        else:
            return pd.read_csv(local_path)

    mode = cfg.task.get("mode", "all")

    train_keys = ["train_proteins", "train_clinical", "supplemental_clinical"]
    test_keys = ["test_proteins", "test"]

    train_data = []
    test_data = []

    if mode in ("train", "all"):
        train_data = [read_csv(cfg.data[key]) for key in train_keys]

    if mode in ("inference", "all"):
        test_data = [read_csv(cfg.data[key]) for key in test_keys]

    if mode == "train":
        return tuple(train_data)
    elif mode == "inference":
        return tuple(test_data)
    elif mode == "all":
        return tuple(train_data + test_data)
    else:
        raise ValueError(f"Некорректный режим: {mode}")
