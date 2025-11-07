"""
Агрегация предсказаний патологий в таблицу с количественными признаками.

Модуль для преобразования предсказаний патологий (из JSON или словаря)
в DataFrame с признаками count и area для каждого типа патологии.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from . import domain


def aggregate_predictions_from_dict(
    predictions: dict[str, list[domain.Prediction]], image_name: str
) -> dict[str, Union[int, float]]:
    """
    Агрегирует предсказания из словаря в статистику по каждому классу.

    Args:
        predictions: Словарь {class_name: [list of Prediction]}
        image_name: Имя изображения/биоптата

    Returns:
        Словарь со статистикой: {class_name_count: int, class_name_area: float, ...}
    """
    stats = {"image": image_name}

    for cls_name, pred_list in predictions.items():
        if len(pred_list) == 0:
            stats[f"{cls_name}_count"] = 0
            stats[f"{cls_name}_area"] = 0.0
        else:
            stats[f"{cls_name}_count"] = len(pred_list)
            stats[f"{cls_name}_area"] = np.sum(
                [
                    p.polygon.area() if p.polygon else p.box.area()
                    for p in pred_list
                ]
            )

    return stats


def aggregate_predictions_from_json(json_path: Union[str, Path]) -> dict[str, Union[int, float]]:
    """
    Загружает предсказания из JSON файла и агрегирует их.

    Args:
        json_path: Путь к JSON файлу с предсказаниями

    Returns:
        Словарь со статистикой
    """
    json_path = Path(json_path)
    predictions = domain.predictions_from_json(str(json_path))
    image_name = json_path.stem

    return aggregate_predictions_from_dict(predictions, image_name)


def load_predictions_batch(predictions_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Загружает все предсказания из директории и создает DataFrame.

    Args:
        predictions_dir: Путь к директории с JSON файлами предсказаний

    Returns:
        DataFrame с колонками: image, {class}_count, {class}_area, ...
    """
    predictions_dir = Path(predictions_dir)
    rows = []

    for json_file in predictions_dir.glob("*.json"):
        stats = aggregate_predictions_from_json(json_file)
        rows.append(stats)

    return pd.DataFrame(rows)


def create_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает относительные признаки, нормализованные по Crypts.

    Args:
        df: DataFrame с колонками {class}_count и {class}_area

    Returns:
        DataFrame с дополнительными колонками:
        - {class}_relative_count = count / Crypts_count
        - {class}_relative_area = area / Crypts_area
        - {class}_mean_relative_area = relative_area / count
    """
    df_new = pd.DataFrame()
    df_new["image"] = df["image"]

    area_cols = [col for col in df.columns if col.endswith("_area")]
    count_cols = [col for col in df.columns if col.endswith("_count")]

    for col in count_cols:
        if col != "Crypts_count" and "Crypts_count" in df.columns:
            df_new[col.replace("count", "relative_count")] = (
                df[col] / df["Crypts_count"]
            )

    for col in area_cols:
        if col != "Crypts_area" and "Crypts_area" in df.columns:
            relative_area = df[col] / df["Crypts_area"]
            df_new[col.replace("area", "relative_area")] = relative_area

            count_col = col.replace("area", "count")
            if count_col in df.columns:
                count_values = df[count_col].replace(0, np.nan)
                df_new[col.replace("area", "mean_relative_area")] = (
                    relative_area / count_values
                )

    return df_new


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выбирает важные колонки признаков для анализа.

    Args:
        df: DataFrame с относительными признаками

    Returns:
        DataFrame с отобранными колонками
    """
    cols = [
        "image",
        "Mild_relative_count",
        "Dysplasia_relative_count",
        "Moderate_relative_count",
        "Meta_relative_count",
        "Plasma Cells_relative_count",
        "Neutrophils_relative_count",
        "EoE_relative_count",
        "Enterocytes_relative_count",
        "Granulomas_relative_count",
        "Mild_relative_area",
        "Mild_mean_relative_area",
        "Surface epithelium_relative_area",
        "Dysplasia_relative_area",
        "Dysplasia_mean_relative_area",
        "Moderate_relative_area",
        "Moderate_mean_relative_area",
        "Muscularis mucosae_relative_area",
        "Meta_relative_area",
        "Meta_mean_relative_area",
        "Plasma Cells_relative_area",
        "Plasma Cells_mean_relative_area",
        "Neutrophils_relative_area",
        "Neutrophils_mean_relative_area",
        "Enterocytes_relative_area",
        "Enterocytes_mean_relative_area",
        "Granulomas_relative_area",
    ]

    available_cols = [col for col in cols if col in df.columns]
    return df[available_cols]

