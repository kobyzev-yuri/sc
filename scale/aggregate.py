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
    
    Исключает:
        - White space (служебный класс, не используется как признак)
    
    Включает:
        - Surface epithelium и Muscularis mucosae (теперь используются как признаки)
    """
    stats = {"image": image_name}
    
    # Список классов для исключения из абсолютных признаков (только White space)
    # Surface epithelium и Muscularis mucosae теперь включены как признаки
    excluded_classes = ["White space"]

    for cls_name, pred_list in predictions.items():
        # Исключаем структурные элементы
        if cls_name in excluded_classes:
            continue
            
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
    
    Исключает:
        - Crypts (используется как нормализатор)
        - White space (служебный класс)
    
    Включает:
        - Surface epithelium и Muscularis mucosae (теперь используются как признаки)
    """
    df_new = pd.DataFrame()
    df_new["image"] = df["image"]

    # Список классов для исключения (только Crypts и White space)
    # Surface epithelium и Muscularis mucosae теперь включены как признаки
    excluded_classes = ["Crypts", "White space"]

    area_cols = [col for col in df.columns if col.endswith("_area")]
    count_cols = [col for col in df.columns if col.endswith("_count")]

    for col in count_cols:
        # Исключаем Crypts и другие структурные элементы
        class_name = col.replace("_count", "")
        if class_name not in excluded_classes and "Crypts_count" in df.columns:
            df_new[col.replace("count", "relative_count")] = (
                df[col] / df["Crypts_count"]
            )

    for col in area_cols:
        # Исключаем Crypts и другие структурные элементы
        class_name = col.replace("_area", "")
        if class_name not in excluded_classes and "Crypts_area" in df.columns:
            relative_area = df[col] / df["Crypts_area"]
            df_new[col.replace("area", "relative_area")] = relative_area

            count_col = col.replace("area", "count")
            if count_col in df.columns:
                # Mean Relative Area = средняя относительная площадь на один объект
                # Формула: mean_relative_area = relative_area / count
                # Где: relative_area = area / Crypts_area
                # Итоговая формула: mean_relative_area = (area / Crypts_area) / count = area / (count * Crypts_area)
                # Это средний размер одного объекта типа X относительно размера крипты
                # Заменяем 0 на NaN, чтобы избежать деления на 0
                count_values = df[count_col].replace(0, np.nan)
                df_new[col.replace("area", "mean_relative_area")] = (
                    relative_area / count_values
                )
            # Для Surface epithelium и Muscularis mucosae создается только relative_area
            # (у них нет count, так как это области, а не отдельные объекты)

    return df_new


def select_all_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выбирает ВСЕ доступные колонки признаков для анализа (старый подход).
    
    Используется когда нужен полный набор признаков без фильтрации по loadings.

    Args:
        df: DataFrame с относительными признаками

    Returns:
        DataFrame с отобранными колонками
    
    Примечание о количестве признаков:
    - Относительные признаки: 10 классов × 3 (relative_count + relative_area + mean_relative_area) = 30 признаков
      Классы: Mild, Dysplasia, Moderate, Meta, Plasma Cells, Neutrophils, EoE, Enterocytes, Granulomas, Paneth
    - Абсолютные признаки: (10 классов + 1 Crypts) × 2 (count + area) = 22 признака
      Исключены: Surface epithelium, Muscularis mucosae (структурные элементы, не используются как признаки)
    - Относительных признаков больше (30 vs 22), потому что для каждого класса создается 3 признака вместо 2
    """
    # Формируем список признаков парами: relative_count, relative_area, mean_relative_area
    # ВАЖНО: Полный набор признаков для каждого патологического класса улучшает качество GMM аппроксимации
    # Исключаем структурные элементы: Surface epithelium и Muscularis mucosae
    # (они уже исключены в create_relative_features, но здесь явно перечисляем только нужные классы)
    cols = [
        "image",
        # Mild: count, area, mean_area
        "Mild_relative_count",
        "Mild_relative_area",
        "Mild_mean_relative_area",
        # Dysplasia: count, area, mean_area
        "Dysplasia_relative_count",
        "Dysplasia_relative_area",
        "Dysplasia_mean_relative_area",
        # Moderate: count, area, mean_area
        "Moderate_relative_count",
        "Moderate_relative_area",
        "Moderate_mean_relative_area",
        # Meta: count, area, mean_area
        "Meta_relative_count",
        "Meta_relative_area",
        "Meta_mean_relative_area",
        # Plasma Cells: count, area, mean_area
        "Plasma Cells_relative_count",
        "Plasma Cells_relative_area",
        "Plasma Cells_mean_relative_area",
        # Neutrophils: count, area, mean_area
        "Neutrophils_relative_count",
        "Neutrophils_relative_area",
        "Neutrophils_mean_relative_area",
        # EoE: count, area, mean_area (ВАЖНО: патологический признак)
        "EoE_relative_count",
        "EoE_relative_area",
        "EoE_mean_relative_area",
        # Enterocytes: count, area, mean_area
        "Enterocytes_relative_count",
        "Enterocytes_relative_area",
        "Enterocytes_mean_relative_area",
        # Granulomas: count, area, mean_area (ВАЖНО: патологический признак)
        "Granulomas_relative_count",
        "Granulomas_relative_area",
        "Granulomas_mean_relative_area",
        # Paneth: count, area, mean_area
        "Paneth_relative_count",
        "Paneth_relative_area",
        "Paneth_mean_relative_area",
        # Структурные элементы (только area, без count):
        # Surface epithelium и Muscularis mucosae - это области, а не отдельные объекты
        # Но теперь включены как признаки
        "Surface epithelium_relative_area",
        "Muscularis mucosae_relative_area",
    ]

    available_cols = [col for col in cols if col in df.columns]
    return df[available_cols]


def select_feature_columns(
    df: pd.DataFrame,
    use_positive_loadings: bool = True,
    min_loading: float = 0.05,
    exclude_paneth: bool = True,
) -> pd.DataFrame:
    """
    Выбирает важные колонки признаков для анализа.
    
    По умолчанию использует только признаки с положительными loadings в PC1,
    что улучшает объясненную дисперсию и корректность классификации.
    
    Args:
        df: DataFrame с относительными признаками
        use_positive_loadings: Если True (по умолчанию), использует только признаки 
                              с положительными loadings. Если False, использует все признаки.
        min_loading: Минимальный loading для включения признака (используется если use_positive_loadings=True)
        exclude_paneth: Если True, исключает Paneth признаки (используется если use_positive_loadings=True)

    Returns:
        DataFrame с отобранными колонками
    
    Примеры:
        # Использовать положительные loadings (по умолчанию)
        df_features = select_feature_columns(df)
        
        # Использовать все признаки (старый подход)
        df_features = select_feature_columns(df, use_positive_loadings=False)
        
        # Использовать положительные loadings с другим порогом
        df_features = select_feature_columns(df, min_loading=0.1)
    """
    if use_positive_loadings:
        return select_positive_loadings_features(df, min_loading=min_loading, exclude_paneth=exclude_paneth)
    else:
        return select_all_feature_columns(df)


def select_positive_loadings_features(
    df: pd.DataFrame,
    min_loading: float = 0.05,
    exclude_paneth: bool = True,
) -> pd.DataFrame:
    """
    Выбирает только признаки с положительными loadings в PC1.
    
    Это альтернативный подход к select_feature_columns(), который использует
    только признаки, которые увеличивают PC1 (т.е. коррелируют с патологией).
    
    Args:
        df: DataFrame с относительными признаками
        min_loading: Минимальный loading для включения признака (по умолчанию 0.05)
        exclude_paneth: Если True, исключает Paneth признаки (по умолчанию True)
        
    Returns:
        DataFrame с отобранными колонками
        
    Примечание:
        Эта функция требует обучения PCA для определения loadings, поэтому
        она должна использоваться на полном наборе данных для обучения.
        Для новых данных используйте те же признаки, что были отобраны при обучении.
    """
    # Сначала получаем все доступные признаки (используем select_all_feature_columns чтобы избежать рекурсии)
    df_all_features = select_all_feature_columns(df)
    feature_cols = [c for c in df_all_features.columns if c != "image"]
    
    if len(feature_cols) == 0:
        return df_all_features
    
    # Обучаем PCA для получения loadings
    from . import pca_scoring
    pca_scorer = pca_scoring.PCAScorer()
    pca_scorer.fit(df_all_features, feature_cols)
    
    # Получаем loadings
    loadings = pca_scorer.get_feature_importance()
    
    # Фильтруем только положительные loadings выше порога
    positive_features = [
        feat for feat, loading in loadings.items()
        if loading > min_loading
    ]
    
    # Исключаем Paneth если нужно
    if exclude_paneth:
        positive_features = [f for f in positive_features if 'Paneth' not in f]
    
    # Возвращаем DataFrame с только положительными признаками
    cols_to_keep = ["image"] + positive_features
    available_cols = [col for col in cols_to_keep if col in df_all_features.columns]
    
    return df_all_features[available_cols]

