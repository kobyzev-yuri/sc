"""
Модуль для интеграции выбора экспериментов в dashboard.

Позволяет:
- Выбирать эксперимент из списка
- Загружать признаки из эксперимента
- Сохранять изменения пользователя отдельно (не перезаписывая исходный эксперимент)
- Отслеживать источник конфигурации
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd

# Добавляем путь к model_development для импорта
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model_development import feature_selection_versioning


def list_available_experiments(experiments_dir: Path = Path("experiments"), use_tracker: bool = True, top_n: Optional[int] = None) -> List[Dict]:
    """
    Возвращает список доступных экспериментов.
    
    Использует ExperimentTracker для получения отсортированного списка лучших экспериментов.
    Если трекер недоступен или пуст, использует fallback на сканирование директорий.
    
    Args:
        experiments_dir: Базовая директория с экспериментами
        use_tracker: Использовать ExperimentTracker (по умолчанию True)
        top_n: Максимальное число экспериментов для возврата (None = все, по умолчанию 3 для dashboard)
        
    Returns:
        Список словарей с информацией об экспериментах, отсортированный по score (лучшие первые)
    """
    experiments = []
    experiments_dir = Path(experiments_dir)
    
    # Пробуем использовать трекер экспериментов
    if use_tracker:
        try:
            from model_development.experiment_tracker import ExperimentTracker
            
            tracker = ExperimentTracker(experiments_dir)
            df_experiments = tracker.list_experiments(sort_by="score", limit=top_n)
            
            if len(df_experiments) > 0:
                # Преобразуем DataFrame в список словарей
                for _, row in df_experiments.iterrows():
                    # Загружаем детали эксперимента для получения признаков
                    exp_details = tracker.get_experiment_details(row['id'])
                    if exp_details:
                        experiments.append({
                            'name': row['name'],
                            'path': str(experiments_dir / row['directory']),
                            'method': row['method'],
                            'score': float(row['score']),
                            'separation': float(row['separation']),
                            'mod_norm': float(row['mod_norm']),
                            'n_features': int(row['n_features']),
                            'features': exp_details.get('parameters', {}).get('selected_features', []),
                            'timestamp': row['timestamp'],
                            'train_set': row.get('train_set', 'unknown'),
                            'aggregation_version': row.get('aggregation_version', 'unknown'),
                        })
                
                # Трекер уже отсортировал по score, возвращаем как есть (уже ограничено top_n)
                return experiments
        except Exception as e:
            # Если трекер недоступен, используем fallback
            pass
    
    # Fallback: сканируем директории напрямую
    for exp_dir in experiments_dir.rglob("*"):
        if not exp_dir.is_dir():
            continue
        
        json_files = list(exp_dir.glob("best_features_*.json"))
        if json_files:
            # Берем последний файл
            best_file = sorted(json_files)[-1]
            
            try:
                with open(best_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                experiments.append({
                    'name': exp_dir.name,
                    'path': str(exp_dir),
                    'method': config.get('method', 'unknown'),
                    'score': config.get('metrics', {}).get('score', 0),
                    'separation': config.get('metrics', {}).get('separation', 0),
                    'mod_norm': config.get('metrics', {}).get('mean_pc1_norm_mod', 0),
                    'n_features': len(config.get('selected_features', [])),
                    'features': config.get('selected_features', []),
                    'timestamp': config.get('timestamp', ''),
                    'train_set': 'unknown',
                    'aggregation_version': 'unknown',
                })
            except Exception:
                continue
    
    # Сортируем по score (лучшие первые)
    experiments.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Ограничиваем до top_n, если указано
    if top_n is not None and top_n > 0:
        experiments = experiments[:top_n]
    
    return experiments


def load_experiment_features(experiment_name: str, experiments_dir: Path = Path("experiments")) -> Optional[Dict]:
    """
    Загружает признаки из эксперимента.
    
    Args:
        experiment_name: Имя эксперимента или путь к директории
        experiments_dir: Базовая директория с экспериментами
        
    Returns:
        Словарь с признаками и метаданными или None
    """
    experiment_path = Path(experiment_name)
    
    # Проверяем, является ли это путем
    if not experiment_path.is_absolute():
        # Пробуем найти в experiments_dir
        experiment_path = experiments_dir / experiment_name
    
    if not experiment_path.exists() or not experiment_path.is_dir():
        return None
    
    # Ищем JSON файл
    json_files = list(experiment_path.glob("best_features_*.json"))
    if not json_files:
        return None
    
    best_file = sorted(json_files)[-1]
    
    try:
        with open(best_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return {
            'features': config.get('selected_features', []),
            'method': config.get('method', 'unknown'),
            'metrics': config.get('metrics', {}),
            'source_experiment': experiment_name,
            'timestamp': config.get('timestamp', ''),
        }
    except Exception:
        return None


def save_user_config(
    selected_features: List[str],
    source_experiment: Optional[str] = None,
    config_file: Path = None,
    use_relative_features: bool = True,
) -> bool:
    """
    Сохраняет конфигурацию пользователя с отслеживанием источника.
    
    Args:
        selected_features: Выбранные признаки
        source_experiment: Исходный эксперимент (если есть)
        config_file: Путь к файлу конфигурации
        use_relative_features: Использовать относительные признаки
        
    Returns:
        True если успешно сохранено
    """
    if config_file is None:
        # Конфигурационные файлы хранятся в scale/cfg для разделения с кодом
        cfg_dir = Path(__file__).parent / "cfg"
        cfg_dir.mkdir(exist_ok=True)  # Создаем директорию, если её нет
        config_file = cfg_dir / (
            "feature_selection_config_relative.json" if use_relative_features 
            else "feature_selection_config_absolute.json"
        )
    
    config_file = Path(config_file)
    
    # Загружаем текущую конфигурацию (если есть)
    current_config = {}
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                current_config = json.load(f)
        except Exception:
            pass
    
    # Формируем новую конфигурацию
    config = {
        "selected_features": selected_features,
        "description": (
            f"Выбранные {'относительные' if use_relative_features else 'абсолютные'} "
            f"признаки для построения шкалы патологии"
        ),
        "last_updated": datetime.now().isoformat(),
        "n_features": len(selected_features),
    }
    
    # Сохраняем информацию об источнике
    if source_experiment:
        config["source_experiment"] = source_experiment
        config["description"] += f" (изменено пользователем, исходный эксперимент: {source_experiment})"
    elif current_config.get("source_experiment"):
        # Сохраняем исходный эксперимент, если он был
        config["source_experiment"] = current_config["source_experiment"]
        config["description"] += f" (изменено пользователем, исходный эксперимент: {current_config['source_experiment']})"
    
    # Сохраняем метрики исходного эксперимента (если есть)
    if current_config.get("metrics"):
        config["original_metrics"] = current_config["metrics"]
    
    # Сохраняем метод исходного эксперимента (если есть)
    if current_config.get("method"):
        config["original_method"] = current_config["method"]
    
    # Добавляем флаг, что это пользовательские изменения
    config["user_modified"] = True
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")
        return False


def restore_experiment_config(
    experiment_name: str,
    config_file: Path = None,
    use_relative_features: bool = True,
) -> bool:
    """
    Восстанавливает исходную конфигурацию эксперимента.
    
    Args:
        experiment_name: Имя эксперимента
        config_file: Путь к файлу конфигурации
        use_relative_features: Использовать относительные признаки
        
    Returns:
        True если успешно восстановлено
    """
    experiment_data = load_experiment_features(experiment_name)
    
    if not experiment_data:
        return False
    
    if config_file is None:
        # Конфигурационные файлы хранятся в scale/cfg для разделения с кодом
        cfg_dir = Path(__file__).parent / "cfg"
        cfg_dir.mkdir(exist_ok=True)  # Создаем директорию, если её нет
        config_file = cfg_dir / (
            "feature_selection_config_relative.json" if use_relative_features 
            else "feature_selection_config_absolute.json"
        )
    
    config_file = Path(config_file)
    
    # Формируем конфигурацию из эксперимента
    config = {
        "selected_features": experiment_data['features'],
        "description": (
            f"Восстановлено из эксперимента '{experiment_name}'. "
            f"Метод: {experiment_data.get('method', 'unknown')}"
        ),
        "last_updated": datetime.now().isoformat(),
        "method": experiment_data.get('method', 'unknown'),
        "metrics": experiment_data.get('metrics', {}),
        "n_features": len(experiment_data['features']),
        "source_experiment": experiment_name,
        "user_modified": False,
    }
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Ошибка при восстановлении: {e}")
        return False


