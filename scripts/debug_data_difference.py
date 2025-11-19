#!/usr/bin/env python3
"""
Скрипт для отладки различий в данных между экспериментом и дашбордом.

Проверяет:
1. Какие JSON файлы использовались в эксперименте
2. Какие JSON файлы используются в дашборде
3. Идентичны ли результаты агрегации
4. Идентичны ли относительные признаки
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scale import aggregate
from model_development.feature_selection_automated import evaluate_feature_set, identify_sample_type


def load_json_files_from_dir(predictions_dir):
    """Загружает все JSON файлы из директории."""
    predictions = {}
    json_files = list(Path(predictions_dir).glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                image_name = json_file.stem
                predictions[image_name] = data
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {json_file}: {e}")
    
    return predictions, json_files


def compare_dataframes_detailed(df1, df2, name1="DataFrame 1", name2="DataFrame 2"):
    """Детальное сравнение двух DataFrame."""
    print(f"\n{'='*80}")
    print(f"ДЕТАЛЬНОЕ СРАВНЕНИЕ: {name1} vs {name2}")
    print(f"{'='*80}")
    
    # Базовые характеристики
    print(f"\n{name1}:")
    print(f"  Строк: {len(df1)}")
    print(f"  Колонок: {len(df1.columns)}")
    
    print(f"\n{name2}:")
    print(f"  Строк: {len(df2)}")
    print(f"  Колонок: {len(df2.columns)}")
    
    # Проверяем идентичность образцов
    if "image" in df1.columns and "image" in df2.columns:
        samples1 = set(df1["image"].unique())
        samples2 = set(df2["image"].unique())
        
        print(f"\nОбразцы:")
        print(f"  {name1}: {len(samples1)} уникальных образцов")
        print(f"  {name2}: {len(samples2)} уникальных образцов")
        
        missing_in_2 = samples1 - samples2
        extra_in_2 = samples2 - samples1
        
        if missing_in_2:
            print(f"  ⚠️ Отсутствуют в {name2}: {list(missing_in_2)[:10]}{'...' if len(missing_in_2) > 10 else ''}")
        if extra_in_2:
            print(f"  ⚠️ Лишние в {name2}: {list(extra_in_2)[:10]}{'...' if len(extra_in_2) > 10 else ''}")
        
        if samples1 == samples2:
            print(f"  ✅ Образцы идентичны")
            
            # Сравниваем значения для общих образцов
            common_samples = sorted(samples1)
            df1_sorted = df1.set_index("image").loc[common_samples].reset_index()
            df2_sorted = df2.set_index("image").loc[common_samples].reset_index()
            
            # Сравниваем числовые колонки
            common_cols = set(df1.columns) & set(df2.columns) - {"image"}
            numeric_cols = [c for c in common_cols if df1[c].dtype in [np.number] and df2[c].dtype in [np.number]]
            
            if numeric_cols:
                print(f"\nСравнение числовых значений ({len(numeric_cols)} колонок):")
                max_diffs = []
                
                for col in numeric_cols:
                    if col in df1_sorted.columns and col in df2_sorted.columns:
                        diff = (df1_sorted[col] - df2_sorted[col]).abs()
                        max_diff = diff.max()
                        mean_diff = diff.mean()
                        
                        if max_diff > 1e-10:
                            max_diffs.append((col, max_diff, mean_diff))
                
                if max_diffs:
                    max_diffs.sort(key=lambda x: x[1], reverse=True)
                    print(f"  ⚠️ Обнаружены различия в {len(max_diffs)} колонках:")
                    for col, max_diff, mean_diff in max_diffs[:10]:
                        print(f"    {col}: макс. разница = {max_diff:.10f}, средняя = {mean_diff:.10f}")
                else:
                    print(f"  ✅ Все числовые значения идентичны (разница < 1e-10)")
        else:
            print(f"  ❌ Образцы различаются!")
    else:
        print(f"  ⚠️ Колонка 'image' отсутствует в одном из DataFrame")


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Отладка различий в данных между экспериментом и дашбордом")
    parser.add_argument("experiment_name", nargs="?", default="feature_selection_all_methods",
                       help="Имя эксперимента")
    parser.add_argument("--predictions-dir", type=str, default="results/predictions",
                       help="Директория с JSON файлами (по умолчанию: results/predictions)")
    
    args = parser.parse_args()
    
    experiments_dir = Path("experiments")
    experiment_dir = experiments_dir / args.experiment_name
    
    if not experiment_dir.exists():
        print(f"❌ Эксперимент не найден: {experiment_dir}")
        sys.exit(1)
    
    print("="*80)
    print("ОТЛАДКА РАЗЛИЧИЙ В ДАННЫХ: ЭКСПЕРИМЕНТ vs ДАШБОРД")
    print("="*80)
    
    # 1. Загружаем данные из эксперимента
    print(f"\n1. Загрузка данных из эксперимента: {args.experiment_name}")
    
    # Ищем CSV файлы с данными
    all_features_files = list(experiment_dir.glob("all_features_*.csv"))
    relative_features_files = list(experiment_dir.glob("relative_features_*.csv"))
    aggregated_files = list(experiment_dir.glob("aggregated_data_*.csv"))
    
    if all_features_files:
        df_experiment_all = pd.read_csv(sorted(all_features_files)[-1])
        print(f"✓ Загружены all_features из: {sorted(all_features_files)[-1].name}")
    elif relative_features_files:
        df_experiment_all = pd.read_csv(sorted(relative_features_files)[-1])
        print(f"✓ Загружены relative_features из: {sorted(relative_features_files)[-1].name}")
    else:
        print(f"❌ Не найдены данные в эксперименте")
        sys.exit(1)
    
    # 2. Загружаем JSON файлы из results/predictions
    print(f"\n2. Загрузка JSON файлов из: {args.predictions_dir}")
    predictions, json_files = load_json_files_from_dir(args.predictions_dir)
    print(f"✓ Загружено {len(predictions)} JSON файлов")
    
    # 3. Агрегируем данные из JSON (как в эксперименте)
    print(f"\n3. Агрегация данных из JSON (как в эксперименте)...")
    df_current_aggregated = aggregate.load_predictions_batch(args.predictions_dir)
    print(f"✓ Агрегировано {len(df_current_aggregated)} образцов")
    
    # 4. Создаем относительные признаки (как в эксперименте)
    print(f"\n4. Создание относительных признаков (как в эксперименте)...")
    df_current_relative = aggregate.create_relative_features(df_current_aggregated)
    print(f"✓ Создано {len(df_current_relative.columns)} колонок")
    
    # 5. Получаем all_features (как в эксперименте)
    print(f"\n5. Получение all_features (как в эксперименте)...")
    df_current_all = aggregate.select_all_feature_columns(df_current_relative)
    print(f"✓ Получено {len(df_current_all.columns)} колонок")
    
    # 6. Сравниваем данные
    print(f"\n6. Сравнение данных...")
    
    # Сравниваем all_features
    compare_dataframes_detailed(
        df_experiment_all, 
        df_current_all,
        "Эксперимент (all_features)", 
        "Текущие данные (all_features)"
    )
    
    # Если есть aggregated_data в эксперименте, сравниваем и их
    if aggregated_files:
        print(f"\n7. Сравнение агрегированных данных...")
        df_experiment_aggregated = pd.read_csv(sorted(aggregated_files)[-1])
        compare_dataframes_detailed(
            df_experiment_aggregated,
            df_current_aggregated,
            "Эксперимент (aggregated)", 
            "Текущие данные (aggregated)"
        )
    
    # 8. Вычисляем score для сравнения
    print(f"\n8. Вычисление score для сравнения...")
    
    # Загружаем конфигурацию эксперимента
    best_features_files = list(experiment_dir.glob("best_features_*.json"))
    if best_features_files:
        with open(sorted(best_features_files)[-1], 'r', encoding='utf-8') as f:
            experiment_config = json.load(f)
        
        features = experiment_config.get('selected_features', [])
        experiment_score = experiment_config.get('metrics', {}).get('score', None)
        
        if features and experiment_score:
            # Определяем образцы
            mod_samples = []
            normal_samples = []
            
            for img_name in df_experiment_all["image"].unique():
                sample_type = identify_sample_type(str(img_name))
                if sample_type == 'mod':
                    mod_samples.append(img_name)
                elif sample_type == 'normal':
                    normal_samples.append(img_name)
            
            # Вычисляем score на данных эксперимента
            metrics_exp = evaluate_feature_set(
                df_experiment_all,
                features,
                mod_samples,
                normal_samples
            )
            
            # Вычисляем score на текущих данных
            mod_samples_current = []
            normal_samples_current = []
            
            for img_name in df_current_all["image"].unique():
                sample_type = identify_sample_type(str(img_name))
                if sample_type == 'mod':
                    mod_samples_current.append(img_name)
                elif sample_type == 'normal':
                    normal_samples_current.append(img_name)
            
            metrics_current = evaluate_feature_set(
                df_current_all,
                features,
                mod_samples_current,
                normal_samples_current
            )
            
            print(f"\nScore:")
            print(f"  Из JSON эксперимента: {experiment_score:.6f}")
            print(f"  Пересчет на данных эксперимента: {metrics_exp['score']:.6f}")
            print(f"  Пересчет на текущих данных: {metrics_current['score']:.6f}")
            
            print(f"\nРазница:")
            print(f"  JSON vs пересчет (эксперимент): {abs(experiment_score - metrics_exp['score']):.6f}")
            print(f"  Эксперимент vs текущие: {abs(metrics_exp['score'] - metrics_current['score']):.6f}")
            
            if abs(metrics_exp['score'] - metrics_current['score']) > 1e-6:
                print(f"\n❌ Score различаются! Это означает, что данные различаются.")
            else:
                print(f"\n✅ Score идентичны - данные совпадают.")
    
    print(f"\n{'='*80}")
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

