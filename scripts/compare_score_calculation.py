#!/usr/bin/env python3
"""
Скрипт для параллельного сравнения вычисления score в эксперименте и дашборде.

Запускает тот же код, что использовался в эксперименте, и сравнивает результаты
с вычислениями в дашборде.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_development.feature_selection_automated import evaluate_feature_set, identify_sample_type
from scale import aggregate


def load_experiment_data(experiment_dir: Path):
    """Загружает данные из эксперимента."""
    # Ищем файлы с данными
    aggregated_files = list(experiment_dir.glob("aggregated_data_*.csv"))
    relative_files = list(experiment_dir.glob("relative_features_*.csv"))
    all_features_files = list(experiment_dir.glob("all_features_*.csv"))
    best_features_files = list(experiment_dir.glob("best_features_*.json"))
    
    if not best_features_files:
        raise ValueError(f"Не найден файл best_features_*.json в {experiment_dir}")
    
    # Загружаем конфигурацию лучшего эксперимента
    best_file = sorted(best_features_files)[-1]
    with open(best_file, 'r', encoding='utf-8') as f:
        experiment_config = json.load(f)
    
    # Загружаем данные
    if all_features_files:
        # Используем all_features (это то, что использовалось в эксперименте)
        all_features_file = sorted(all_features_files)[-1]
        df_all = pd.read_csv(all_features_file)
        print(f"✓ Загружены данные из: {all_features_file.name}")
    elif relative_files:
        # Используем relative_features
        relative_file = sorted(relative_files)[-1]
        df_all = pd.read_csv(relative_file)
        print(f"✓ Загружены данные из: {relative_file.name}")
    elif aggregated_files:
        # Создаем относительные признаки из агрегированных данных
        aggregated_file = sorted(aggregated_files)[-1]
        df = pd.read_csv(aggregated_file)
        df_all = aggregate.select_all_feature_columns(
            aggregate.create_relative_features(df)
        )
        print(f"✓ Созданы относительные признаки из: {aggregated_file.name}")
    else:
        raise ValueError(f"Не найдены данные в {experiment_dir}")
    
    return df_all, experiment_config


def calculate_score_experiment_way(df_all, experiment_config):
    """Вычисляет score тем же способом, что и в эксперименте."""
    features = experiment_config.get('selected_features', [])
    if not features:
        raise ValueError("Признаки не найдены в конфигурации эксперимента")
    
    # Определяем mod и normal образцы
    mod_samples = []
    normal_samples = []
    
    if "image" not in df_all.columns:
        raise ValueError("Колонка 'image' отсутствует в данных")
    
    for img_name in df_all["image"].unique():
        sample_type = identify_sample_type(str(img_name))
        if sample_type == 'mod':
            mod_samples.append(img_name)
        elif sample_type == 'normal':
            normal_samples.append(img_name)
    
    if len(mod_samples) == 0 or len(normal_samples) == 0:
        raise ValueError(f"Не найдены образцы: mod={len(mod_samples)}, normal={len(normal_samples)}")
    
    print(f"✓ Mod образцов: {len(mod_samples)}")
    print(f"✓ Normal образцов: {len(normal_samples)}")
    print(f"✓ Признаков: {len(features)}")
    
    # Вычисляем метрики
    metrics = evaluate_feature_set(
        df_all,
        features,
        mod_samples,
        normal_samples
    )
    
    return metrics, features, mod_samples, normal_samples


def calculate_score_dashboard_way(df_features, selected_features):
    """Вычисляет score тем же способом, что и в дашборде."""
    # Определяем mod и normal образцы
    mod_samples = []
    normal_samples = []
    
    if "image" not in df_features.columns:
        raise ValueError("Колонка 'image' отсутствует в данных")
    
    for img_name in df_features["image"].unique():
        sample_type = identify_sample_type(str(img_name))
        if sample_type == 'mod':
            mod_samples.append(img_name)
        elif sample_type == 'normal':
            normal_samples.append(img_name)
    
    if len(mod_samples) == 0 or len(normal_samples) == 0:
        raise ValueError(f"Не найдены образцы: mod={len(mod_samples)}, normal={len(normal_samples)}")
    
    # Сортируем признаки (как в дашборде)
    sorted_features = sorted([col for col in selected_features if col in df_features.columns])
    
    # Вычисляем метрики
    metrics = evaluate_feature_set(
        df_features,
        sorted_features,
        mod_samples,
        normal_samples
    )
    
    return metrics, sorted_features, mod_samples, normal_samples


def compare_dataframes(df1, df2, name1="DataFrame 1", name2="DataFrame 2"):
    """Сравнивает два DataFrame."""
    print(f"\n{'='*80}")
    print(f"СРАВНЕНИЕ ДАННЫХ: {name1} vs {name2}")
    print(f"{'='*80}")
    
    print(f"\n{name1}:")
    print(f"  Строк: {len(df1)}")
    print(f"  Колонок: {len(df1.columns)}")
    print(f"  Колонки: {list(df1.columns)[:10]}{'...' if len(df1.columns) > 10 else ''}")
    
    print(f"\n{name2}:")
    print(f"  Строк: {len(df2)}")
    print(f"  Колонок: {len(df2.columns)}")
    print(f"  Колонки: {list(df2.columns)[:10]}{'...' if len(df2.columns) > 10 else ''}")
    
    # Проверяем идентичность
    if len(df1) == len(df2) and set(df1.columns) == set(df2.columns):
        # Сравниваем значения для общих колонок
        common_cols = set(df1.columns) & set(df2.columns)
        if "image" in common_cols:
            # Сортируем по image для сравнения
            df1_sorted = df1.sort_values("image").reset_index(drop=True)
            df2_sorted = df2.sort_values("image").reset_index(drop=True)
            
            # Сравниваем числовые колонки
            numeric_cols = [c for c in common_cols if c != "image" and df1[c].dtype in [np.number]]
            if numeric_cols:
                diff = (df1_sorted[numeric_cols] - df2_sorted[numeric_cols]).abs()
                max_diff = diff.max().max()
                print(f"\n✓ Максимальная разница в числовых значениях: {max_diff:.10f}")
                if max_diff > 1e-6:
                    print(f"⚠️ Обнаружены различия в данных!")
                    # Показываем колонки с наибольшими различиями
                    max_diff_cols = diff.max().sort_values(ascending=False).head(5)
                    print(f"  Колонки с наибольшими различиями:")
                    for col, val in max_diff_cols.items():
                        print(f"    {col}: {val:.10f}")
                else:
                    print(f"✅ Данные идентичны (разница < 1e-6)")
        else:
            print(f"⚠️ Колонка 'image' отсутствует в одном из DataFrame")
    else:
        print(f"⚠️ DataFrame различаются по размеру или колонкам")


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Сравнение вычисления score в эксперименте и дашборде")
    parser.add_argument("experiment_name", nargs="?", default="feature_selection_all_methods",
                       help="Имя эксперимента (по умолчанию: feature_selection_all_methods)")
    parser.add_argument("--dashboard-data", type=str, default=None,
                       help="Путь к CSV файлу с данными из дашборда (опционально)")
    
    args = parser.parse_args()
    
    experiments_dir = Path("experiments")
    experiment_dir = experiments_dir / args.experiment_name
    
    if not experiment_dir.exists():
        print(f"❌ Эксперимент не найден: {experiment_dir}")
        sys.exit(1)
    
    print("="*80)
    print("СРАВНЕНИЕ ВЫЧИСЛЕНИЯ SCORE: ЭКСПЕРИМЕНТ vs ДАШБОРД")
    print("="*80)
    
    # Загружаем данные из эксперимента
    print(f"\n1. Загрузка данных из эксперимента: {args.experiment_name}")
    try:
        df_experiment, experiment_config = load_experiment_data(experiment_dir)
    except Exception as e:
        print(f"❌ Ошибка загрузки данных эксперимента: {e}")
        sys.exit(1)
    
    # Вычисляем score способом эксперимента
    print(f"\n2. Вычисление score способом эксперимента")
    try:
        metrics_experiment, features_experiment, mod_samples_exp, normal_samples_exp = calculate_score_experiment_way(
            df_experiment, experiment_config
        )
        print(f"✓ Score (эксперимент): {metrics_experiment['score']:.6f}")
        print(f"  Separation: {metrics_experiment['separation']:.6f}")
        print(f"  Mod (норм. PC1): {metrics_experiment['mean_pc1_norm_mod']:.6f}")
        print(f"  Explained variance: {metrics_experiment['explained_variance']:.6f}")
    except Exception as e:
        print(f"❌ Ошибка вычисления score (эксперимент): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Загружаем данные из дашборда (если указаны)
    if args.dashboard_data:
        print(f"\n3. Загрузка данных из дашборда: {args.dashboard_data}")
        try:
            df_dashboard = pd.read_csv(args.dashboard_data)
            print(f"✓ Загружены данные из дашборда")
        except Exception as e:
            print(f"❌ Ошибка загрузки данных дашборда: {e}")
            sys.exit(1)
    else:
        # Используем те же данные, что и в эксперименте
        print(f"\n3. Использование тех же данных для дашборда")
        df_dashboard = df_experiment.copy()
    
    # Вычисляем score способом дашборда
    print(f"\n4. Вычисление score способом дашборда")
    try:
        metrics_dashboard, features_dashboard, mod_samples_dash, normal_samples_dash = calculate_score_dashboard_way(
            df_dashboard, features_experiment
        )
        print(f"✓ Score (дашборд): {metrics_dashboard['score']:.6f}")
        print(f"  Separation: {metrics_dashboard['separation']:.6f}")
        print(f"  Mod (норм. PC1): {metrics_dashboard['mean_pc1_norm_mod']:.6f}")
        print(f"  Explained variance: {metrics_dashboard['explained_variance']:.6f}")
    except Exception as e:
        print(f"❌ Ошибка вычисления score (дашборд): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Сравниваем результаты
    print(f"\n{'='*80}")
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print(f"{'='*80}")
    
    print(f"\nScore:")
    print(f"  Эксперимент: {metrics_experiment['score']:.6f}")
    print(f"  Дашборд:     {metrics_dashboard['score']:.6f}")
    score_diff = metrics_dashboard['score'] - metrics_experiment['score']
    print(f"  Разница:     {score_diff:+.6f}")
    
    print(f"\nSeparation:")
    print(f"  Эксперимент: {metrics_experiment['separation']:.6f}")
    print(f"  Дашборд:     {metrics_dashboard['separation']:.6f}")
    sep_diff = metrics_dashboard['separation'] - metrics_experiment['separation']
    print(f"  Разница:     {sep_diff:+.6f}")
    
    print(f"\nMod (норм. PC1):")
    print(f"  Эксперимент: {metrics_experiment['mean_pc1_norm_mod']:.6f}")
    print(f"  Дашборд:     {metrics_dashboard['mean_pc1_norm_mod']:.6f}")
    mod_diff = metrics_dashboard['mean_pc1_norm_mod'] - metrics_experiment['mean_pc1_norm_mod']
    print(f"  Разница:     {mod_diff:+.6f}")
    
    print(f"\nExplained variance:")
    print(f"  Эксперимент: {metrics_experiment['explained_variance']:.6f}")
    print(f"  Дашборд:     {metrics_dashboard['explained_variance']:.6f}")
    var_diff = metrics_dashboard['explained_variance'] - metrics_experiment['explained_variance']
    print(f"  Разница:     {var_diff:+.6f}")
    
    # Проверяем признаки
    print(f"\nПризнаки:")
    print(f"  Эксперимент: {len(features_experiment)} признаков")
    print(f"  Дашборд:     {len(features_dashboard)} признаков")
    if sorted(features_experiment) == sorted(features_dashboard):
        print(f"  ✅ Признаки идентичны")
    else:
        print(f"  ❌ Признаки различаются!")
        missing = set(features_experiment) - set(features_dashboard)
        extra = set(features_dashboard) - set(features_experiment)
        if missing:
            print(f"    Отсутствуют в дашборде: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        if extra:
            print(f"    Лишние в дашборде: {list(extra)[:5]}{'...' if len(extra) > 5 else ''}")
    
    # Проверяем образцы
    print(f"\nОбразцы:")
    print(f"  Mod (эксперимент): {len(mod_samples_exp)}")
    print(f"  Mod (дашборд):     {len(mod_samples_dash)}")
    print(f"  Normal (эксперимент): {len(normal_samples_exp)}")
    print(f"  Normal (дашборд):     {len(normal_samples_dash)}")
    if set(mod_samples_exp) == set(mod_samples_dash) and set(normal_samples_exp) == set(normal_samples_dash):
        print(f"  ✅ Образцы идентичны")
    else:
        print(f"  ❌ Образцы различаются!")
        mod_missing = set(mod_samples_exp) - set(mod_samples_dash)
        mod_extra = set(mod_samples_dash) - set(mod_samples_exp)
        if mod_missing:
            print(f"    Mod отсутствуют в дашборде: {list(mod_missing)[:5]}{'...' if len(mod_missing) > 5 else ''}")
        if mod_extra:
            print(f"    Mod лишние в дашборде: {list(mod_extra)[:5]}{'...' if len(mod_extra) > 5 else ''}")
    
    # Сравниваем данные
    if args.dashboard_data:
        compare_dataframes(df_experiment, df_dashboard, "Эксперимент", "Дашборд")
    else:
        print(f"\n✓ Использованы те же данные для обоих вычислений")
    
    # Итоговый вывод
    print(f"\n{'='*80}")
    if abs(score_diff) < 1e-6:
        print("✅ Score ИДЕНТИЧНЫ!")
    else:
        print(f"❌ Score РАЗЛИЧАЮТСЯ на {abs(score_diff):.6f}")
        print(f"   Это может быть из-за:")
        print(f"   1. Разных данных (разные значения признаков)")
        print(f"   2. Разных образцов (mod/normal)")
        print(f"   3. Разных признаков")
        print(f"   4. Численной нестабильности PCA (несмотря на random_state=42)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

