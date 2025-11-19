#!/usr/bin/env python3
"""
Скрипт для обновления score в JSON файлах экспериментов.

Пересчитывает score с использованием текущего кода (с сортировкой признаков и random_state=42)
и обновляет JSON файлы экспериментов.
"""

import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_development.feature_selection_automated import evaluate_feature_set, identify_sample_type


def update_experiment_score(experiment_dir: Path, dry_run: bool = False):
    """Обновляет score в JSON файле эксперимента."""
    experiment_dir = Path(experiment_dir)
    
    # Ищем JSON файлы
    json_files = list(experiment_dir.glob("best_features_*.json"))
    if not json_files:
        print(f"❌ Не найдено best_features_*.json в {experiment_dir}")
        return False
    
    # Берем последний файл
    json_file = sorted(json_files)[-1]
    
    # Загружаем конфигурацию
    with open(json_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Загружаем данные из эксперимента
    all_features_files = list(experiment_dir.glob("all_features_*.csv"))
    if not all_features_files:
        print(f"❌ Не найдено all_features_*.csv в {experiment_dir}")
        return False
    
    df_all = pd.read_csv(sorted(all_features_files)[-1])
    
    # Получаем признаки и метрики
    features = config.get('selected_features', [])
    old_score = config.get('metrics', {}).get('score', None)
    
    if not features:
        print(f"❌ Признаки не найдены в конфигурации")
        return False
    
    # Определяем образцы
    mod_samples = []
    normal_samples = []
    
    for img_name in df_all["image"].unique():
        sample_type = identify_sample_type(str(img_name))
        if sample_type == 'mod':
            mod_samples.append(img_name)
        elif sample_type == 'normal':
            normal_samples.append(img_name)
    
    if len(mod_samples) == 0 or len(normal_samples) == 0:
        print(f"❌ Не найдены образцы: mod={len(mod_samples)}, normal={len(normal_samples)}")
        return False
    
    # Пересчитываем метрики
    print(f"\nПересчет метрик для эксперимента: {experiment_dir.name}")
    print(f"  Признаков: {len(features)}")
    print(f"  Mod образцов: {len(mod_samples)}")
    print(f"  Normal образцов: {len(normal_samples)}")
    
    metrics = evaluate_feature_set(
        df_all,
        features,
        mod_samples,
        normal_samples
    )
    
    new_score = metrics['score']
    
    print(f"\nСтарый score: {old_score:.6f}")
    print(f"Новый score:  {new_score:.6f}")
    print(f"Разница:      {abs(new_score - old_score):.6f}")
    
    if abs(new_score - old_score) < 1e-6:
        print(f"\n✅ Score идентичны - обновление не требуется")
        return True
    
    # Обновляем метрики в конфигурации
    config['metrics'] = {
        'score': float(metrics['score']),
        'separation': float(metrics['separation']),
        'mean_pc1_norm_mod': float(metrics['mean_pc1_norm_mod']),
        'explained_variance': float(metrics['explained_variance']),
        'mean_pc1_mod': float(metrics['mean_pc1_mod']),
        'mean_pc1_normal': float(metrics['mean_pc1_normal']),
    }
    
    if not dry_run:
        # Сохраняем обновленную конфигурацию
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\n✅ JSON файл обновлен: {json_file}")
    else:
        print(f"\n⚠️ DRY RUN: JSON файл НЕ был обновлен")
    
    return True


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Обновление score в JSON файлах экспериментов")
    parser.add_argument("experiment_name", nargs="?", default="feature_selection_all_methods",
                       help="Имя эксперимента")
    parser.add_argument("--dry-run", action="store_true",
                       help="Только показать изменения, не сохранять")
    
    args = parser.parse_args()
    
    experiments_dir = Path("experiments")
    experiment_dir = experiments_dir / args.experiment_name
    
    if not experiment_dir.exists():
        print(f"❌ Эксперимент не найден: {experiment_dir}")
        sys.exit(1)
    
    print("="*80)
    print("ОБНОВЛЕНИЕ SCORE В ЭКСПЕРИМЕНТЕ")
    print("="*80)
    
    if args.dry_run:
        print("⚠️ DRY RUN MODE - изменения не будут сохранены")
    
    success = update_experiment_score(experiment_dir, dry_run=args.dry_run)
    
    if success:
        print(f"\n{'='*80}")
        print("ОБНОВЛЕНИЕ ЗАВЕРШЕНО")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("ОБНОВЛЕНИЕ НЕ УДАЛОСЬ")
        print(f"{'='*80}")
        sys.exit(1)


if __name__ == "__main__":
    main()

