"""
Экспорт результатов подбора признаков для dashboard и медиков.

Модуль для сохранения результатов автоматизированного подбора признаков
в формате, совместимом с dashboard, и создания отчетов для медиков.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


def export_to_dashboard_config(
    selected_features: List[str],
    output_dir: Path,
    method_name: str,
    metrics: Dict,
    use_relative_features: bool = True,
    description: Optional[str] = None,
) -> Path:
    """
    Экспортирует отобранные признаки в формат конфигурации dashboard.
    
    Args:
        selected_features: Список отобранных признаков
        output_dir: Директория для сохранения
        method_name: Название метода подбора
        metrics: Словарь с метриками качества
        use_relative_features: Использовать относительные признаки (True) или абсолютные (False)
        description: Описание (если None, генерируется автоматически)
        
    Returns:
        Путь к сохраненному конфигурационному файлу
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем имя файла конфигурации
    if use_relative_features:
        config_filename = "feature_selection_config_relative.json"
    else:
        config_filename = "feature_selection_config_absolute.json"
    
    # Путь для dashboard (в директории scale/)
    # Используем родительскую директорию проекта для доступа к scale/
    project_root = Path(__file__).parent.parent
    dashboard_config_path = project_root / "scale" / config_filename
    
    # Путь для сохранения копии (в output_dir)
    backup_config_path = output_dir / config_filename
    
    # Генерируем описание
    if description is None:
        description = (
            f"Признаки отобраны методом '{method_name}'. "
            f"Score: {metrics.get('score', 0):.4f}, "
            f"Separation: {metrics.get('separation', 0):.4f}, "
            f"Mod (норм. PC1): {metrics.get('mean_pc1_norm_mod', 0):.4f}, "
            f"Объясненная дисперсия: {metrics.get('explained_variance', 0):.4f}"
        )
    
    # Формируем конфигурацию
    config = {
        "selected_features": selected_features,
        "description": description,
        "last_updated": datetime.now().isoformat(),
        "method": method_name,
        "metrics": {
            "score": float(metrics.get('score', 0)),
            "separation": float(metrics.get('separation', 0)),
            "mean_pc1_norm_mod": float(metrics.get('mean_pc1_norm_mod', 0)),
            "explained_variance": float(metrics.get('explained_variance', 0)),
            "mean_pc1_mod": float(metrics.get('mean_pc1_mod', 0)),
            "mean_pc1_normal": float(metrics.get('mean_pc1_normal', 0)),
        },
        "n_features": len(selected_features),
    }
    
    # Сохраняем в dashboard директорию
    with open(dashboard_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Сохраняем копию в output_dir
    with open(backup_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Конфигурация сохранена в dashboard: {dashboard_config_path}")
    print(f"✓ Копия сохранена в: {backup_config_path}")
    
    return dashboard_config_path


def create_medical_report(
    results_df: pd.DataFrame,
    output_dir: Path,
    predictions_dir: Optional[Path] = None,
) -> Path:
    """
    Создает отчет для медиков с результатами подбора признаков.
    
    Args:
        results_df: DataFrame с результатами сравнения методов
        output_dir: Директория для сохранения отчета
        predictions_dir: Директория с предсказаниями (для статистики)
        
    Returns:
        Путь к сохраненному отчету
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"medical_report_{timestamp}.md"
    
    # Сортируем результаты по score
    results_sorted = results_df.sort_values('score', ascending=False)
    best_result = results_sorted.iloc[0]
    
    # Формируем отчет
    report_lines = [
        "# Отчет по подбору признаков для медицинской шкалы",
        "",
        f"**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Резюме",
        "",
        f"**Лучший метод:** {best_result['method']}",
        f"**Количество признаков:** {best_result['n_features']}",
        f"**Оценка качества (score):** {best_result['score']:.4f}",
        "",
        "### Ключевые метрики:",
        f"- **Разделение между группами (separation):** {best_result['separation']:.4f}",
        f"  - Чем больше, тем лучше разделение между патологическими и нормальными образцами",
        f"- **Позиционирование mod образцов (норм. PC1):** {best_result['mean_pc1_norm_mod']:.4f}",
        f"  - Цель: близко к 1.0 (патологические образцы должны иметь высокие значения)",
        f"- **Объясненная дисперсия PC1:** {best_result['explained_variance']:.4f}",
        f"  - Доля вариации, объясняемая первой главной компонентой",
        "",
        "---",
        "",
        "## Сравнение методов",
        "",
        "| Метод | Количество признаков | Score | Separation | Mod (норм.) | Объясненная дисперсия |",
        "|-------|---------------------|-------|------------|-------------|----------------------|",
    ]
    
    for _, row in results_sorted.iterrows():
        report_lines.append(
            f"| {row['method']} | {row['n_features']} | {row['score']:.4f} | "
            f"{row['separation']:.4f} | {row['mean_pc1_norm_mod']:.4f} | "
            f"{row['explained_variance']:.4f} |"
        )
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Отобранные признаки (лучший метод)",
        "",
        f"**Метод:** {best_result['method']}",
        f"**Количество признаков:** {best_result['n_features']}",
        "",
        "### Список признаков:",
        "",
    ])
    
    features = best_result['features']
    for i, feat in enumerate(features, 1):
        report_lines.append(f"{i:2d}. {feat}")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Интерпретация результатов",
        "",
        "### Что означает каждая метрика:",
        "",
        "1. **Score (комплексная оценка):**",
        "   - Комбинированная метрика, учитывающая разделение групп, позиционирование патологических образцов и объясненную дисперсию",
        "   - Чем выше, тем лучше",
        "",
        "2. **Separation (разделение):**",
        "   - Разница между средними значениями PC1 для патологических (mod) и нормальных образцов",
        "   - Положительное значение означает, что патологические образцы имеют более высокие значения PC1",
        "   - Цель: > 2.0",
        "",
        "3. **Mod (норм. PC1):**",
        "   - Среднее нормализованное значение PC1 для патологических образцов (шкала 0-1)",
        "   - Значение 0.0 означает минимальное значение PC1, 1.0 - максимальное",
        "   - Цель: > 0.7 (ближе к 1.0)",
        "",
        "4. **Объясненная дисперсия:**",
        "   - Доля общей вариации данных, объясняемая первой главной компонентой (PC1)",
        "   - Показывает, насколько хорошо PC1 описывает различия между образцами",
        "   - Цель: > 0.3 (30%)",
        "",
        "---",
        "",
        "## Рекомендации",
        "",
    ])
    
    # Добавляем рекомендации на основе метрик
    if best_result['mean_pc1_norm_mod'] < 0.7:
        report_lines.append(
            "⚠️ **Позиционирование патологических образцов можно улучшить** "
            f"(текущее значение: {best_result['mean_pc1_norm_mod']:.4f}, цель: > 0.7)"
        )
        report_lines.append("")
    
    if best_result['separation'] < 2.0:
        report_lines.append(
            "⚠️ **Разделение между группами можно улучшить** "
            f"(текущее значение: {best_result['separation']:.4f}, цель: > 2.0)"
        )
        report_lines.append("")
    
    if best_result['explained_variance'] < 0.3:
        report_lines.append(
            "⚠️ **Объясненная дисперсия можно улучшить** "
            f"(текущее значение: {best_result['explained_variance']:.4f}, цель: > 0.3)"
        )
        report_lines.append("")
    
    if best_result['mean_pc1_norm_mod'] >= 0.7 and best_result['separation'] >= 2.0:
        report_lines.append("✅ **Результаты соответствуют целевым метрикам!**")
        report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## Использование результатов",
        "",
        "### В dashboard:",
        "",
        "1. Конфигурация признаков автоматически сохранена в файл:",
        f"   - `scale/feature_selection_config_relative.json` (для относительных признаков)",
        "",
        "2. При следующем запуске dashboard эти признаки будут автоматически загружены",
        "",
        "3. Вы можете вручную изменить набор признаков в dashboard через интерфейс",
        "",
        "### Для дальнейшего анализа:",
        "",
        "1. Используйте отобранные признаки для построения PCA шкалы",
        "2. Проверьте визуализацию результатов в dashboard",
        "3. Валидируйте результаты на известных образцах",
        "",
        "---",
        "",
        f"*Отчет создан автоматически на основе результатов подбора признаков*",
    ])
    
    # Сохраняем отчет
    report_content = "\n".join(report_lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ Медицинский отчет сохранен: {report_path}")
    
    return report_path


def export_complete_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    use_relative_features: bool = True,
    auto_export_to_dashboard: bool = False,  # По умолчанию НЕ экспортируем автоматически
    df_aggregated: Optional[pd.DataFrame] = None,  # Агрегированные данные (абсолютные признаки)
    df_features: Optional[pd.DataFrame] = None,  # Относительные признаки
    df_all_features: Optional[pd.DataFrame] = None,  # Все доступные признаки
) -> Dict[str, Path]:
    """
    Полный экспорт результатов подбора признаков.
    
    Args:
        results_df: DataFrame с результатами сравнения методов
        output_dir: Директория для сохранения
        use_relative_features: Использовать относительные признаки
        auto_export_to_dashboard: Автоматически экспортировать лучший результат в dashboard
        
    Returns:
        Словарь с путями к сохраненным файлам
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    # Сортируем результаты
    results_sorted = results_df.sort_values('score', ascending=False)
    best_result = results_sorted.iloc[0]
    
    # 1. Сохраняем CSV с результатами
    csv_path = output_dir / f"feature_selection_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    saved_files['csv'] = csv_path
    print(f"✓ CSV сохранен: {csv_path}")
    
    # 2. Сохраняем JSON с лучшим результатом
    json_path = output_dir / f"best_features_{timestamp}.json"
    best_config = {
        'method': best_result['method'],
        'selected_features': best_result['features'],
        'metrics': {
            'score': float(best_result['score']),
            'separation': float(best_result['separation']),
            'mean_pc1_norm_mod': float(best_result['mean_pc1_norm_mod']),
            'explained_variance': float(best_result['explained_variance']),
            'mean_pc1_mod': float(best_result['mean_pc1_mod']),
            'mean_pc1_normal': float(best_result['mean_pc1_normal']),
        },
        'timestamp': timestamp,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    saved_files['json'] = json_path
    print(f"✓ JSON сохранен: {json_path}")
    
    # 3. Экспортируем в dashboard конфигурацию (только если явно запрошено)
    if auto_export_to_dashboard:
        dashboard_config_path = export_to_dashboard_config(
            selected_features=best_result['features'],
            output_dir=output_dir,
            method_name=best_result['method'],
            metrics=best_result.to_dict(),
            use_relative_features=use_relative_features,
        )
        saved_files['dashboard_config'] = dashboard_config_path
        print(f"\n⚠️ ВНИМАНИЕ: Конфигурация dashboard была обновлена!")
        print(f"   Если хотите вернуться к предыдущей версии, используйте:")
        print(f"   python3 -m scale.feature_selection_versioning export <experiment_name>")
    else:
        print(f"\n💡 Конфигурация dashboard НЕ была обновлена (для безопасности)")
        print(f"   Чтобы экспортировать этот эксперимент в dashboard, используйте:")
        print(f"   python3 -m scale.feature_selection_versioning export {output_dir.name}")
    
    # 4. Создаем медицинский отчет
    report_path = create_medical_report(
        results_df=results_df,
        output_dir=output_dir,
    )
    saved_files['medical_report'] = report_path
    
    # 5. Сохраняем агрегированные данные (если предоставлены)
    if df_aggregated is not None:
        aggregated_path = output_dir / f"aggregated_data_{timestamp}.csv"
        df_aggregated.to_csv(aggregated_path, index=False)
        saved_files['aggregated_data'] = aggregated_path
        print(f"✓ Агрегированные данные сохранены: {aggregated_path}")
    
    if df_features is not None:
        features_path = output_dir / f"relative_features_{timestamp}.csv"
        df_features.to_csv(features_path, index=False)
        saved_files['relative_features'] = features_path
        print(f"✓ Относительные признаки сохранены: {features_path}")
    
    if df_all_features is not None:
        all_features_path = output_dir / f"all_features_{timestamp}.csv"
        df_all_features.to_csv(all_features_path, index=False)
        saved_files['all_features'] = all_features_path
        print(f"✓ Все доступные признаки сохранены: {all_features_path}")
    
    return saved_files


def export_to_experiment_format(
    selected_features: List[str],
    output_dir: Path,
    method_name: str,
    metrics: Dict,
    df_results: Optional[pd.DataFrame] = None,
    analyzer: Optional[object] = None,
    use_relative_features: bool = True,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Экспортирует результаты подбора признаков в формат experiments для использования в dashboard.
    
    Формат experiments включает:
    - results.csv - DataFrame с результатами спектрального анализа
    - spectral_analyzer.pkl - обученная модель (если предоставлена)
    - metadata.json - метаданные эксперимента
    - best_features_*.json - конфигурация признаков
    
    Args:
        selected_features: Список отобранных признаков
        output_dir: Директория для сохранения эксперимента
        method_name: Название метода подбора
        metrics: Словарь с метриками качества
        df_results: DataFrame с результатами спектрального анализа (опционально)
        analyzer: Обученный SpectralAnalyzer (опционально)
        use_relative_features: Использовать относительные признаки
        metadata: Дополнительные метаданные (опционально)
        
    Returns:
        Путь к директории эксперимента
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Сохраняем конфигурацию признаков в формате best_features_*.json
    json_path = output_dir / f"best_features_{timestamp}.json"
    config = {
        'method': method_name,
        'selected_features': selected_features,
        'metrics': {
            'score': float(metrics.get('score', 0)),
            'separation': float(metrics.get('separation', 0)),
            'mean_pc1_norm_mod': float(metrics.get('mean_pc1_norm_mod', 0)),
            'explained_variance': float(metrics.get('explained_variance', 0)),
            'mean_pc1_mod': float(metrics.get('mean_pc1_mod', 0)),
            'mean_pc1_normal': float(metrics.get('mean_pc1_normal', 0)),
        },
        'timestamp': timestamp,
        'use_relative_features': use_relative_features,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ Конфигурация признаков сохранена: {json_path}")
    
    # 2. Сохраняем результаты спектрального анализа (если предоставлены)
    if df_results is not None:
        csv_path = output_dir / "results.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"✓ Результаты сохранены: {csv_path}")
    
    # 3. Сохраняем модель (если предоставлена)
    if analyzer is not None:
        model_path = output_dir / "spectral_analyzer.pkl"
        analyzer.save(model_path)
        print(f"✓ Модель сохранена: {model_path}")
    
    # 4. Сохраняем метаданные
    if metadata is None:
        metadata = {}
    
    metadata.update({
        "timestamp": datetime.now().isoformat(),
        "method": method_name,
        "n_features": len(selected_features),
        "use_relative_features": use_relative_features,
        "metrics": config['metrics'],
    })
    
    if df_results is not None:
        metadata["n_samples"] = len(df_results)
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ Метаданные сохранены: {metadata_path}")
    
    print(f"\n✅ Эксперимент сохранен в формате experiments: {output_dir}")
    
    return output_dir

