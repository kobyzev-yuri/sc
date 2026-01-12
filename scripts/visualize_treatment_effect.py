#!/usr/bin/env python3
"""
Скрипт для визуализации изменений признаков между двумя образцами (до/после лечения).

Создает "елочку" - визуализацию изменений признаков, отсортированных по убыванию
абсолютной величины изменения.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Добавляем путь к модулям scale
sys.path.insert(0, str(Path(__file__).parent.parent))

from scale import aggregate, domain, spectral_analysis, pca_scoring
from scale.dashboard_experiment_selector import load_experiment_features, list_available_experiments
import json
import pickle


def load_analyzer_from_best_experiment(experiments_dir: Path = None):
    """
    Загружает обученный SpectralAnalyzer из лучшего эксперимента (как в dashboard).
    Использует тот же эксперимент, что указан в feature_selection_config_relative.json.
    
    Args:
        experiments_dir: Директория с экспериментами
        
    Returns:
        SpectralAnalyzer или None если не найден
    """
    if experiments_dir is None:
        experiments_dir = Path(__file__).parent.parent / "experiments"
    
    # Сначала пробуем загрузить из эксперимента, указанного в конфиге
    config_path = Path(__file__).parent.parent / "scale" / "cfg" / "feature_selection_config_relative.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                source_exp = config.get('source_experiment', '')
                if source_exp:
                    # Извлекаем имя эксперимента из пути
                    exp_name = source_exp.split('/')[-1] if '/' in source_exp else source_exp
                    exp_path = experiments_dir / exp_name
                    analyzer_path = exp_path / "spectral_analyzer.pkl"
                    if analyzer_path.exists():
                        analyzer = spectral_analysis.SpectralAnalyzer()
                        analyzer.load(analyzer_path)
                        print(f"✅ Загружен analyzer из эксперимента из конфига: '{exp_name}'")
                        print(f"   Путь: {analyzer_path}")
                        print(f"   Признаков в модели: {len(analyzer.feature_columns) if analyzer.feature_columns else 0}")
                        return analyzer
        except Exception as e:
            print(f"⚠️  Не удалось загрузить из конфига: {e}")
    
    # Если не нашли в конфиге, ищем analyzer в любом эксперименте (как fallback)
    try:
        # Сначала пробуем лучшие эксперименты
        experiments = list_available_experiments(experiments_dir=experiments_dir, top_n=10)
        for exp in experiments:
            exp_path = Path(exp['path'])
            analyzer_path = exp_path / "spectral_analyzer.pkl"
            if analyzer_path.exists():
                analyzer = spectral_analysis.SpectralAnalyzer()
                analyzer.load(analyzer_path)
                print(f"✅ Загружен analyzer из эксперимента '{exp['name']}'")
                print(f"   Путь: {analyzer_path}")
                print(f"   Признаков в модели: {len(analyzer.feature_columns) if analyzer.feature_columns else 0}")
                return analyzer
        
        # Если не нашли в лучших, ищем во всех экспериментах
        print("   Ищем analyzer во всех экспериментах...")
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                analyzer_path = exp_dir / "spectral_analyzer.pkl"
                if analyzer_path.exists():
                    analyzer = spectral_analysis.SpectralAnalyzer()
                    analyzer.load(analyzer_path)
                    print(f"✅ Загружен analyzer из эксперимента '{exp_dir.name}'")
                    print(f"   Путь: {analyzer_path}")
                    print(f"   Признаков в модели: {len(analyzer.feature_columns) if analyzer.feature_columns else 0}")
                    return analyzer
    except Exception as e:
        print(f"⚠️  Не удалось загрузить analyzer: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def load_best_experiment_features(experiments_dir: Path = None) -> list:
    """
    Загружает признаки из лучшего эксперимента.
    
    Args:
        experiments_dir: Директория с экспериментами
        
    Returns:
        Список признаков или None если не найден
    """
    if experiments_dir is None:
        experiments_dir = Path(__file__).parent.parent / "experiments"
    
    # Пробуем загрузить из конфига
    config_path = Path(__file__).parent.parent / "scale" / "cfg" / "feature_selection_config_relative.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                features = config.get('selected_features', [])
                if features:
                    print(f"✅ Загружены признаки из конфига: {len(features)} признаков")
                    return features
        except Exception as e:
            print(f"⚠️  Не удалось загрузить из конфига: {e}")
    
    # Пробуем загрузить из лучшего эксперимента
    try:
        experiments = list_available_experiments(experiments_dir=experiments_dir, top_n=1)
        if experiments:
            best_exp = experiments[0]
            features = best_exp.get('features', [])
            if features:
                print(f"✅ Загружены признаки из лучшего эксперимента '{best_exp['name']}': {len(features)} признаков")
                return features
    except Exception as e:
        print(f"⚠️  Не удалось загрузить из эксперимента: {e}")
    
    print("⚠️  Не найдены признаки из эксперимента, используем все признаки")
    return None


def load_and_aggregate_sample(json_path: str, use_relative: bool = True, selected_features: list = None) -> pd.DataFrame:
    """
    Загружает и агрегирует образец из JSON файла.
    
    Args:
        json_path: Путь к JSON файлу
        use_relative: Использовать относительные признаки
        
    Returns:
        Series с признаками
    """
    # Загружаем предсказания
    predictions = domain.predictions_from_json(json_path)
    image_name = Path(json_path).stem
    
    # Агрегируем в признаки
    stats = aggregate.aggregate_predictions_from_dict(predictions, image_name)
    df = pd.DataFrame([stats])
    
    # Создаем относительные признаки если нужно
    if use_relative:
        df = aggregate.create_relative_features(df)
    
    # Убираем колонку image
    if 'image' in df.columns:
        df = df.drop(columns=['image'])
    
    # Фильтруем по выбранным признакам если указаны
    if selected_features:
        # Оставляем только те признаки, которые есть и в данных, и в списке
        available_features = [f for f in selected_features if f in df.columns]
        if available_features:
            df = df[available_features]
        else:
            print(f"⚠️  Ни один из выбранных признаков не найден в данных")
    
    return df


def create_treatment_comparison_plot(
    before_path: str,
    after_path: str,
    output_path: str = "treatment_effect_comparison.png",
    use_relative: bool = True,
    top_n: int = 30,
    use_best_experiment_features: bool = True
):
    """
    Создает визуализацию "елочки" изменений признаков.
    
    Args:
        before_path: Путь к JSON файлу "до" лечения
        after_path: Путь к JSON файлу "после" лечения
        output_path: Путь для сохранения картинки
        use_relative: Использовать относительные признаки
        top_n: Количество топ признаков для отображения
    """
    print(f"📊 Загрузка образцов...")
    print(f"   До: {before_path}")
    print(f"   После: {after_path}")
    
    # Загружаем признаки и analyzer из лучшего эксперимента если нужно
    selected_features = None
    analyzer = None
    if use_best_experiment_features:
        print(f"\n🔍 Загрузка признаков из лучшего эксперимента...")
        selected_features = load_best_experiment_features()
        print(f"\n🔍 Загрузка analyzer из лучшего эксперимента (как в dashboard)...")
        base_dir = Path(__file__).parent.parent
        analyzer = load_analyzer_from_best_experiment(experiments_dir=base_dir / "experiments")
        
        if analyzer is None:
            print("⚠️  Analyzer не найден в эксперименте.")
            print("   ⚠️  ВНИМАНИЕ: Для правильного вычисления PC1_spectrum нужен analyzer из эксперимента!")
            print("   ⚠️  Без analyzer значения на шкале будут некорректными.")
            print("   💡 Убедитесь, что в лучшем эксперименте есть файл spectral_analyzer.pkl")
            return None, None
    
    # Загружаем и агрегируем оба образца
    before_df = load_and_aggregate_sample(before_path, use_relative, selected_features)
    after_df = load_and_aggregate_sample(after_path, use_relative, selected_features)
    
    before_features = before_df.iloc[0]
    after_features = after_df.iloc[0]
    
    # Вычисляем PC1_spectrum для обоих образцов используя analyzer из эксперимента (как в dashboard)
    before_spectrum = None
    after_spectrum = None
    before_name = Path(before_path).stem
    after_name = Path(after_path).stem
    
    if analyzer and analyzer.feature_columns:
        try:
            # ВАЖНО: Используем ТОЧНО те же признаки, что были при обучении модели (как в dashboard)
            required_features = analyzer.feature_columns.copy()
            
            # Проверяем наличие всех необходимых признаков
            missing_before = [f for f in required_features if f not in before_df.columns]
            missing_after = [f for f in required_features if f not in after_df.columns]
            
            # Автоматически добавляем недостающие признаки с нулевыми значениями (как в dashboard)
            if missing_before:
                for feat in missing_before:
                    before_df[feat] = 0.0
            if missing_after:
                for feat in missing_after:
                    after_df[feat] = 0.0
            
            # Используем ТОЛЬКО признаки из модели (в том же порядке) - КРИТИЧЕСКИ ВАЖНО!
            before_df_features = before_df[required_features].copy()
            after_df_features = after_df[required_features].copy()
            
            # Вычисляем PC1 для обоих образцов (как в dashboard)
            before_pca = analyzer.transform_pca(before_df_features)
            after_pca = analyzer.transform_pca(after_df_features)
            
            # Вычисляем spectrum (используем уже обученный spectrum из эксперимента)
            before_spectrum_df = analyzer.transform_to_spectrum(before_pca)
            after_spectrum_df = analyzer.transform_to_spectrum(after_pca)
            
            before_spectrum = before_spectrum_df['PC1_spectrum'].iloc[0]
            after_spectrum = after_spectrum_df['PC1_spectrum'].iloc[0]
            
            print(f"\n📊 Значения на шкале (из analyzer лучшего эксперимента):")
            print(f"   {before_name}: PC1_spectrum = {before_spectrum:.4f}")
            print(f"   {after_name}: PC1_spectrum = {after_spectrum:.4f}")
            
            # Определяем направление улучшения
            spectrum_change = after_spectrum - before_spectrum
            if spectrum_change < 0:
                print(f"   ✅ Улучшение: движение к норме (↓ {abs(spectrum_change):.4f})")
                improvement_direction = -1  # Уменьшение spectrum = улучшение
            else:
                print(f"   ❌ Ухудшение: движение к воспалению (↑ {abs(spectrum_change):.4f})")
                improvement_direction = 1  # Увеличение spectrum = ухудшение
        except Exception as e:
            print(f"⚠️  Не удалось вычислить PC1_spectrum: {e}")
            import traceback
            traceback.print_exc()
            analyzer = None
    
    # Вычисляем разницу (после - до)
    diff = after_features - before_features
    
    # Определяем улучшение/ухудшение для каждого признака
    # Если analyzer доступен, используем loadings PCA для определения направления
    feature_improvement = {}
    if analyzer and analyzer.pca is not None and analyzer.feature_columns:
        # Получаем loadings первой компоненты
        pc1_loadings = analyzer.pca.components_[0]
        feature_to_loading = dict(zip(analyzer.feature_columns, pc1_loadings))
        
        # Для каждого признака: если его увеличение увеличивает PC1 (положительный loading),
        # то увеличение признака = движение к воспалению = ухудшение
        # Если его увеличение уменьшает PC1 (отрицательный loading),
        # то увеличение признака = движение к норме = улучшение
        for feat in diff.index:
            if feat in feature_to_loading:
                loading = feature_to_loading[feat]
                feat_change = diff[feat]
                
                # Если loading > 0: увеличение признака → увеличение PC1 → ухудшение
                # Если loading < 0: увеличение признака → уменьшение PC1 → улучшение
                if loading > 0:
                    # Положительный loading: увеличение признака = ухудшение
                    feature_improvement[feat] = -1 if feat_change > 0 else 1
                else:
                    # Отрицательный loading: увеличение признака = улучшение
                    feature_improvement[feat] = 1 if feat_change > 0 else -1
            else:
                # Если признак не в loadings, используем общее направление spectrum
                if before_spectrum is not None and after_spectrum is not None:
                    feature_improvement[feat] = improvement_direction if diff[feat] != 0 else 0
                else:
                    feature_improvement[feat] = 0
    else:
        # Если analyzer недоступен, используем простое правило: уменьшение = улучшение
        # (но это не совсем правильно, нужно знать loadings)
        for feat in diff.index:
            feature_improvement[feat] = 0  # Неизвестно
    
    # Сортируем по абсолютной величине изменения (убывание)
    diff_abs = diff.abs().sort_values(ascending=False)
    
    # Берем топ N признаков
    top_features = diff_abs.head(top_n)
    top_diff = diff[top_features.index]
    
    print(f"\n✅ Найдено {len(diff)} признаков")
    print(f"   Показываем топ {len(top_features)} изменений")
    
    # Определяем улучшение/ухудшение на основе изменений
    if analyzer and before_spectrum is not None and after_spectrum is not None:
        print(f"\n📊 Значения на шкале:")
        print(f"   {before_name}: PC1_spectrum = {before_spectrum:.4f}")
        print(f"   {after_name}: PC1_spectrum = {after_spectrum:.4f}")
        
        spectrum_change = after_spectrum - before_spectrum
        if spectrum_change < 0:
            print(f"   ✅ Общее улучшение: движение к норме (↓ {abs(spectrum_change):.4f})")
        else:
            print(f"   ❌ Общее ухудшение: движение к воспалению (↑ {abs(spectrum_change):.4f})")
    
    print(f"\n🔝 Топ-5 изменений признаков:")
    for i, (feat, change) in enumerate(top_diff.head(5).items(), 1):
        improvement = feature_improvement.get(feat, 0)
        if improvement > 0:
            direction = "→ улучшение (к норме)"
        elif improvement < 0:
            direction = "→ ухудшение (к воспалению)"
        else:
            direction = "→ изменение"
        print(f"   {i}. {feat}: {change:+.4f} {direction}")
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(14, max(10, len(top_features) * 0.4)))
    
    # Цвета на основе направления изменения по шкале:
    # Зеленый = изменение направлено в сторону нормы (улучшение)
    # Красный = изменение направлено в сторону воспаления (ухудшение)
    colors = []
    for feat in top_diff.index:
        improvement = feature_improvement.get(feat, 0)
        if improvement > 0:
            colors.append('#2ecc71')  # Зеленый - движение к норме (улучшение)
        elif improvement < 0:
            colors.append('#e74c3c')  # Красный - движение к воспалению (ухудшение)
        else:
            colors.append('#95a5a6')  # Серый - неизвестно
    
    # Создаем горизонтальный bar chart (елочка)
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_diff.values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Настройка осей
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat.replace('_', ' ').title() for feat in top_features.index], fontsize=9)
    ax.set_xlabel('Изменение признака (после - до)', fontsize=12, fontweight='bold')
    
    # Заголовок с информацией о шкале
    title_parts = [
        f'Визуализация эффекта лечения',
        f'Топ-{len(top_features)} изменений признаков',
        f'ДО: {before_name} | ПОСЛЕ: {after_name}'
    ]
    if before_spectrum is not None and after_spectrum is not None:
        spectrum_change = after_spectrum - before_spectrum
        change_text = f"↓ {abs(spectrum_change):.3f}" if spectrum_change < 0 else f"↑ {abs(spectrum_change):.3f}"
        title_parts.append(f'Шкала: {before_spectrum:.3f} → {after_spectrum:.3f} ({change_text}) | 0=норма, 1=воспаление')
    else:
        title_parts.append('⚠️ Analyzer недоступен - направление улучшения/ухудшения не определено')
    
    ax.set_title('\n'.join(title_parts), fontsize=13, fontweight='bold', pad=20)
    
    # Добавляем вертикальную линию на нуле
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Добавляем значения на барах
    for i, (bar, val) in enumerate(zip(bars, top_diff.values)):
        width = bar.get_width()
        label_x = width + (0.02 * max(abs(top_diff.min()), abs(top_diff.max())) if width >= 0 
                          else -0.02 * max(abs(top_diff.min()), abs(top_diff.max())))
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{val:+.4f}', 
                ha='left' if width >= 0 else 'right',
                va='center', fontsize=8, fontweight='bold')
    
    # Легенда: зеленый = к норме, красный = к воспалению
    green_patch = mpatches.Patch(color='#2ecc71', label='Изменение к норме (улучшение)')
    red_patch = mpatches.Patch(color='#e74c3c', label='Изменение к воспалению (ухудшение)')
    if any(c == '#95a5a6' for c in colors):
        gray_patch = mpatches.Patch(color='#95a5a6', label='Направление неизвестно')
        ax.legend(handles=[green_patch, red_patch, gray_patch], loc='lower right', fontsize=10)
    else:
        ax.legend(handles=[green_patch, red_patch], loc='lower right', fontsize=10)
    
    # Инвертируем ось Y чтобы топ изменения были сверху
    ax.invert_yaxis()
    
    # Улучшаем внешний вид
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Сохраняем полную версию
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Картинка сохранена: {output_path}")
    
    # Создаем дополнительную версию с топ-10 для презентации
    if len(top_features) > 10:
        top_10_features = top_features.head(10)
        top_10_diff = diff[top_10_features.index]
        top_10_colors = colors[:10]
        
        # Создаем новую фигуру для топ-10
        fig_top10, ax_top10 = plt.subplots(figsize=(12, 6))
        
        y_pos_top10 = np.arange(len(top_10_features))
        bars_top10 = ax_top10.barh(y_pos_top10, top_10_diff.values, color=top_10_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax_top10.set_yticks(y_pos_top10)
        ax_top10.set_yticklabels([feat.replace('_', ' ').title() for feat in top_10_features.index], fontsize=10)
        ax_top10.set_xlabel('Изменение признака (после - до)', fontsize=12, fontweight='bold')
        
        # Заголовок с информацией о шкале
        title_parts_top10 = [
            f'Визуализация эффекта лечения (Топ-10)',
            f'ДО: {before_name} | ПОСЛЕ: {after_name}'
        ]
        if before_spectrum is not None and after_spectrum is not None:
            spectrum_change = after_spectrum - before_spectrum
            change_text = f"↓ {abs(spectrum_change):.3f}" if spectrum_change < 0 else f"↑ {abs(spectrum_change):.3f}"
            title_parts_top10.append(f'Шкала: {before_spectrum:.3f} → {after_spectrum:.3f} ({change_text}) | 0=норма, 1=воспаление')
        
        ax_top10.set_title('\n'.join(title_parts_top10), fontsize=13, fontweight='bold', pad=20)
        
        # Добавляем вертикальную линию на нуле
        ax_top10.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Добавляем значения на барах
        for i, (bar, val) in enumerate(zip(bars_top10, top_10_diff.values)):
            width = bar.get_width()
            label_x = width + (0.02 * max(abs(top_10_diff.min()), abs(top_10_diff.max())) if width >= 0 
                          else -0.02 * max(abs(top_10_diff.min()), abs(top_10_diff.max())))
            ax_top10.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'{val:+.4f}', 
                    ha='left' if width >= 0 else 'right',
                    va='center', fontsize=9, fontweight='bold')
        
        # Легенда
        green_patch = mpatches.Patch(color='#2ecc71', label='Изменение к норме (улучшение)')
        red_patch = mpatches.Patch(color='#e74c3c', label='Изменение к воспалению (ухудшение)')
        ax_top10.legend(handles=[green_patch, red_patch], loc='lower right', fontsize=10)
        
        # Инвертируем ось Y
        ax_top10.invert_yaxis()
        
        # Улучшаем внешний вид
        ax_top10.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Сохраняем топ-10 версию
        output_path_top10 = output_path.replace('.png', '_top10.png')
        plt.savefig(output_path_top10, dpi=300, bbox_inches='tight')
        print(f"💾 Картинка (Топ-10) сохранена: {output_path_top10}")
        plt.close(fig_top10)
    
    # Также сохраняем таблицу с изменениями
    comparison_data = {
        'До': before_features[top_features.index],
        'После': after_features[top_features.index],
        'Изменение': top_diff,
        'Абсолютное изменение': top_features
    }
    
    # Добавляем информацию о направлении улучшения/ухудшения если доступна
    if analyzer and analyzer.pca is not None and analyzer.feature_columns:
        pc1_loadings = analyzer.pca.components_[0]
        feature_to_loading = dict(zip(analyzer.feature_columns, pc1_loadings))
        
        loadings_list = []
        direction_list = []
        for feat in top_features.index:
            if feat in feature_to_loading:
                loading = feature_to_loading[feat]
                loadings_list.append(loading)
                # Определяем направление
                feat_change = diff[feat]
                if loading > 0:
                    direction = "ухудшение" if feat_change > 0 else "улучшение"
                else:
                    direction = "улучшение" if feat_change > 0 else "ухудшение"
                direction_list.append(direction)
            else:
                loadings_list.append(None)
                direction_list.append("неизвестно")
        
        comparison_data['PCA_loading'] = loadings_list
        comparison_data['Направление'] = direction_list
    
    comparison_df = pd.DataFrame(comparison_data)
    csv_path = output_path.replace('.png', '_data.csv')
    comparison_df.to_csv(csv_path, index=True)
    print(f"📊 Таблица сохранена: {csv_path}")
    
    # Сохраняем также информацию о значениях на шкале
    if before_spectrum is not None and after_spectrum is not None:
        spectrum_info = {
            'Образец': [before_name, after_name],
            'PC1_spectrum': [before_spectrum, after_spectrum],
            'Изменение_spectrum': [0, after_spectrum - before_spectrum],
            'Направление': ['базовая линия', 'улучшение' if (after_spectrum - before_spectrum) < 0 else 'ухудшение']
        }
        spectrum_df = pd.DataFrame(spectrum_info)
        spectrum_csv_path = output_path.replace('.png', '_spectrum_values.csv')
        spectrum_df.to_csv(spectrum_csv_path, index=False)
        print(f"📊 Значения на шкале сохранены: {spectrum_csv_path}")
    
    return fig, comparison_df


if __name__ == "__main__":
    # Пути к файлам
    base_dir = Path(__file__).parent.parent
    inference_dir = base_dir / "results" / "inference"
    
    before_file = inference_dir / "9_ibd_mod_2mod.json"
    after_file = inference_dir / "9_ibd_mod_9mod.json"
    
    # Проверка существования файлов
    if not before_file.exists():
        print(f"❌ Файл не найден: {before_file}")
        sys.exit(1)
    if not after_file.exists():
        print(f"❌ Файл не найден: {after_file}")
        sys.exit(1)
    
    # Создаем визуализацию
    output_file = base_dir / "treatment_effect_comparison.png"
    
    create_treatment_comparison_plot(
        before_path=str(before_file),
        after_path=str(after_file),
        output_path=str(output_file),
        use_relative=True,  # Используем относительные признаки
        top_n=30,  # Показываем топ-30 изменений
        use_best_experiment_features=True  # Используем только признаки из лучшего эксперимента
    )
    
    print("\n✅ Готово!")

