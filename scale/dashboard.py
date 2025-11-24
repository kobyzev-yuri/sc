"""
Веб-интерфейс для анализа патологий и визуализации результатов.

Модуль для создания интерактивного дашборда на Streamlit для:
- Загрузки предсказаний (JSON файлы)
- Агрегации данных и создания признаков
- Построения графиков и визуализации спектра
- Сохранения результатов экспериментов
"""

import sys
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

# Добавляем путь к проекту для импортов
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from scipy import stats

try:
    import streamlit as st
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Для работы без GUI
except ImportError as e:
    raise ImportError(
        f"Требуются зависимости для дашборда."
    ) from e

from scale import aggregate, spectral_analysis, domain, scale_comparison, pca_scoring, preprocessing, eda
from model_development.feature_selection_automated import evaluate_feature_set, identify_sample_type

# Импорт общих функций из dashboard_common для синхронизации с dashboard_minimal.py
from scale.dashboard_common import (
    safe_session_get,
    safe_session_set,
    safe_session_del,
    safe_session_has,
    load_predictions_from_files,
    load_predictions_from_gdrive,
    render_gdrive_upload_section,
    render_gdrive_load_section,
    GDRIVE_ENABLED,
    GCS_ENABLED,
)

# Настройка логирования для отладки
DEBUG_MODE = False  # Отладочные сообщения выводятся в логи Streamlit


def create_experiment_dir(base_dir: Path = Path("experiments")) -> Path:
    """
    Создает директорию для нового эксперимента.

    Args:
        base_dir: Базовая директория для экспериментов

    Returns:
        Путь к директории эксперимента
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"experiment_{timestamp}"
    exp_dir.mkdir(exist_ok=True)

    return exp_dir


def save_experiment(
    exp_dir: Path,
    df: pd.DataFrame,
    analyzer: Optional[spectral_analysis.SpectralAnalyzer] = None,
    metadata: Optional[dict] = None,
    selected_features: Optional[List[str]] = None,
    metrics: Optional[dict] = None,
    use_relative_features: bool = True,
) -> None:
    """
    Сохраняет результаты эксперимента в формате, совместимом с experiments.

    Формат experiments включает:
    - results.csv - DataFrame с результатами спектрального анализа
    - spectral_analyzer.pkl - обученная модель (если предоставлена)
    - metadata.json - метаданные эксперимента
    - best_features_*.json - конфигурация признаков (если есть выбранные признаки)

    Args:
        exp_dir: Директория эксперимента
        df: DataFrame с результатами
        analyzer: Обученный SpectralAnalyzer (опционально)
        metadata: Дополнительные метаданные (опционально)
        selected_features: Список выбранных признаков (опционально)
        metrics: Словарь с метриками качества (опционально)
        use_relative_features: Использовать относительные признаки
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Сохранение DataFrame
    csv_path = exp_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    # Сохранение модели спектрального анализа
    if analyzer is not None:
        model_path = exp_dir / "spectral_analyzer.pkl"
        analyzer.save(model_path)

    # Сохранение конфигурации признаков в формате best_features_*.json (если есть выбранные признаки)
    if selected_features:
        json_path = exp_dir / f"best_features_{timestamp}.json"
        
        # Подготавливаем метрики
        if metrics is None:
            metrics = {}
        
        config = {
            'method': metadata.get('method', 'dashboard_manual') if metadata else 'dashboard_manual',
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

    # Сохранение метаданных
    if metadata is None:
        metadata = {}

    metadata["timestamp"] = datetime.now().isoformat()
    metadata["n_samples"] = len(df)
    
    if selected_features:
        metadata["n_features"] = len(selected_features)
        metadata["selected_features"] = selected_features
        metadata["use_relative_features"] = use_relative_features
    
    if metrics:
        metadata["metrics"] = metrics

    metadata_path = exp_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Регистрируем эксперимент в трекере для отслеживания лучших результатов
    try:
        from model_development.experiment_tracker import ExperimentTracker, register_experiment_from_directory
        
        tracker = ExperimentTracker()
        exp_id = register_experiment_from_directory(
            experiment_dir=exp_dir,
            tracker=tracker,
            train_set=metadata.get("train_set", "results/predictions"),
            aggregation_version=metadata.get("aggregation_version", "current"),
        )
        st.success(f"✓ Эксперимент зарегистрирован в трекере (ID: {exp_id})")
    except Exception as e:
        # Не критично, если трекер недоступен
        pass


def render_dashboard():
    """
    Основная функция для рендеринга дашборда Streamlit.
    """
    st.set_page_config(
        page_title="Анализ патологий WSI",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Анализ патологий Whole Slide Images")
    st.markdown("---")
    
    # Боковая панель для загрузки файлов
    with st.sidebar:
        st.header("📁 Загрузка данных")

        # Выбор источника данных
        # Безопасная проверка session_state для предотвращения ошибки инициализации
        use_dir = safe_session_get("use_directory", False)
        use_exp = safe_session_get("use_experiment", False)
        use_gdrive = safe_session_get("use_gdrive", False)
        
        # Формируем список опций
        data_source_options = ["Использовать директорию", "Использовать данные из эксперимента"]
        # Добавляем отдельные опции для каждого доступного источника
        if GDRIVE_ENABLED:
            data_source_options.append("Google Drive")
        if GCS_ENABLED:
            data_source_options.append("Google Cloud Storage (GCS)")
        
        # Определяем индекс по умолчанию
        use_cloud_storage = safe_session_get("use_cloud_storage", False)
        # Проверяем, установлен ли data_source из session state (после загрузки из GCS/GDrive)
        saved_data_source = safe_session_get("data_source", None)
        
        # КРИТИЧНО: Если есть загруженные данные из cloud storage, приоритетно устанавливаем правильный data_source
        cloud_predictions = safe_session_get("predictions_cloud", None)
        if cloud_predictions and len(cloud_predictions) > 0:
            # Есть загруженные данные из cloud storage - устанавливаем правильный data_source
            if GCS_ENABLED and "Google Cloud Storage (GCS)" in data_source_options:
                saved_data_source = "Google Cloud Storage (GCS)"
                safe_session_set("data_source", saved_data_source)
                logger.debug(f"[DATA_SOURCE] Установлен data_source: {saved_data_source} (есть данные из GCS)")
            elif GDRIVE_ENABLED and "Google Drive" in data_source_options:
                saved_data_source = "Google Drive"
                safe_session_set("data_source", saved_data_source)
                logger.debug(f"[DATA_SOURCE] Установлен data_source: {saved_data_source} (есть данные из GDrive)")
        elif use_cloud_storage:
            # Флаг установлен, но данных может не быть - все равно устанавливаем правильный data_source
            if saved_data_source and saved_data_source in data_source_options:
                # Используем сохраненный data_source
                pass  # уже установлен
            elif GCS_ENABLED and "Google Cloud Storage (GCS)" in data_source_options:
                saved_data_source = "Google Cloud Storage (GCS)"
                safe_session_set("data_source", saved_data_source)
                logger.debug(f"[DATA_SOURCE] Установлен data_source: {saved_data_source} (use_cloud_storage=True)")
            elif GDRIVE_ENABLED and "Google Drive" in data_source_options:
                saved_data_source = "Google Drive"
                safe_session_set("data_source", saved_data_source)
                logger.debug(f"[DATA_SOURCE] Установлен data_source: {saved_data_source} (use_cloud_storage=True)")
        
        if saved_data_source and saved_data_source in data_source_options:
            # Используем сохраненный data_source
            default_index = data_source_options.index(saved_data_source)
        elif use_cloud_storage and GCS_ENABLED and "Google Cloud Storage (GCS)" in data_source_options:
            # Находим индекс для GCS опции
            default_index = data_source_options.index("Google Cloud Storage (GCS)")
        elif use_gdrive and GDRIVE_ENABLED and "Google Drive" in data_source_options:
            # Находим индекс для Google Drive опции
            default_index = data_source_options.index("Google Drive")
        elif use_exp:
            default_index = 1
        elif use_dir:
            default_index = 0
        else:
            default_index = 0
        
        data_source = st.radio(
            "Источник данных",
            data_source_options,
            index=default_index,
            key="data_source_selector"  # Добавляем key для стабильности
        )
        
        # КРИТИЧНО: Сохраняем выбранный data_source в session state ТОЛЬКО если он из правильного списка
        if data_source in data_source_options:
            safe_session_set("data_source", data_source)
            logger.debug(f"[DATA_SOURCE] Сохранен data_source из radio: {data_source}")
        else:
            logger.warning(f"[DATA_SOURCE] Попытка сохранить неправильный data_source: {data_source}, опции: {data_source_options}")
            # Если data_source неправильный, но есть данные из cloud, устанавливаем правильный
            if cloud_predictions and len(cloud_predictions) > 0:
                if GCS_ENABLED and "Google Cloud Storage (GCS)" in data_source_options:
                    safe_session_set("data_source", "Google Cloud Storage (GCS)")
                    logger.debug(f"[DATA_SOURCE] Исправлен data_source на: Google Cloud Storage (GCS)")
                elif GDRIVE_ENABLED and "Google Drive" in data_source_options:
                    safe_session_set("data_source", "Google Drive")
                    logger.debug(f"[DATA_SOURCE] Исправлен data_source на: Google Drive")
        
        use_default_data = (data_source == "Использовать директорию")
        use_experiment_data = (data_source == "Использовать данные из эксперимента")
        use_gdrive_data = (data_source == "Google Drive") and GDRIVE_ENABLED
        use_cloud_storage_data = (data_source == "Google Cloud Storage (GCS)") and GCS_ENABLED
        
        # Поля ввода для Google Drive и GCS в sidebar (левая панель)
        if use_gdrive_data:
            st.markdown("---")
            st.subheader("📥 Google Drive")
            saved_url = safe_session_get("gdrive_load_url", "")
            drive_folder_url = st.text_input(
                "Ссылка на папку Google Drive",
                value=saved_url if saved_url else "",
                placeholder="https://drive.google.com/drive/folders/1ABC123xyz...",
                help="Вставьте ссылку на расшаренную папку с JSON файлами",
                key="gdrive_load_folder_url_sidebar"
            )
            # Сохраняем URL в session state
            if drive_folder_url:
                safe_session_set("gdrive_load_url", drive_folder_url)
            
            # Проверка авторизации Google Drive
            import os
            default_creds_path = os.path.join(os.path.expanduser('~'), '.config', 'gdrive', 'credentials.json')
            creds_path = os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH', default_creds_path)
            from scale.dashboard_common import get_credentials
            credentials = get_credentials(credentials_path=creds_path)
            
            # Показываем кнопку авторизации, если нужно
            if not credentials:
                st.warning("⚠️ Требуется авторизация Google Drive")
                if st.button("🔐 Авторизоваться в Google Drive", key="gdrive_authorize_button_sidebar"):
                    from scale.dashboard_common import authorize_gdrive
                    if authorize_gdrive(creds_path):
                        st.success("✅ Авторизация успешна!")
                        st.rerun()
                    else:
                        st.error("❌ Ошибка авторизации")
        
        if use_cloud_storage_data:
            st.markdown("---")
            st.subheader("📥 Google Cloud Storage")
            bucket_name_default = safe_session_get("gcs_bucket_name", "scalebucket")
            bucket_name = st.text_input(
                "Имя GCS bucket",
                value=bucket_name_default,
                placeholder="scalebucket",
                help="Имя вашего Google Cloud Storage bucket",
                key="gcs_bucket_name_input_sidebar"
            )
            # Сохраняем значение в session state
            if bucket_name:
                safe_session_set("gcs_bucket_name", bucket_name)
            
            # Проверка авторизации GCS
            auth_ok = False
            try:
                from google.cloud import storage
                client = storage.Client()
                try:
                    _ = list(client.list_buckets(max_results=1))
                    auth_ok = True
                except Exception:
                    auth_ok = False
            except Exception:
                auth_ok = False
            
            if not auth_ok:
                st.warning("⚠️ **Требуется авторизация GCS**")
                with st.expander("🔧 Как настроить авторизацию"):
                    st.markdown("""
                    **Вариант 1 (для локальной разработки):**
                    ```bash
                    gcloud auth application-default login
                    ```
                    
                    **Вариант 2 (для Cloud Run / Service Account):**
                    ```bash
                    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
                    ```
                    
                    **Вариант 3 (если используете gcloud):**
                    ```bash
                    gcloud auth login
                    gcloud config set project YOUR_PROJECT_ID
                    ```
                    
                    После настройки перезапустите dashboard.
                    """)
        
        # КРИТИЧНО: Если в session state установлен флаг use_cloud_storage и есть данные,
        # принудительно используем cloud storage независимо от выбора в селекторе
        if safe_session_get("use_cloud_storage", False) and safe_session_get("predictions_cloud", None):
            use_cloud_storage_data = True
            # Принудительно устанавливаем правильный data_source
            if data_source not in ["Google Drive / GCS", "Google Cloud Storage (GCS)"]:
                if GCS_ENABLED:
                    data_source = "Google Cloud Storage (GCS)"
                    safe_session_set("data_source", data_source)
                elif GDRIVE_ENABLED:
                    data_source = "Google Drive / GCS"
                    safe_session_set("data_source", data_source)
        
        # Безопасное сохранение в session_state
        safe_session_set("use_directory", use_default_data)
        safe_session_set("use_experiment", use_experiment_data)
        safe_session_set("use_gdrive", use_gdrive_data)
        safe_session_set("use_cloud_storage", use_cloud_storage_data)

        if use_default_data:
            # Предустановленные директории
            default_dirs = [
                "results/predictions",
                "test/predictions",
                "scale_results/predictions",
            ]
            
            # Поиск существующих директорий
            existing_dirs = []
            for dir_path in default_dirs:
                p = Path(dir_path)
                if p.exists() and list(p.glob("*.json")):
                    json_count = len(list(p.glob("*.json")))
                    existing_dirs.append(f"{dir_path} ({json_count} файлов)")
            
            if existing_dirs:
                # Выбор из существующих директорий
                selected_dir_label = st.selectbox(
                    "Выберите директорию",
                    existing_dirs,
                    index=0
                )
                # Извлекаем путь из строки (убираем " (N файлов)")
                predictions_dir_str = selected_dir_label.split(" (")[0]
            else:
                predictions_dir_str = default_dirs[0]
            
            # Возможность ввести свой путь
            custom_dir = st.text_input(
                "Или введите свой путь к директории",
                value="",
                placeholder="например: my_data/predictions"
            )
            
            if custom_dir:
                predictions_dir_str = custom_dir
            
            # Сохраняем выбранную директорию в session_state
            safe_session_set("predictions_dir", predictions_dir_str)
            predictions_dir = Path(predictions_dir_str)
            
            if predictions_dir.exists():
                json_files = list(predictions_dir.glob("*.json"))
                if json_files:
                    st.success(f"✓ Найдено {len(json_files)} файлов в {predictions_dir}")
                else:
                    st.warning(f"⚠ В директории {predictions_dir} нет JSON файлов")
                    use_default_data = False
            else:
                st.error(f"❌ Директория {predictions_dir} не найдена")
                use_default_data = False
        
        # Выбор эксперимента для загрузки данных
        experiment_data = None
        experiment_name = None
        if use_experiment_data:
            try:
                from scale import dashboard_experiment_selector
            except ImportError:
                from . import dashboard_experiment_selector
            
            # Опция показать все эксперименты
            show_all_experiments = st.checkbox(
                "Показать все эксперименты",
                value=safe_session_get("show_all_experiments", False),
                key="show_all_experiments",
                help="Если включено, показываются все эксперименты из директории, а не только топ-3"
            )
            
            # Опция ввести имя эксперимента вручную
            use_custom_experiment = st.checkbox(
                "Ввести имя эксперимента вручную",
                value=safe_session_get("use_custom_experiment", False),
                key="use_custom_experiment",
                help="Если включено, можно ввести имя любого эксперимента из директории experiments/"
            )
            
            experiment_name = None
            
            # Если включен ручной ввод, показываем текстовое поле
            if use_custom_experiment:
                # Показываем список всех доступных директорий для справки
                all_exp_dirs = sorted([d.name for d in Path("experiments").iterdir() 
                                     if d.is_dir() and d.name != "archive"])
                
                if all_exp_dirs:
                    st.caption(f"Доступные эксперименты: {', '.join(all_exp_dirs[:10])}{'...' if len(all_exp_dirs) > 10 else ''}")
                
                custom_exp_name = st.text_input(
                    "Имя эксперимента",
                    value="",
                    placeholder="например: feature_selection_quick",
                    help="Введите имя директории эксперимента из experiments/"
                )
                
                if custom_exp_name:
                    experiment_dir = Path("experiments") / custom_exp_name
                    if experiment_dir.exists() and experiment_dir.is_dir():
                        experiment_name = custom_exp_name
                    else:
                        st.error(f"❌ Эксперимент '{custom_exp_name}' не найден в директории experiments/")
                        experiment_name = None
            else:
                # Получаем список доступных экспериментов
                if show_all_experiments:
                    # Показываем все эксперименты (без ограничения top_n, но с проверкой данных)
                    experiments = dashboard_experiment_selector.list_available_experiments(use_tracker=True, top_n=None, check_data=True)
                    # Если с проверкой данных мало экспериментов, пробуем без проверки
                    if len(experiments) <= 3:
                        experiments_all = dashboard_experiment_selector.list_available_experiments(use_tracker=True, top_n=None, check_data=False)
                        if len(experiments_all) > len(experiments):
                            st.info(f"💡 Найдено {len(experiments)} экспериментов с данными. Показаны все доступные эксперименты ({len(experiments_all)}).")
                            experiments = experiments_all
                else:
                    # Показываем только топ-3 с лучшим score и с данными
                    experiments = dashboard_experiment_selector.list_available_experiments(use_tracker=True, top_n=3, check_data=True)
                
                if len(experiments) > 0:
                    # Создаем список для выбора с более подробной информацией
                    # Лучшие эксперименты уже отсортированы и будут вверху списка
                    experiment_options = [
                        f"🏆 {exp['name']} (score={exp['score']:.4f}, sep={exp['separation']:.4f}, method={exp['method']})"
                        if exp.get('score', 0) > 0.8 else  # Выделяем лучшие эксперименты
                        f"{exp['name']} (score={exp['score']:.4f}, method={exp['method']})"
                        for exp in experiments
                    ]
                    
                    # Используем key, который зависит от состояния чекбокса и количества экспериментов
                    # Это заставит Streamlit пересоздать виджет при изменении
                    selectbox_key = f"experiment_selectbox_{show_all_experiments}_{len(experiments)}_{hash(tuple(exp['name'] for exp in experiments))}"
                    
                    # Показываем информацию о количестве экспериментов
                    if show_all_experiments:
                        st.caption(f"📊 Показано {len(experiments)} экспериментов")
                    
                    selected_exp_label = st.selectbox(
                        "Выберите эксперимент",
                        experiment_options,
                        index=0,
                        key=selectbox_key,
                        help=f"Выберите эксперимент для загрузки сохраненных данных. {'Показаны все эксперименты' if show_all_experiments else 'Показаны топ-3 эксперимента'}."
                    )
                    
                    # Извлекаем имя эксперимента (убираем эмодзи 🏆 если есть)
                    experiment_name = selected_exp_label.split(" (")[0].replace("🏆 ", "")
                else:
                    # Если экспериментов нет, но пользователь хочет показать все, попробуем без проверки данных
                    if show_all_experiments:
                        st.info("💡 Экспериментов с данными не найдено. Попробуйте включить 'Ввести имя эксперимента вручную' для выбора любого эксперимента.")
            
            if experiment_name:
                
                # Проверяем, изменился ли эксперимент
                previous_experiment = safe_session_get("experiment_name", None)
                experiment_changed = (previous_experiment is not None and previous_experiment != experiment_name)
                
                # Загружаем данные из эксперимента
                experiment_dir = Path("experiments") / experiment_name
                
                # Ищем CSV файлы с данными
                aggregated_files = list(experiment_dir.glob("aggregated_data_*.csv"))
                relative_files = list(experiment_dir.glob("relative_features_*.csv"))
                all_features_files = list(experiment_dir.glob("all_features_*.csv"))
                
                # Проверяем наличие данных (может быть необязательно для ручного ввода)
                has_data = bool(aggregated_files or relative_files or all_features_files)
                
                if has_data:
                    st.success(f"✓ Найдены данные эксперимента: {experiment_name}")
                elif use_custom_experiment:
                    # Для ручного ввода показываем предупреждение, но не блокируем
                    st.warning(f"⚠️ В эксперименте {experiment_name} не найдены сохраненные данные (aggregated_data, relative_features, all_features)")
                    st.info("💡 Эксперимент будет использован, но данные могут быть недоступны")
                
                if has_data or use_custom_experiment:
                    # Если эксперимент изменился, очищаем кэш данных
                    if experiment_changed:
                        # Очищаем кэш данных
                        keys_to_remove = [
                            "df", "df_features", "df_features_full", "df_features_for_selection",
                            "df_all_features", "df_results", "selected_features",
                            "analyzer", "df_spectrum", "comparison"
                        ]
                        for key in keys_to_remove:
                            safe_session_del(key)
                        
                        # Очищаем кэш спектра и GMM
                        try:
                            all_keys = list(st.session_state.keys()) if hasattr(st, 'session_state') else []
                            cache_keys_to_remove = [key for key in all_keys 
                                                    if key.startswith("df_aggregated_") or 
                                                       key.startswith("df_features_full_") or
                                                       key.startswith("predictions_") or
                                                       key.startswith("gmm_quality_")]
                            for key in cache_keys_to_remove:
                                safe_session_del(key)
                        except (RuntimeError, AttributeError):
                            pass
                        
                        st.info(f"🔄 Эксперимент изменен: {previous_experiment} → {experiment_name}. Данные будут перезагружены.")
                    
                    # Сохраняем информацию об эксперименте
                    safe_session_set("experiment_name", experiment_name)
                    safe_session_set("experiment_dir", str(experiment_dir))
                    
                    # Показываем доступные файлы
                    if has_data:
                        with st.expander("📁 Доступные данные эксперимента"):
                            if aggregated_files:
                                st.text(f"✓ Агрегированные данные: {len(aggregated_files)} файл(ов)")
                            if relative_files:
                                st.text(f"✓ Относительные признаки: {len(relative_files)} файл(ов)")
                            if all_features_files:
                                st.text(f"✓ Все доступные признаки: {len(all_features_files)} файл(ов)")
                elif not use_custom_experiment:
                    # Только для автоматически найденных экспериментов показываем ошибку
                    st.warning(f"⚠️ В эксперименте {experiment_name} не найдены сохраненные данные")
                    st.info("💡 Используйте 'Использовать директорию' для загрузки из JSON файлов или включите 'Ввести имя эксперимента вручную'")
                    use_experiment_data = False
                    experiment_name = None
            else:
                st.warning("⚠️ Не найдено экспериментов с сохраненными данными")
                st.info("💡 Сначала запустите подбор признаков для создания эксперимента")
                use_experiment_data = False
        
        elif use_gdrive_data or use_cloud_storage_data:
            # КРИТИЧНО: Проверяем session state ПЕРВЫМ делом
            # Если данные уже загружены, не вызываем render_gdrive_load_section снова
            existing_predictions = safe_session_get("predictions_cloud", None)
            if existing_predictions and len(existing_predictions) > 0:
                st.success(f"✅ Данные уже загружены: {len(existing_predictions)} файлов")
                st.info("💡 Используйте данные для анализа в основной области")
            else:
                # Данных нет - показываем интерфейс загрузки
                # НЕ вызываем здесь - это делается в основной области ниже
                pass
            # Данные уже сохранены в session state внутри render_gdrive_load_section
            # Не нужно обрабатывать их здесь, чтобы избежать дублирования
        
        elif (data_source == "Google Drive" or data_source == "Google Drive / GCS" or data_source == "Google Cloud Storage (GCS)") and not GDRIVE_ENABLED and not GCS_ENABLED:
            st.error("❌ Интеграция с облачным хранилищем недоступна")

        st.markdown("---")

        st.header("⚙️ Настройки")

        with st.expander("ℹ️ Относительные vs Абсолютные признаки"):
            st.markdown("""
            **Относительные признаки (нормализация по Crypts):**
            - ✅ Устраняют влияние размера биоптата
            - ✅ Позволяют сравнивать образцы разного размера
            - ✅ Фокус на плотности/интенсивности патологии
            - ✅ Хорошо для выявления паттернов независимо от размера
            - 📊 Формула: `X_count / Crypts_count`, `X_area / Crypts_area`
            - 📈 **Количество:** 30 признаков (10 классов × 3 типа признаков)
              - `relative_count` - относительное количество
              - `relative_area` - относительная площадь
              - `mean_relative_area` - средняя относительная площадь на объект
            
            **Абсолютные значения:**
            - ✅ Сохраняют информацию о размере биоптата
            - ✅ Важны, когда размер сам по себе значим
            - ✅ Полезны для оценки общей тяжести
            - ✅ Могут лучше работать при большом разбросе размеров
            - 📊 Формула: `X_count`, `X_area` (без нормализации)
            - 📈 **Количество:** 22 признака (11 классов × 2 типа признаков)
              - 10 классов патологий + 1 Crypts (нормализатор)
              - `count` - абсолютное количество объектов
              - `area` - абсолютная площадь
              - Примечание: если в данных есть дополнительные классы (Surface epithelium, Muscularis mucosae и др.),
                абсолютных признаков может быть больше (26-28)
            
            **Почему относительных признаков больше?**
            - Для каждого класса создается 3 относительных признака вместо 2 абсолютных
            - Добавлен `mean_relative_area` - средний размер объекта относительно крипты
            - Исключены: Crypts (нормализатор, используется для нормализации)
            - Структурные элементы (Surface epithelium, Muscularis mucosae) могут быть включены как признаки, если выбраны явно
            
            **Рекомендация:**
            - Начать с относительных признаков (по умолчанию)
            - Попробовать абсолютные, если относительные не дают хорошего разделения
            - Можно сравнить оба подхода экспериментально
            
            📖 Подробнее см. [docs/FEATURES.md](docs/FEATURES.md)
            """)

        use_relative_features = st.checkbox(
            "Использовать относительные признаки", value=True
        )
        
        # Убрали режим выбора признаков - теперь используется интерфейс с чекбоксами
        # Эти переменные больше не используются, но оставляем для совместимости со старым кодом
        use_positive_loadings = False
        min_loading = 0.05
        exclude_paneth = True

        use_spectral_analysis = st.checkbox(
            "Применить спектральный анализ", value=True
        )

        percentile_low = st.slider(
            "Нижний процентиль", 0.0, 10.0, 1.0, 0.1
        )

        percentile_high = st.slider(
            "Верхний процентиль", 90.0, 100.0, 99.0, 0.1
        )

        st.markdown("---")

        st.header("🔮 Инференс")
        
        # Выбор директории для инференса
        # КРИТИЧНО: results/inference должна быть по умолчанию
        inference_default_dirs = [
            "results/inference",  # ПРИОРИТЕТ 1: По умолчанию
            "test/predictions",
            "results/predictions",
        ]
        
        # Получаем сохраненную директорию или используем results/inference по умолчанию
        saved_inference_dir = safe_session_get("inference_dir", "results/inference")
        
        # Проверяем, что сохраненная директория все еще существует
        if saved_inference_dir and Path(saved_inference_dir).exists() and list(Path(saved_inference_dir).glob("*.json")):
            default_inference_dir = saved_inference_dir
        else:
            # Используем results/inference по умолчанию
            default_inference_dir = "results/inference"
        
        # Поиск существующих директорий (в правильном порядке приоритета)
        inference_existing_dirs = []
        inference_dir_paths = []
        for dir_path in inference_default_dirs:
            p = Path(dir_path)
            if p.exists() and list(p.glob("*.json")):
                json_count = len(list(p.glob("*.json")))
                label = f"{dir_path} ({json_count} файлов)"
                inference_existing_dirs.append(label)
                inference_dir_paths.append(dir_path)
        
        # Определяем индекс по умолчанию
        default_index = 0
        if default_inference_dir in inference_dir_paths:
            default_index = inference_dir_paths.index(default_inference_dir)
        elif inference_existing_dirs:
            # Если results/inference нет в списке, но есть другие - используем первый доступный
            # Но приоритет остается за results/inference
            if "results/inference" in [p for p in inference_default_dirs if Path(p).exists()]:
                # Если results/inference существует, но пустая - все равно ставим её первой
                inference_existing_dirs.insert(0, "results/inference (0 файлов)")
                inference_dir_paths.insert(0, "results/inference")
                default_index = 0
        
        if inference_existing_dirs:
            selected_inference_dir_label = st.selectbox(
                "Директория для инференса",
                inference_existing_dirs,
                index=default_index,
                key="inference_dir_selectbox",
                help="Выберите директорию с JSON файлами для инференса. По умолчанию: results/inference"
            )
            inference_dir_str = selected_inference_dir_label.split(" (")[0]
        else:
            # Если нет существующих директорий, используем results/inference по умолчанию
            inference_dir_str = "results/inference"
            st.info(f"💡 Директория по умолчанию: {inference_dir_str} (будет использована при наличии файлов)")
        
        # Возможность ввести свой путь
        custom_inference_dir = st.text_input(
            "Или введите свой путь",
            value="",
            placeholder="например: my_data/inference",
            key="custom_inference_dir"
            )
        
        if custom_inference_dir:
            inference_dir_str = custom_inference_dir
        
        # Сохраняем в session_state (всегда, даже если это значение по умолчанию)
        safe_session_set("inference_dir", inference_dir_str)
        inference_dir = Path(inference_dir_str)
        
        if inference_dir.exists():
            json_files = list(inference_dir.glob("*.json"))
            if json_files:
                st.success(f"✓ Найдено {len(json_files)} файлов для инференса")
            else:
                st.warning(f"⚠ В директории {inference_dir} нет JSON файлов")
        else:
            st.info(f"💡 Директория {inference_dir} будет создана при необходимости")

        st.markdown("---")

        st.header("💾 Эксперименты")
        st.caption("Сохраняет результаты в формате experiments, совместимом с загрузкой через 'Использовать данные из эксперимента'")

        if st.button("Сохранить эксперимент"):
            # Определяем, какие данные сохранять
            df_to_save = None
            if safe_session_has("df_results"):
                df_to_save = safe_session_get("df_results")
            elif safe_session_has("df_spectrum"):
                df_to_save = safe_session_get("df_spectrum")
            else:
                df_to_save = None
            
            if df_to_save is not None:
                exp_dir = create_experiment_dir()
                
                # Получаем выбранные признаки и метрики
                selected_features = safe_session_get("selected_features")
                current_metrics = safe_session_get("current_metrics")
                settings = safe_session_get("settings", {})
                use_relative_features = settings.get("use_relative_features", True)
                
                # Подготавливаем метаданные
                metadata = {"settings": settings}
                
                # Если есть информация об исходном эксперименте, сохраняем её
                if use_experiment_data and safe_session_has("experiment_name"):
                    metadata["source_experiment"] = safe_session_get("experiment_name")
                    metadata["method"] = "experiment_loaded"
                elif selected_features:
                    metadata["method"] = "dashboard_manual"
                    metadata["user_modified"] = True
                    if use_experiment_data and safe_session_has("experiment_name"):
                        metadata["source_experiment"] = safe_session_get("experiment_name")
                
                save_experiment(
                    exp_dir,
                    df_to_save,
                    safe_session_get("analyzer"),
                    metadata,
                    selected_features=selected_features,
                    metrics=current_metrics,
                    use_relative_features=use_relative_features,
                )
                
                # Сохранение результатов сравнения, если они есть
                if safe_session_has("comparison"):
                    try:
                        comparison = safe_session_get("comparison")
                        comparison.save_results(exp_dir / "comparison")
                        st.success(f"Результаты сравнения сохранены в: {exp_dir / 'comparison'}")
                    except Exception as e:
                        st.warning(f"Не удалось сохранить результаты сравнения: {e}")
                
                st.success(f"✅ Эксперимент сохранен: {exp_dir}")
                st.info(f"💡 Эксперимент сохранен в формате, совместимом с загрузкой через 'Использовать данные из эксперимента'")
                
                if selected_features:
                    st.caption(f"📊 Сохранено {len(selected_features)} признаков")
                if current_metrics:
                    st.caption(f"📈 Метрики: Score={current_metrics.get('score', 0):.4f}, Separation={current_metrics.get('separation', 0):.4f}")
            else:
                st.warning("⚠️ Нет данных для сохранения. Выполните анализ данных перед сохранением.")

    # Основная область
    predictions = None
    
    # ОТЛАДКА: Логируем состояние session state в начале рендера
    if DEBUG_MODE or safe_session_get("debug_mode", False):
        logger.debug("Состояние session state:")
        logger.debug(f"  - predictions_cloud: {safe_session_get('predictions_cloud', 'НЕТ')}")
        logger.debug(f"  - use_cloud_storage: {safe_session_get('use_cloud_storage', False)}")
        logger.debug(f"  - data_source: {safe_session_get('data_source', 'НЕТ')}")
        if safe_session_get('predictions_cloud', None):
            logger.debug(f"  - Ключи predictions_cloud: {list(safe_session_get('predictions_cloud', {}).keys())[:3]}")

    # Загрузка данных с кэшированием
    if use_default_data:
        # Получаем выбранную директорию из session_state
        predictions_dir_val = safe_session_get("predictions_dir")
        if predictions_dir_val:
            predictions_dir = Path(predictions_dir_val)
        else:
            # Если не выбрана, используем первую доступную или дефолтную
            default_dirs = ["results/predictions", "test/predictions", "scale_results/predictions"]
            predictions_dir = None
            for dir_path in default_dirs:
                p = Path(dir_path)
                if p.exists() and list(p.glob("*.json")):
                    predictions_dir = p
                    safe_session_set("predictions_dir", str(p))
                    break
            if predictions_dir is None:
                predictions_dir = Path("results/predictions")
                safe_session_set("predictions_dir", "results/predictions")
        
        # Ключ кэша для предиктов
        predictions_cache_key = f"predictions_{predictions_dir}"
        
        # Проверяем кэш
        if (safe_session_has(predictions_cache_key) and 
            safe_session_get("predictions_dir_cache") == str(predictions_dir)):
            predictions = safe_session_get(predictions_cache_key)
        elif predictions_dir.exists():
            json_files = list(predictions_dir.glob("*.json"))
            if json_files:
                with st.spinner(f"Загрузка предсказаний из {predictions_dir}..."):
                    predictions = {}
                    for json_file in json_files:
                        try:
                            preds = domain.predictions_from_json(str(json_file))
                            image_name = json_file.stem
                            predictions[image_name] = preds
                        except Exception as e:
                            st.error(f"Ошибка при загрузке {json_file.name}: {e}")
                    # Сохраняем в кэш
                    safe_session_set(predictions_cache_key, predictions)
                    safe_session_set("predictions_dir_cache", str(predictions_dir))
    
    elif use_gdrive_data:
        # Загрузка из Google Drive (через render_gdrive_load_section)
        # КРИТИЧНО: ВСЕГДА показываем интерфейс загрузки, чтобы можно было переключаться между источниками
        # Сначала проверяем session state для использования уже загруженных данных
        cloud_predictions = safe_session_get("predictions_cloud", None)
        if cloud_predictions and len(cloud_predictions) > 0:
            predictions = cloud_predictions
            # Сохраняем в кэш для использования ниже
            safe_session_set("predictions_cloud_cache", predictions)
            st.info(f"✅ Используются данные из Google Drive: {len(predictions)} файлов")
        
        # ВСЕГДА показываем интерфейс загрузки, чтобы можно было загрузить другие данные или переключиться
        source_info, cloud_predictions_new = render_gdrive_load_section(data_source_selected="Google Drive")
        # Если функция вернула новые данные, используем их (это перезапишет старые)
        if cloud_predictions_new and len(cloud_predictions_new) > 0:
            predictions = cloud_predictions_new
            safe_session_set("predictions_cloud_cache", predictions)
            # Обновляем cloud_predictions для использования ниже
            cloud_predictions = cloud_predictions_new
    
    elif use_cloud_storage_data:
        # Загрузка из Google Drive или GCS (через render_gdrive_load_section)
        # КРИТИЧНО: ВСЕГДА показываем интерфейс загрузки, чтобы можно было переключаться между источниками
        # Сначала проверяем session state для использования уже загруженных данных
        cloud_predictions = safe_session_get("predictions_cloud", None)
        if cloud_predictions and len(cloud_predictions) > 0:
            predictions = cloud_predictions
            # Сохраняем в кэш для использования ниже
            safe_session_set("predictions_cloud_cache", predictions)
            st.info(f"✅ Используются данные из облачного хранилища: {len(predictions)} файлов")
        
        # ВСЕГДА показываем интерфейс загрузки, чтобы можно было загрузить другие данные или переключиться
        source_info, cloud_predictions_new = render_gdrive_load_section()
        # Если функция вернула новые данные, используем их (это перезапишет старые)
        if cloud_predictions_new and len(cloud_predictions_new) > 0:
            predictions = cloud_predictions_new
            safe_session_set("predictions_cloud_cache", predictions)
            # Обновляем cloud_predictions для использования ниже
            cloud_predictions = cloud_predictions_new
    
    # ВАЖНО: Проверяем, есть ли загруженные данные из GCS/GDrive ПЕРЕД проверкой других источников
    # Это нужно для случая, когда данные загружены, но источник не выбран в селекторе
    # Приоритет: если данные загружены из cloud storage, используем их в первую очередь
    logger.debug(f"Проверка predictions перед cloud fallback:")
    logger.debug(f"  - predictions: {predictions is not None}, len: {len(predictions) if predictions else 0}")
    logger.debug(f"  - use_cloud_storage_data: {use_cloud_storage_data}")
    logger.debug(f"  - use_gdrive_data: {use_gdrive_data}")
    
    if not predictions or len(predictions) == 0:
        cloud_predictions = safe_session_get("predictions_cloud", None)
        
        logger.debug(f"Проверка cloud_predictions:")
        logger.debug(f"  - cloud_predictions is not None: {cloud_predictions is not None}")
        if cloud_predictions:
            logger.debug(f"  - Тип cloud_predictions: {type(cloud_predictions)}")
            logger.debug(f"  - Длина cloud_predictions: {len(cloud_predictions) if hasattr(cloud_predictions, '__len__') else 'N/A'}")
            if isinstance(cloud_predictions, dict):
                logger.debug(f"  - Ключи cloud_predictions: {list(cloud_predictions.keys())[:3]}")
        
        if cloud_predictions and len(cloud_predictions) > 0:
            logger.debug("✅ Использую cloud_predictions!")
            predictions = cloud_predictions
            # Убеждаемся, что флаги установлены правильно
            safe_session_set("use_cloud_storage", True)
            # КРИТИЧНО: Устанавливаем правильный data_source в зависимости от доступных опций
            current_data_source = safe_session_get("data_source", None)
            # Проверяем, что data_source правильный, и если нет - устанавливаем правильный
            valid_cloud_sources = []
            if GCS_ENABLED:
                valid_cloud_sources.append("Google Cloud Storage (GCS)")
            if GDRIVE_ENABLED:
                valid_cloud_sources.append("Google Drive")
            
            if current_data_source not in valid_cloud_sources:
                # Устанавливаем правильный data_source
                if GCS_ENABLED:
                    safe_session_set("data_source", "Google Cloud Storage (GCS)")
                    logger.debug(f"[FALLBACK] Установлен data_source: Google Cloud Storage (GCS)")
                elif GDRIVE_ENABLED:
                    safe_session_set("data_source", "Google Drive")
                    logger.debug(f"[FALLBACK] Установлен data_source: Google Drive")
        else:
            logger.debug("❌ cloud_predictions пусто или None")

    # Загрузка данных из эксперимента (если выбран этот источник)
    if use_experiment_data and safe_session_has("experiment_dir"):
        experiment_dir = Path(safe_session_get("experiment_dir"))
        current_experiment = safe_session_get("experiment_name", None)
        
        # Проверяем, изменился ли эксперимент или нужно перезагрузить данные
        previous_experiment = safe_session_get("last_loaded_experiment", None)
        experiment_changed = (previous_experiment is not None and previous_experiment != current_experiment)
        need_reload = (
            experiment_changed or
            not safe_session_has("df") or 
            safe_session_get("df") is None or
            safe_session_get("experiment_name") != current_experiment
        )
        
        # Ищем файлы с данными
        aggregated_files = sorted(experiment_dir.glob("aggregated_data_*.csv"))
        relative_files = sorted(experiment_dir.glob("relative_features_*.csv"))
        all_features_files = sorted(experiment_dir.glob("all_features_*.csv"))
        
        # Если эксперимент изменился, очищаем кэш
        if experiment_changed:
            keys_to_remove = [
                "df", "df_features", "df_features_full", "df_features_for_selection",
                "df_all_features", "df_results", "selected_features",
                "analyzer", "df_spectrum", "comparison", "experiment_config_cache",
                "spectral_settings_key", "features_key", "spectrum_cache_key"  # ОПТИМИЗАЦИЯ: Очищаем ключи кэширования
            ]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Очищаем кэш спектра и GMM
            try:
                all_keys = list(st.session_state.keys()) if hasattr(st, 'session_state') else []
                cache_keys_to_remove = [key for key in all_keys 
                                        if key.startswith("df_aggregated_") or 
                                           key.startswith("df_features_full_") or
                                           key.startswith("predictions_") or
                                           key.startswith("gmm_quality_")]
                for key in cache_keys_to_remove:
                    safe_session_del(key)
            except (RuntimeError, AttributeError):
                pass
            
            # Очищаем ключи для отслеживания типа признаков и загруженного эксперимента
            try:
                all_keys = list(st.session_state.keys()) if hasattr(st, 'session_state') else []
                features_type_keys = [key for key in all_keys 
                                     if key.startswith("features_type_") or key.startswith("loaded_experiment_")]
                for key in features_type_keys:
                    safe_session_del(key)
            except (RuntimeError, AttributeError):
                pass
        
        if (aggregated_files or relative_files or all_features_files) and need_reload:
            with st.spinner(f"Загрузка данных из эксперимента {current_experiment}..."):
                try:
                    if aggregated_files:
                        df_from_experiment = pd.read_csv(aggregated_files[-1])
                    else:
                        df_from_experiment = None
                    
                    if relative_files:
                        df_features_from_experiment = pd.read_csv(relative_files[-1])
                    else:
                        df_features_from_experiment = None
                    
                    if all_features_files:
                        df_all_from_experiment = pd.read_csv(all_features_files[-1])
                    else:
                        df_all_from_experiment = None
                    
                    # Используем данные из эксперимента
                    if df_from_experiment is not None:
                        df = df_from_experiment.copy()
                        
                        # Если есть относительные признаки, используем их
                        if df_features_from_experiment is not None:
                            df_features_full = df_features_from_experiment.copy()
                        else:
                            # Создаем относительные признаки из агрегированных данных
                            df_features_full = aggregate.create_relative_features(df)
                        
                        # Если есть все доступные признаки, используем их
                        if df_all_from_experiment is not None:
                            df_all_features = df_all_from_experiment.copy()
                        else:
                            df_all_features = aggregate.select_all_feature_columns(df_features_full)
                        
                        # КРИТИЧНО: Сохраняем данные из эксперимента в session_state для использования в расчете score
                        safe_session_set("df_from_experiment", df_from_experiment)
                        safe_session_set("df_features_from_experiment", df_features_from_experiment)
                        safe_session_set("df_all_from_experiment", df_all_from_experiment)
                        
                        st.success(f"✓ Данные загружены из эксперимента: {safe_session_get('experiment_name', 'unknown')}")
                        st.info("💡 JSON файлы не загружаются - используются сохраненные агрегированные данные")
                        
                        # Загружаем конфигурацию признаков из эксперимента
                        # Проверяем, изменился ли эксперимент - если да, нужно восстановить оригинальные метрики
                        experiment_changed = (
                            not safe_session_has("last_loaded_experiment") or 
                            safe_session_get("last_loaded_experiment") != current_experiment
                        )
                        
                        try:
                            from scale import dashboard_experiment_selector
                            experiment_config = dashboard_experiment_selector.load_experiment_features(current_experiment)
                            if experiment_config:
                                # Сохраняем полную конфигурацию эксперимента в session_state
                                safe_session_set("experiment_config_cache", experiment_config)
                                
                                if experiment_config.get('features'):
                                    # Загружаем признаки из эксперимента
                                    experiment_features = experiment_config['features']
                                    # Сохраняем все признаки из эксперимента в кэш (включая те, что могут отсутствовать в текущих данных)
                                    experiment_config['selected_features'] = experiment_features
                                    safe_session_set("experiment_config_cache", experiment_config)
                                    
                                    # Фильтруем только существующие признаки для текущей сессии
                                    valid_features = [f for f in experiment_features if f in df_features_full.columns]
                                    if valid_features:
                                        safe_session_set("selected_features", valid_features)
                                        # Помечаем, что признаки были загружены из эксперимента
                                        safe_session_set("features_loaded_from_experiment", True)
                                        
                                        # КРИТИЧНО: Сохраняем хэш признаков для проверки изменений
                                        valid_features_sorted = sorted(valid_features)
                                        features_hash_from_experiment = hash(tuple(valid_features_sorted))
                                        safe_session_set("features_key", features_hash_from_experiment)
                                        
                                        
                                        # Восстанавливаем оригинальные метрики из эксперимента
                                        # Делаем это всегда при загрузке эксперимента (включая повторный выбор)
                                        if experiment_config.get("metrics"):
                                            experiment_metrics = experiment_config.get("metrics", {})
                                            safe_session_set("current_metrics", experiment_metrics)
                                            # Сохраняем признаки из эксперимента для метрик
                                            safe_session_set("metrics_features", valid_features.copy())
                                            # Очищаем флаг features_applied, чтобы не пересчитывать метрики
                                            safe_session_del("features_applied")
                                            
                                            # Если эксперимент изменился или был повторно выбран, показываем сообщение о восстановлении
                                            if experiment_changed:
                                                st.success(f"✅ Восстановлены оригинальные метрики из эксперимента (Score: {experiment_metrics.get('score', 0):.4f})")
                                        
                                        if len(valid_features) < len(experiment_features):
                                            st.warning(f"⚠️ Загружено {len(valid_features)} из {len(experiment_features)} признаков эксперимента. Некоторые признаки отсутствуют в текущих данных.")
                                        else:
                                            st.info(f"💡 Загружены все {len(valid_features)} признаков из эксперимента")
                                    else:
                                        st.error(f"❌ Ни один из {len(experiment_features)} признаков эксперимента не найден в текущих данных!")
                        except Exception as e:
                            st.warning(f"⚠️ Не удалось загрузить конфигурацию из эксперимента: {e}")
                        
                        # Сохраняем информацию о загруженном эксперименте
                        safe_session_set("last_loaded_experiment", current_experiment)
                        safe_session_set("experiment_name", current_experiment)
                        
                        # Обрабатываем данные для дальнейшей работы (аналогично обычной загрузке)
                        if use_relative_features:
                            # Используем полный набор для интерфейса выбора признаков
                            df_features_for_selection = df_features_full.copy()
                            
                            # Применяем выбранные признаки из session_state (если есть) для анализа
                            selected_features_val = safe_session_get("selected_features")
                            if selected_features_val:
                                current_selected = [f for f in selected_features_val if f in df_features_full.columns]
                                if current_selected:
                                    cols_to_keep = ["image"] + current_selected
                                    available_cols = [col for col in cols_to_keep if col in df_features_full.columns]
                                    df_features = df_features_full[available_cols]
                                else:
                                    df_features = df_features_full.copy()
                            else:
                                # Используем старый метод если нет выбранных признаков
                                df_features = aggregate.select_feature_columns(
                                    df_features_full,
                                    use_positive_loadings=use_positive_loadings,
                                    min_loading=min_loading,
                                    exclude_paneth=exclude_paneth
                                )
                        else:
                            # Для абсолютных признаков используем только df
                            df_features = df.copy()
                            # Удаляем относительные признаки, если они случайно попали
                            relative_cols = [col for col in df_features.columns if 'relative' in col.lower()]
                            if relative_cols:
                                df_features = df_features.drop(columns=relative_cols)
                            # Удаляем White space, если он попал
                            white_space_cols = [col for col in df_features.columns if 'white space' in col.lower()]
                            if white_space_cols:
                                df_features = df_features.drop(columns=white_space_cols)
                            
                            # Используем df для интерфейса выбора признаков
                            df_features_for_selection = df_features.copy()
                            
                            # Применяем выбранные признаки из нового интерфейса
                            selected_features_val = safe_session_get("selected_features")
                            if selected_features_val:
                                current_selected = [f for f in selected_features_val if f in df_features.columns]
                                if current_selected:
                                    cols_to_keep = ["image"] + current_selected
                                    available_cols = [col for col in cols_to_keep if col in df_features.columns]
                                    df_features = df_features[available_cols]
                        
                        # Сохраняем в session_state для дальнейшей работы
                        safe_session_set("df_results", df_features)
                        safe_session_set("df", df)
                        safe_session_set("df_features", df_features)
                        safe_session_set("df_features_full", df_features_full if use_relative_features else None)
                        safe_session_set("df_features_for_selection", df_features_for_selection)
                        safe_session_set("df_all_features", df_all_features if 'df_all_features' in locals() else None)
                        safe_session_set("settings", {
                            "use_relative_features": use_relative_features,
                            "use_spectral_analysis": use_spectral_analysis,
                            "percentile_low": percentile_low,
                            "percentile_high": percentile_high,
                        })
                        
                        # Устанавливаем predictions в None, чтобы не загружать JSON
                        predictions = None
                    else:
                        st.error("❌ Не удалось загрузить агрегированные данные из эксперимента")
                        use_experiment_data = False
                except Exception as e:
                    st.error(f"❌ Ошибка при загрузке данных из эксперимента: {e}")
                    use_experiment_data = False
    
    # Используем кэшированные данные если они есть
    # КРИТИЧНО: Проверяем, что данные уже загружены и не нужно их перезагружать
    df = None
    df_features = None
    df_features_full = None
    df_features_for_selection = None
    df_all_features = None
    
    # Проверяем, есть ли данные в кэше
    data_already_loaded = safe_session_has("df") and safe_session_get("df") is not None
    
    if data_already_loaded:
        df = safe_session_get("df", None)
        df_features = safe_session_get("df_features", None)
        df_features_full = safe_session_get("df_features_full", None)
        df_features_for_selection = safe_session_get("df_features_for_selection", None)
        df_all_features = safe_session_get("df_all_features", None)
    
    # Обработка данных с кэшированием (только если данные еще не загружены)
    # КРИТИЧНО: Пропускаем обработку, если данные уже есть в кэше
    if not data_already_loaded and ((not use_experiment_data and predictions and len(predictions) > 0) or (use_experiment_data and not safe_session_has("df"))):
        # Ключ кэша для агрегированных данных
        # КРИТИЧНО: Проверяем, что predictions не None перед использованием
        if predictions is not None and len(predictions) > 0:
            df_cache_key = f"df_aggregated_{hash(str(sorted(predictions.keys())))}"
        else:
            # Если predictions None (данные из эксперимента), используем фиксированный ключ
            df_cache_key = f"df_aggregated_experiment_{safe_session_get('experiment_name', 'unknown')}"
        
        # Проверяем кэш агрегированных данных
        if safe_session_has(df_cache_key):
            df = safe_session_get(df_cache_key)
        else:
            # Проверяем, что predictions не None перед использованием
            if predictions is not None and len(predictions) > 0:
                st.success(f"Загружено {len(predictions)} файлов")
            else:
                # Данные из эксперимента - используем уже загруженные данные
                df = safe_session_get("df")
                if df is None:
                    st.error("❌ Данные не найдены. Загрузите данные или выберите эксперимент.")
                    return
                # Сохраняем в кэш
                safe_session_set(df_cache_key, df)
                return
            # Агрегация данных
            with st.spinner("Агрегация данных..."):
                rows = []

                for image_name, preds in predictions.items():
                    pred_stats = aggregate.aggregate_predictions_from_dict(
                        preds, image_name
                    )
                    rows.append(pred_stats)

                df = pd.DataFrame(rows)
                # Сохраняем в кэш
                safe_session_set(df_cache_key, df)

        # Кэширование df_features_full (только если не используем данные из эксперимента)
        # Создаем признаки только если они еще не созданы
        if not use_experiment_data and use_relative_features and df_features_full is None:
            # Ключ кэша для полного набора признаков
            df_features_full_cache_key = f"df_features_full_{df_cache_key}_{use_relative_features}"
            
            if safe_session_has(df_features_full_cache_key):
                df_features_full = safe_session_get(df_features_full_cache_key)
            else:
                # Создаем полный набор относительных признаков
                df_features_full = aggregate.create_relative_features(df)
                # Сохраняем в кэш
                safe_session_set(df_features_full_cache_key, df_features_full)
            
            # Используем полный набор для интерфейса выбора признаков
            df_features_for_selection = df_features_full.copy()
            
            # Применяем выбранные признаки из session_state (если есть) для анализа
            selected_features_val = safe_session_get("selected_features")
            if selected_features_val:
                current_selected = [f for f in selected_features_val if f in df_features_full.columns]
                if current_selected:
                    cols_to_keep = ["image"] + current_selected
                    available_cols = [col for col in cols_to_keep if col in df_features_full.columns]
                    df_features = df_features_full[available_cols]
                else:
                    df_features = df_features_full.copy()
            else:
                # Используем старый метод если нет выбранных признаков
                df_features = aggregate.select_feature_columns(
                    df_features_full,
                    use_positive_loadings=use_positive_loadings,
                    min_loading=min_loading,
                    exclude_paneth=exclude_paneth
                )
        else:
            # Для абсолютных признаков используем только df, но убеждаемся, что нет относительных признаков
            df_features = df.copy()
            # Удаляем относительные признаки, если они случайно попали (из предыдущего анализа)
            relative_cols = [col for col in df_features.columns if 'relative' in col.lower()]
            if relative_cols:
                df_features = df_features.drop(columns=relative_cols)
            # Удаляем White space, если он попал (служебный класс)
            white_space_cols = [col for col in df_features.columns if 'white space' in col.lower()]
            if white_space_cols:
                df_features = df_features.drop(columns=white_space_cols)
            # Crypts остается в абсолютных признаках (он является признаком)
            # Crypts исключается только из относительных признаков, так как используется как нормализатор
            
            # Используем df для интерфейса выбора признаков (без относительных признаков)
            df_features_for_selection = df_features.copy()
            
            # Применяем выбранные признаки из нового интерфейса
            selected_features_val = safe_session_get("selected_features")
            if selected_features_val:
                current_selected = [f for f in selected_features_val if f in df_features.columns]
                if current_selected:
                    cols_to_keep = ["image"] + current_selected
                    available_cols = [col for col in cols_to_keep if col in df_features.columns]
                    df_features = df_features[available_cols]

        # Сохраняем в session_state только если еще не сохранено (для данных из эксперимента уже сохранено)
        if not safe_session_has("df_results") or not use_experiment_data:
            safe_session_set("df_results", df_features)
            safe_session_set("df", df if 'df' in locals() else None)
            safe_session_set("df_features", df_features if 'df_features' in locals() else None)
            safe_session_set("df_features_full", df_features_full if use_relative_features and 'df_features_full' in locals() else None)
            safe_session_set("df_features_for_selection", df_features_for_selection if 'df_features_for_selection' in locals() else (df_features_full.copy() if use_relative_features and 'df_features_full' in locals() else df_features.copy() if 'df_features' in locals() else None))
            safe_session_set("settings", {
                "use_relative_features": use_relative_features,
                "use_spectral_analysis": use_spectral_analysis,
                "percentile_low": percentile_low,
                "percentile_high": percentile_high,
            })
        
        # Восстанавливаем переменные из session_state для использования во вкладках (только если они еще не загружены)
        if df is None:
            df = safe_session_get("df", None)
        if df_features is None:
            df_features = safe_session_get("df_features", None)
        if df_features_full is None:
            df_features_full = safe_session_get("df_features_full", None)
        if df_features_for_selection is None:
            df_features_for_selection = safe_session_get("df_features_for_selection", None)
    
    # Проверяем, есть ли данные для отображения вкладок
    # Данные могут быть либо из predictions, либо из эксперимента
    has_data = False
    if use_experiment_data:
        # Для данных из эксперимента проверяем session_state
        has_data = (safe_session_has("df") and safe_session_get("df") is not None) or \
                   (safe_session_has("df_features") and safe_session_get("df_features") is not None)
    else:
        # Для обычной загрузки проверяем predictions или session_state
        # КРИТИЧНО: Также проверяем cloud_predictions, если они есть в session state
        cloud_predictions_check = safe_session_get("predictions_cloud", None)
        has_data = (predictions is not None and len(predictions) > 0) or \
                   (cloud_predictions_check is not None and len(cloud_predictions_check) > 0) or \
                   (safe_session_has("df") and safe_session_get("df") is not None)
        
        # ОТЛАДКА: Логируем проверку has_data
        logger.debug(f"Проверка has_data:")
        logger.debug(f"  - predictions is not None: {predictions is not None}")
        logger.debug(f"  - len(predictions): {len(predictions) if predictions else 0}")
        logger.debug(f"  - cloud_predictions_check: {cloud_predictions_check is not None}, len: {len(cloud_predictions_check) if cloud_predictions_check else 0}")
        logger.debug(f"  - safe_session_has('df'): {safe_session_has('df')}")
        logger.debug(f"  - has_data: {has_data}")
        logger.debug(f"  - use_cloud_storage_data: {use_cloud_storage_data}")
        logger.debug(f"  - use_gdrive_data: {use_gdrive_data}")
        logger.debug(f"  - data_source: {data_source}")
    
    # Создаем вкладки только если есть данные
    if has_data:
        # Восстанавливаем переменные из session_state для использования во вкладках (только если они еще не загружены)
        if df is None:
            df = safe_session_get("df", None)
        if df_features is None:
            df_features = safe_session_get("df_features", None)
        if df_features_full is None:
            df_features_full = safe_session_get("df_features_full", None)
        if df_features_for_selection is None:
            df_features_for_selection = safe_session_get("df_features_for_selection", None)

        # Вкладки для визуализации
        tab_names = [
            "🎯 Выбор признаков",
            "📊 Данные",
            "📈 Распределения",
            "🔬 Спектральный анализ",
            "🔍 Анализ образцов",
            "📋 Статистика",
            "🔮 Инференс"
        ]
        
        tabs = st.tabs(tab_names)
        tab_features, tab1, tab2, tab3, tab4, tab5, tab_inference = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4], tabs[5], tabs[6]
        
        # ПРИМЕЧАНИЕ: Streamlit не сохраняет активную вкладку при rerun от виджетов
        # Это известное ограничение Streamlit - при rerun от selectbox/checkbox активная вкладка может сброситься
        # К сожалению, нет прямого способа программно сохранить активную вкладку
        # Решение: убедиться, что выбор образца не вызывает ненужных rerun
        
        # Проблема: Streamlit не предоставляет способ узнать, какая вкладка активна
        # Но мы можем использовать query_params для сохранения активной вкладки
        # Или просто убедиться, что выбор образца не вызывает ненужных rerun

        with tab1:
            st.header("Загруженные данные")
            
            # Пояснение для relative_count
            if use_relative_features:
                with st.expander("ℹ️ Пояснение к relative_count и relative_area"):
                    st.markdown("""
                    **Relative Count (относительное количество):**
                    - Каждое значение = `X_count / Crypts_count`
                    - Это отношение количества объектов типа X к количеству крипт
                    - **Сумма по строке НЕ равна 1**, так как это независимые отношения каждого признака к Crypts
                    
                    **Пример для Count:**
                    - Если Mild_count = 10, Dysplasia_count = 5, Crypts_count = 100
                    - То Mild_relative_count = 0.1, Dysplasia_relative_count = 0.05
                    - Сумма = 0.15 (не 1!)
                    
                    **Relative Area (относительная площадь):**
                    - Аналогично: `X_area / Crypts_area`
                    - Отношение площади объектов типа X к площади крипт
                    - **Сумма по строке ТАКЖЕ НЕ равна 1**, по той же причине - это независимые отношения
                    
                    **Пример для Area:**
                    - Если Mild_area = 1000, Dysplasia_area = 500, Crypts_area = 10000
                    - То Mild_relative_area = 0.1, Dysplasia_relative_area = 0.05
                    - Сумма = 0.15 (не 1!)
                    
                    **Mean Relative Area:**
                    - Средняя относительная площадь на один объект
                    - Формула: `mean_relative_area = relative_area / count`
                    - Где `relative_area = area / Crypts_area`
                    - Итоговая формула: `mean_relative_area = (area / Crypts_area) / count = area / (count * Crypts_area)`
                    - Это средний размер одного объекта типа X относительно размера крипты
                    
                    **Пример:**
                    - Если Dysplasia_area = 1000, Dysplasia_count = 10, Crypts_area = 10000
                    - То Dysplasia_relative_area = 1000 / 10000 = 0.1
                    - И Dysplasia_mean_relative_area = 0.1 / 10 = 0.01
                    - Это означает, что средний размер одной дисплазии составляет 1% от размера крипты
                    """)
            
            st.dataframe(df_features, use_container_width=True)

            # Скачивание CSV
            csv = df_features.to_csv(index=False)
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        with tab_features:
            st.header("🎯 Выбор признаков для анализа")
            st.markdown("Выберите признаки для построения шкалы патологии. PCA автоматически пересчитывается при загрузке данных. Кнопка 'Применить признаки' появится только при изменении признаков.")
            
            # Расчет текущего score для сравнения с экспериментальным
            # Пересчет score
            current_selected_for_score = safe_session_get("selected_features", [])
            if current_selected_for_score:
                # Используем данные из эксперимента, если они загружены
                # Иначе используем данные из загруженных файлов
                df_features_for_score = None
                if use_experiment_data:
                    # Пробуем получить данные из эксперимента
                    df_all_from_experiment = safe_session_get("df_all_from_experiment", None)
                    df_features_from_experiment = safe_session_get("df_features_from_experiment", None)
                    
                    if df_all_from_experiment is not None:
                        # Используем all_features из эксперимента (это то, что использовалось в эксперименте)
                        df_features_for_score = df_all_from_experiment.copy()
                    elif df_features_from_experiment is not None:
                        # Используем relative_features из эксперимента
                        df_features_for_score = df_features_from_experiment.copy()
                
                # Если данные из эксперимента не найдены, используем обычные данные
                if df_features_for_score is None:
                    df_features_for_score = safe_session_get("df_features", None)
                
                if df_features_for_score is not None and len(df_features_for_score) > 0 and len(current_selected_for_score) > 0:
                    # Определяем mod и normal образцы
                    mod_samples_for_score = []
                    normal_samples_for_score = []
                    
                    if "image" in df_features_for_score.columns:
                        for img_name in df_features_for_score["image"].unique():
                            sample_type = identify_sample_type(str(img_name))
                            if sample_type == 'mod':
                                mod_samples_for_score.append(img_name)
                            elif sample_type == 'normal':
                                normal_samples_for_score.append(img_name)
                    
                    # Вычисляем score если есть и mod, и normal образцы
                    if len(mod_samples_for_score) > 0 and len(normal_samples_for_score) > 0:
                        try:
                            feature_cols_for_score = [col for col in current_selected_for_score if col in df_features_for_score.columns]
                            if len(feature_cols_for_score) > 0:
                                sorted_feature_cols_for_score = sorted(feature_cols_for_score)
                                current_score_metrics = evaluate_feature_set(
                                    df_features_for_score,
                                    sorted_feature_cols_for_score,
                                    mod_samples_for_score,
                                    normal_samples_for_score
                                )
                                
                                current_score = current_score_metrics.get('score', None)
                                
                                # Показываем текущий score и сравниваем с экспериментальным
                                if current_score is not None and current_score != -np.inf:
                                    # Получаем экспериментальный score если есть
                                    experiment_score = None
                                    experiment_metrics_full = None
                                    if use_experiment_data and safe_session_has("experiment_config_cache"):
                                        experiment_config = safe_session_get("experiment_config_cache")
                                        if experiment_config and experiment_config.get("metrics"):
                                            experiment_metrics_full = experiment_config.get("metrics", {})
                                            experiment_score = experiment_metrics_full.get('score', None)
                                    
                                    # Отображаем score для сравнения
                                    score_col1, score_col2 = st.columns(2)
                                    with score_col1:
                                        st.metric("📊 Текущий Score", f"{current_score:.4f}")
                                    with score_col2:
                                        if experiment_score is not None:
                                            diff = current_score - experiment_score
                                            diff_color = "normal" if abs(diff) < 0.0001 else "inverse"
                                            st.metric(
                                                "📈 Экспериментальный Score", 
                                                f"{experiment_score:.4f}",
                                                delta=f"{diff:+.4f}" if abs(diff) >= 0.0001 else "0.0000",
                                                delta_color=diff_color
                                            )
                                        else:
                                            st.metric("📈 Экспериментальный Score", "—")
                                    
                                    # Детальное сравнение компонентов score
                                    if experiment_score is not None and abs(current_score - experiment_score) >= 0.0001:
                                        st.warning(f"⚠️ Разница между текущим и экспериментальным score: {abs(current_score - experiment_score):.4f}")
                                        
                                        # Показываем детальное сравнение компонентов
                                        with st.expander("🔍 Детальное сравнение компонентов score", expanded=True):
                                            current_separation = current_score_metrics.get('separation', 0)
                                            current_mod_norm = current_score_metrics.get('mean_pc1_norm_mod', 0)
                                            current_variance = current_score_metrics.get('explained_variance', 0)
                                            
                                            exp_separation = experiment_metrics_full.get('separation', 0) if experiment_metrics_full else 0
                                            exp_mod_norm = experiment_metrics_full.get('mean_pc1_norm_mod', 0) if experiment_metrics_full else 0
                                            exp_variance = experiment_metrics_full.get('explained_variance', 0) if experiment_metrics_full else 0
                                            
                                            st.markdown("**Компоненты score:**")
                                            st.markdown(f"- Separation: текущий = {current_separation:.6f}, экспериментальный = {exp_separation:.6f}, разница = {current_separation - exp_separation:+.6f}")
                                            st.markdown(f"- Mod (норм. PC1): текущий = {current_mod_norm:.6f}, экспериментальный = {exp_mod_norm:.6f}, разница = {current_mod_norm - exp_mod_norm:+.6f}")
                                            st.markdown(f"- Explained variance: текущий = {current_variance:.6f}, экспериментальный = {exp_variance:.6f}, разница = {current_variance - exp_variance:+.6f}")
                                            
                                            # Проверяем признаки
                                            experiment_features = None
                                            if experiment_config and experiment_config.get('features'):
                                                experiment_features = sorted(experiment_config.get('features', []))
                                            current_features_sorted = sorted(feature_cols_for_score)
                                            
                                            st.markdown("**Признаки:**")
                                            if experiment_features:
                                                if current_features_sorted == experiment_features:
                                                    st.success(f"✅ Признаки идентичны ({len(current_features_sorted)} признаков)")
                                                else:
                                                    st.error(f"❌ Признаки различаются!")
                                                    st.markdown(f"- Текущих признаков: {len(current_features_sorted)}")
                                                    st.markdown(f"- Экспериментальных признаков: {len(experiment_features)}")
                                                    missing = set(experiment_features) - set(current_features_sorted)
                                                    extra = set(current_features_sorted) - set(experiment_features)
                                                    if missing:
                                                        st.warning(f"⚠️ Отсутствуют в текущих: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
                                                    if extra:
                                                        st.warning(f"⚠️ Лишние в текущих: {list(extra)[:5]}{'...' if len(extra) > 5 else ''}")
                                            
                                            # Проверяем образцы
                                            st.markdown("**Образцы:**")
                                            st.markdown(f"- Mod образцов: текущий = {len(mod_samples_for_score)}, экспериментальный = ?")
                                            st.markdown(f"- Normal образцов: текущий = {len(normal_samples_for_score)}, экспериментальный = ?")
                                            
                                            # Проверяем данные
                                            st.markdown("**Данные:**")
                                            st.markdown(f"- Всего строк в df_features: {len(df_features_for_score)}")
                                            st.markdown(f"- Колонок в df_features: {len(df_features_for_score.columns)}")
                                            
                                            # Проверяем, используются ли те же данные из эксперимента
                                            df_from_experiment = safe_session_get("df_from_experiment")
                                            df_features_from_experiment = safe_session_get("df_features_from_experiment")
                                            
                                            if df_features_from_experiment is not None:
                                                st.info(f"💡 Используются данные из эксперимента: {len(df_features_from_experiment)} строк")
                                                
                                                # Сравниваем данные
                                                if len(df_features_for_score) == len(df_features_from_experiment):
                                                    # Проверяем идентичность данных
                                                    common_cols = set(df_features_for_score.columns) & set(df_features_from_experiment.columns)
                                                    if sorted_feature_cols_for_score:
                                                        # Сравниваем значения признаков
                                                        try:
                                                            df_current_sorted = df_features_for_score.sort_values("image").reset_index(drop=True)
                                                            df_exp_sorted = df_features_from_experiment.sort_values("image").reset_index(drop=True)
                                                            
                                                            # Проверяем, что образцы совпадают
                                                            if list(df_current_sorted["image"]) == list(df_exp_sorted["image"]):
                                                                # Сравниваем значения признаков
                                                                for feat in sorted_feature_cols_for_score[:5]:  # Первые 5 признаков
                                                                    if feat in common_cols:
                                                                        diff = (df_current_sorted[feat] - df_exp_sorted[feat]).abs().max()
                                                                        if diff > 1e-6:
                                                                            st.warning(f"⚠️ Признак {feat}: максимальная разница = {diff:.10f}")
                                                                # Проверяем все признаки
                                                                all_diffs = []
                                                                for feat in sorted_feature_cols_for_score:
                                                                    if feat in common_cols:
                                                                        diff = (df_current_sorted[feat] - df_exp_sorted[feat]).abs().max()
                                                                        all_diffs.append(diff)
                                                                if all_diffs:
                                                                    max_diff = max(all_diffs)
                                                                    if max_diff > 1e-6:
                                                                        st.error(f"❌ Обнаружены различия в данных! Максимальная разница: {max_diff:.10f}")
                                                                    else:
                                                                        st.success(f"✅ Данные идентичны (разница < 1e-6)")
                                                            else:
                                                                st.warning("⚠️ Образцы различаются между текущими данными и данными эксперимента")
                                                        except Exception as e:
                                                                st.warning(f"⚠️ Ошибка сравнения данных: {e}")
                                                else:
                                                    st.warning(f"⚠️ Разное количество строк: текущие = {len(df_features_for_score)}, эксперимент = {len(df_features_from_experiment)}")
                                            else:
                                                st.warning("⚠️ Данные НЕ из эксперимента - возможно, используются другие данные!")
                                                
                                                # Показываем, откуда данные
                                                df_from_upload = safe_session_get("df")
                                                if df_from_upload is not None:
                                                    st.info(f"💡 Данные загружены из загруженных файлов: {len(df_from_upload)} строк")
                                                else:
                                                    inference_dir = safe_session_get("inference_dir", "results/inference")
                                                    st.info(f"💡 Данные загружены из директории: {inference_dir}")
                                                
                                                st.markdown("**💡 Рекомендация:** Загрузите эксперимент, чтобы использовать те же данные, что и в эксперименте")
                                        
                                    elif experiment_score is not None:
                                        st.success("✅ Score совпадает с экспериментальным!")
                                    
                                    st.markdown("---")
                        except Exception as e:
                                st.warning(f"⚠️ Не удалось вычислить текущий score: {e}")
            
            # Используем полный набор признаков для интерфейса выбора
            # Сначала проверяем session_state, потом локальные переменные
            df_features_for_selection_val = safe_session_get("df_features_for_selection")
            if df_features_for_selection_val is not None:
                df_features_for_ui = df_features_for_selection_val
            elif 'df_features_for_selection' in locals() and df_features_for_selection is not None:
                df_features_for_ui = df_features_for_selection
            elif use_relative_features:
                df_features_full_val = safe_session_get("df_features_full")
                if df_features_full_val is not None:
                    df_features_for_ui = df_features_full_val
                elif 'df_features_full' in locals() and df_features_full is not None:
                    df_features_for_ui = df_features_full
                else:
                    df_features_val = safe_session_get("df_features")
                    if df_features_val is not None:
                        df_features_for_ui = df_features_val
                    elif 'df_features' in locals() and df_features is not None:
                        df_features_for_ui = df_features
                    else:
                        df_features_for_ui = None
            else:
                df_features_val = safe_session_get("df_features")
                if df_features_val is not None:
                    df_features_for_ui = df_features_val
                elif 'df_features' in locals() and df_features is not None:
                    df_features_for_ui = df_features
                else:
                    df_features_for_ui = None
            
            if df_features_for_ui is not None and len(df_features_for_ui) > 0:
                # Для абсолютных признаков Crypts остается в df_features (он является признаком)
                # Crypts исключается только из относительных признаков, так как используется как нормализатор
                
                # Используем полный набор признаков для интерфейса
                numeric_cols = df_features_for_ui.select_dtypes(include=[np.number]).columns.tolist()
                if "image" in numeric_cols:
                    numeric_cols.remove("image")
                
                # Фильтруем только признаки классов (исключаем служебные колонки)
                # Служебные колонки: PC1, PC1_spectrum, PC1_mode и другие, которые могут быть добавлены в процессе анализа
                service_columns = [
                    'pc1', 'pc1_spectrum', 'pc1_mode', 'pc1_norm', 'pc1_mode_spectrum', 
                    'pc1_mode_gmm', 'pc1_mode_combined', 'pc1_nearest_mode', 'pc1_mode_distance',
                    'gmm_component', 'gmm_max_prob',
                    'cluster', 'score', 'silhouette', 'calinski', 'davies'
                ]
                
                # Определяем паттерны признаков классов
                if use_relative_features:
                    # Относительные признаки: должны заканчиваться на _relative_count, _relative_area, _mean_relative_area
                    feature_patterns = ['_relative_count', '_relative_area', '_mean_relative_area']
                    feature_cols = [
                        col for col in numeric_cols 
                        if any(col.endswith(pattern) for pattern in feature_patterns)
                        and not any(service in col.lower() for service in service_columns)
                    ]
                else:
                    # Абсолютные признаки: должны заканчиваться на _count или _area
                    # Исключаем:
                    # 1. Относительные признаки (если они случайно попали)
                    # 2. Служебные колонки
                    # Crypts ВКЛЮЧАЕТСЯ в абсолютные признаки (он является признаком)
                    # Crypts исключается только из относительных признаков, так как используется как нормализатор
                    feature_patterns = ['_count', '_area']
                    feature_cols = [
                        col for col in numeric_cols 
                        if any(col.endswith(pattern) for pattern in feature_patterns)
                        and not any(service in col.lower() for service in service_columns)
                        and 'relative' not in col.lower()  # Исключаем относительные признаки
                    ]
                
                # Сортируем признаки для удобства
                feature_cols = sorted(feature_cols)
                
                # Путь к конфигурационному файлу
                # Определяем файл конфигурации в зависимости от типа признаков
                # Конфигурационные файлы хранятся в scale/cfg для разделения с кодом
                cfg_dir = Path(__file__).parent / "cfg"
                cfg_dir.mkdir(exist_ok=True)  # Создаем директорию, если её нет
                config_file_relative = cfg_dir / "feature_selection_config_relative.json"
                config_file_absolute = cfg_dir / "feature_selection_config_absolute.json"
                config_file = config_file_relative if use_relative_features else config_file_absolute
                
                # Вспомогательная функция для получения признаков по умолчанию (определяем ДО использования)
                def _get_default_positive_loadings_features(df_features_for_ui, feature_cols, use_relative_features):
                    """Получает признаки по умолчанию (положительные loadings + EoE)."""
                    # Если нет признаков, возвращаем пустой список
                    if not feature_cols or len(feature_cols) == 0:
                        return []
                    
                    # Если df_features_for_ui пустой или не содержит данных, возвращаем пустой список
                    if df_features_for_ui is None or len(df_features_for_ui) == 0:
                        return []
                    
                    try:
                        from scale import pca_scoring
                        df_all_features = aggregate.select_all_feature_columns(df_features_for_ui)
                        all_feature_cols = [c for c in df_all_features.columns if c != "image"]
                        
                        if len(all_feature_cols) == 0:
                            # Если нет признаков для PCA, возвращаем пустой список
                            return []
                        
                        if len(df_all_features) < 2:
                            # Если слишком мало образцов для PCA, возвращаем пустой список
                            return []
                        
                        pca_scorer = pca_scoring.PCAScorer()
                        pca_scorer.fit(df_all_features, all_feature_cols)
                        loadings = pca_scorer.get_feature_importance()
                        
                        # Для относительных признаков исключаем Paneth, для абсолютных - нет
                        if use_relative_features:
                            positive_features = [
                                feat for feat, loading in loadings.items()
                                if loading > 0.05 and 'Paneth' not in feat
                            ]
                        else:
                            # Для абсолютных признаков берем все положительные loadings
                            positive_features = [
                                feat for feat, loading in loadings.items()
                                if loading > 0.05
                            ]
                        
                        eoe_features = [f for f in feature_cols if 'EoE' in f or 'eoe' in f.lower()]
                        default_selected = list(set(positive_features + eoe_features))
                        result = [f for f in default_selected if f in feature_cols]
                        
                        # Если результат пустой, используем топ признаков по модулю loadings (fallback)
                        if not result:
                            # Сортируем признаки по абсолютному значению loadings
                            sorted_loadings = sorted(
                                [(feat, abs(loading)) for feat, loading in loadings.items() if feat in feature_cols],
                                key=lambda x: x[1],
                                reverse=True
                            )
                            
                            # Берем топ 10-15 признаков (или все, если их меньше)
                            top_n = min(15, len(sorted_loadings))
                            if top_n > 0:
                                # Для относительных признаков исключаем Paneth из топ-списка
                                if use_relative_features:
                                    top_features = [
                                        feat for feat, _ in sorted_loadings[:top_n * 2]  # Берем больше, чтобы после фильтрации осталось достаточно
                                        if 'Paneth' not in feat
                                    ][:top_n]
                                else:
                                    top_features = [feat for feat, _ in sorted_loadings[:top_n]]
                                
                                # Добавляем EoE, если есть
                                top_features = list(set(top_features + eoe_features))
                                result = [f for f in top_features if f in feature_cols]
                        
                        return result
                    except Exception as e:
                        # В случае ошибки возвращаем пустой список (не все признаки!)
                        return []
                
                # Функция для загрузки конфигурации
                def load_feature_config():
                    """Загружает выбранные признаки из конфигурационного файла."""
                    if config_file.exists():
                        try:
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                return config.get("selected_features", [])
                        except Exception as e:
                            st.warning(f"⚠️ Не удалось загрузить конфигурацию: {e}")
                            return []
                    return []
                
                # Функция для сохранения конфигурации
                def save_feature_config(selected_features_list):
                    """Сохраняет выбранные признаки в конфигурационный файл."""
                    try:
                        # Загружаем текущую конфигурацию для сохранения метаданных
                        current_config = {}
                        if config_file.exists():
                            try:
                                with open(config_file, 'r', encoding='utf-8') as f:
                                    current_config = json.load(f)
                            except Exception:
                                pass
                        
                        config = {
                            "selected_features": selected_features_list,
                            "description": f"Выбранные {'относительные' if use_relative_features else 'абсолютные'} признаки для построения шкалы патологии",
                            "last_updated": datetime.now().isoformat(),
                            "n_features": len(selected_features_list),
                        }
                        
                        # Сохраняем информацию об исходном эксперименте (если есть)
                        if current_config.get("source_experiment"):
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
                        
                        with open(config_file, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=2, ensure_ascii=False)
                        return True
                    except Exception as e:
                        st.error(f"❌ Не удалось сохранить конфигурацию: {e}")
                        return False
                
                # Инициализация session state для выбранных признаков
                # Ключ для отслеживания типа признаков
                features_type_key = f"features_type_{use_relative_features}"
                
                # Если изменился тип признаков, очищаем выбранные признаки
                # НО: не очищаем, если признаки загружены из эксперимента
                if not safe_session_has(features_type_key) or safe_session_get(features_type_key) != use_relative_features:
                    # Тип признаков изменился - очищаем и загружаем новый конфиг
                    # НО: не очищаем, если используется эксперимент
                    if not (use_experiment_data and safe_session_has("experiment_config_cache")):
                        safe_session_del("selected_features")
                    safe_session_set(features_type_key, use_relative_features)
                
                if not safe_session_has("selected_features"):
                    # Пытаемся загрузить из конфигурационного файла
                    config_features = load_feature_config()
                    
                    if config_features:
                        # Фильтруем только существующие признаки
                        valid_config_features = [f for f in config_features if f in feature_cols]
                        if valid_config_features:
                            safe_session_set("selected_features", valid_config_features)
                        else:
                            # Если конфигурация не подходит, используем положительные loadings + EoE
                            # Показываем, какие признаки не совпали
                            missing_features = [f for f in config_features if f not in feature_cols]
                            if missing_features:
                                st.warning(f"⚠️ Все признаки из конфига не найдены в данных. Не найдено: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
                                # Показываем примеры реальных признаков для отладки
                                if feature_cols:
                                    st.info(f"💡 Примеры доступных признаков: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
                            default_features = _get_default_positive_loadings_features(
                                df_features_for_ui, feature_cols, use_relative_features
                            )
                            if default_features:
                                safe_session_set("selected_features", default_features)
                            else:
                                # Если и по умолчанию ничего не получилось, используем базовый набор
                                basic_features = []
                                pathology_patterns = ['Dysplasia', 'Mild', 'Moderate', 'Meta', 'Neutrophils', 'Plasma Cells', 'Granulomas']
                                for pattern in pathology_patterns:
                                    matching = [f for f in feature_cols if pattern.lower() in f.lower()]
                                    basic_features.extend(matching)
                                eoe_features = [f for f in feature_cols if 'EoE' in f or 'eoe' in f.lower()]
                                basic_features.extend(eoe_features)
                                basic_features = list(set([f for f in basic_features if f in feature_cols]))
                                if basic_features:
                                    safe_session_set("selected_features", basic_features)
                    else:
                        # Если конфигурации нет, используем положительные loadings + EoE
                        default_features = _get_default_positive_loadings_features(
                            df_features_for_ui, feature_cols, use_relative_features
                        )
                        if default_features:
                            safe_session_set("selected_features", default_features)
                        else:
                            # Если и по умолчанию ничего не получилось, используем базовый набор
                            basic_features = []
                            pathology_patterns = ['Dysplasia', 'Mild', 'Moderate', 'Meta', 'Neutrophils', 'Plasma Cells', 'Granulomas']
                            for pattern in pathology_patterns:
                                matching = [f for f in feature_cols if pattern.lower() in f.lower()]
                                basic_features.extend(matching)
                            eoe_features = [f for f in feature_cols if 'EoE' in f or 'eoe' in f.lower()]
                            basic_features.extend(eoe_features)
                            basic_features = list(set([f for f in basic_features if f in feature_cols]))
                            if basic_features:
                                safe_session_set("selected_features", basic_features)
                
                # Обновляем список выбранных признаков если изменились доступные признаки
                # НО только если список не пустой (чтобы не очищать загруженный конфиг)
                # Если используется эксперимент, не перезаписываем selected_features значениями по умолчанию
                # Это позволяет сохранить признаки из эксперимента
                # ВАЖНО: Если признаки загружены из эксперимента, только фильтруем существующие, но не перезаписываем
                selected_features_val = safe_session_get("selected_features")
                if selected_features_val:
                    # Если используется эксперимент, только фильтруем существующие признаки, но сохраняем все
                    if use_experiment_data and safe_session_has("experiment_config_cache"):
                        # Только фильтруем, но не перезаписываем - сохраняем все признаки из эксперимента
                        current_selected = [f for f in selected_features_val if f in feature_cols]
                        # Обновляем только если есть различия, но сохраняем оригинальный список в кэше
                        if len(current_selected) != len(selected_features_val):
                            # Обновляем для текущей сессии, но сохраняем оригинал в кэше
                            safe_session_set("selected_features", current_selected)
                    else:
                        # Не эксперимент - обычная логика
                        current_selected = [f for f in selected_features_val if f in feature_cols]
                        if len(current_selected) != len(selected_features_val):
                            # Обновляем только если есть различия, но не очищаем полностью
                            if current_selected:
                                safe_session_set("selected_features", current_selected)
                            # Если после фильтрации список стал пустым, это означает, что признаки изменились
                            # В этом случае используем значения по умолчанию
                            elif len(selected_features_val) > 0:
                                # Признаки были, но не совпали - используем значения по умолчанию
                                default_features = _get_default_positive_loadings_features(
                                    df_features_for_ui, feature_cols, use_relative_features
                                )
                                if default_features:
                                    safe_session_set("selected_features", default_features)
                
                # Если после всех операций список пустой, используем значения по умолчанию
                # НО: не делаем это для экспериментов, чтобы не потерять загруженные признаки
                selected_features_val = safe_session_get("selected_features")
                if (not selected_features_val or len(selected_features_val) == 0) and \
                   not (use_experiment_data and safe_session_has("experiment_config_cache")):
                    default_features = _get_default_positive_loadings_features(
                        df_features_for_ui, feature_cols, use_relative_features
                    )
                    if default_features:
                        safe_session_set("selected_features", default_features)
                    else:
                        # Если и по умолчанию ничего не получилось, используем базовый набор признаков
                        # Выбираем патологические признаки + EoE (если есть)
                        basic_features = []
                        # Патологические признаки
                        pathology_patterns = ['Dysplasia', 'Mild', 'Moderate', 'Meta', 'Neutrophils', 'Plasma Cells', 'Granulomas']
                        for pattern in pathology_patterns:
                            matching = [f for f in feature_cols if pattern.lower() in f.lower()]
                            basic_features.extend(matching)
                        
                        # EoE
                        eoe_features = [f for f in feature_cols if 'EoE' in f or 'eoe' in f.lower()]
                        basic_features.extend(eoe_features)
                        
                        # Убираем дубликаты и фильтруем только существующие признаки
                        basic_features = list(set([f for f in basic_features if f in feature_cols]))
                        
                        if basic_features:
                            safe_session_set("selected_features", basic_features)
                        # Если и базовых признаков нет, оставляем пустым
                
                # Проверяем, не выбраны ли случайно все признаки (это может быть ошибка)
                # Если выбрано больше 90% признаков, вероятно это ошибка - очищаем
                # НО: не делаем это для экспериментов, чтобы не потерять загруженные признаки
                selected_features_val = safe_session_get("selected_features")
                if selected_features_val and (len(selected_features_val) > 0.9 * len(feature_cols) and 
                    not (use_experiment_data and safe_session_has("experiment_config_cache"))):
                    # Если почти все признаки выбраны, но это не было сделано явно через кнопку "Выбрать все",
                    # то вероятно это ошибка инициализации - очищаем и используем только положительные loadings
                    if not safe_session_has("features_all_selected_explicitly"):
                        # Пересчитываем положительные loadings
                        default_features = _get_default_positive_loadings_features(
                            df_features_for_ui, feature_cols, use_relative_features
                        )
                        if default_features:
                            safe_session_set("selected_features", default_features)
                
                # НЕ устанавливаем все признаки по умолчанию - пользователь должен выбрать явно
                
                # ОПТИМИЗАЦИЯ: Кэшируем группировку признаков, чтобы не выполнять ее при каждом rerun
                feature_groups_cache_key = f"feature_groups_{hash(tuple(sorted(feature_cols)))}"
                if safe_session_has(feature_groups_cache_key):
                    # Используем кэшированную группировку
                    cached_groups = safe_session_get(feature_groups_cache_key)
                    pathology_features = cached_groups.get("pathology", [])
                    meta_features = cached_groups.get("meta", [])
                    immune_features = cached_groups.get("immune", [])
                    structural_features = cached_groups.get("structural", [])
                    paneth_features = cached_groups.get("paneth", [])
                    other_features = cached_groups.get("other", [])
                else:
                    # Группируем признаки по категориям для удобства отображения
                    pathology_features = [f for f in feature_cols if any(x in f.lower() for x in 
                        ['dysplasia', 'mild', 'moderate', 'eoe', 'granulomas'])]
                    meta_features = [f for f in feature_cols if 'meta' in f.lower()]
                    immune_features = [f for f in feature_cols if any(x in f.lower() for x in 
                        ['neutrophils', 'plasma', 'enterocytes', 'plasma cells'])]
                    structural_features = [f for f in feature_cols if any(x in f.lower() for x in 
                        ['surface epithelium', 'muscularis mucosae', 'surface_epithelium', 'muscularis_mucosae'])]
                    paneth_features = [f for f in feature_cols if 'paneth' in f.lower()]
                    other_features = [f for f in feature_cols if f not in pathology_features + meta_features + 
                        immune_features + structural_features + paneth_features]
                    # Сохраняем в кэш
                    safe_session_set(feature_groups_cache_key, {
                        "pathology": pathology_features,
                        "meta": meta_features,
                        "immune": immune_features,
                        "structural": structural_features,
                        "paneth": paneth_features,
                        "other": other_features
                    })
                
                # Проверяем, есть ли структурные признаки в данных
                if not structural_features:
                    st.warning("⚠️ Surface epithelium и Muscularis mucosae не найдены в данных. "
                             "Убедитесь, что они присутствуют в исходных предсказаниях (JSON файлах).")
                
                # УБИРАЕМ ФОРМУ - используем обычные checkbox'ы с key
                # Это предотвращает ненужные rerun при изменении checkbox'ов
                # Группируем в колонки для компактности
                col1, col2, col3 = st.columns(3)
                
                # Получаем текущие выбранные признаки из session_state (через key checkbox'ов)
                # Checkbox'ы с key автоматически сохраняют состояние в session_state
                selected_features_dict = {}
                
                # ОПТИМИЗАЦИЯ: Получаем selected_features ОДИН РАЗ для всех checkbox'ов
                saved_selected_features_list = safe_session_get("selected_features", [])
                saved_selected_features_set = set(saved_selected_features_list) if saved_selected_features_list else set()
                
                with col1:
                    if pathology_features:
                        # Считаем выбранные через session_state (checkbox'ы с key автоматически сохраняют состояние)
                        # ОПТИМИЗАЦИЯ: Используем кэшированное значение вместо множественных вызовов safe_session_get
                        selected_count = sum(1 for f in pathology_features if safe_session_get(f"feat_{f}", False))
                        st.markdown(f"**Патологические:** ({selected_count}/{len(pathology_features)} выбрано)")
                        for feat in pathology_features:
                            checkbox_key = f"feat_{feat}"
                            # КРИТИЧНО: Инициализируем значение ПЕРЕД созданием виджета
                            if not safe_session_has(checkbox_key):
                                initial_value = feat in saved_selected_features_set
                                safe_session_set(checkbox_key, initial_value)
                            # Создаем виджет БЕЗ value - значение автоматически возьмется из session_state
                            checkbox_value = st.checkbox(feat, key=checkbox_key)
                            selected_features_dict[feat] = checkbox_value
                    else:
                        st.markdown("**Патологические:** (нет признаков)")
                    
                    if meta_features:
                        selected_count = sum(1 for f in meta_features if safe_session_get(f"feat_{f}", False))
                        st.markdown(f"**Метаплазия:** ({selected_count}/{len(meta_features)} выбрано)")
                        for feat in meta_features:
                            checkbox_key = f"feat_{feat}"
                            if not safe_session_has(checkbox_key):
                                initial_value = feat in saved_selected_features_set
                                safe_session_set(checkbox_key, initial_value)
                            checkbox_value = st.checkbox(feat, key=checkbox_key)
                            selected_features_dict[feat] = checkbox_value
                    else:
                        st.markdown("**Метаплазия:** (нет признаков)")
                    
                with col2:
                    if immune_features:
                        selected_count = sum(1 for f in immune_features if safe_session_get(f"feat_{f}", False))
                        st.markdown(f"**Иммунные клетки:** ({selected_count}/{len(immune_features)} выбрано)")
                        for feat in immune_features:
                            checkbox_key = f"feat_{feat}"
                            if not safe_session_has(checkbox_key):
                                initial_value = feat in saved_selected_features_set
                                safe_session_set(checkbox_key, initial_value)
                            checkbox_value = st.checkbox(feat, key=checkbox_key)
                            selected_features_dict[feat] = checkbox_value
                    else:
                        st.markdown("**Иммунные клетки:** (нет признаков)")
                    
                    if paneth_features:
                        selected_count = sum(1 for f in paneth_features if safe_session_get(f"feat_{f}", False))
                        st.markdown(f"**Paneth:** ({selected_count}/{len(paneth_features)} выбрано)")
                        for feat in paneth_features:
                            checkbox_key = f"feat_{feat}"
                            if not safe_session_has(checkbox_key):
                                initial_value = feat in saved_selected_features_set
                                safe_session_set(checkbox_key, initial_value)
                            checkbox_value = st.checkbox(feat, key=checkbox_key)
                            selected_features_dict[feat] = checkbox_value
                    else:
                        st.markdown("**Paneth:** (нет признаков)")
                
                with col3:
                    if structural_features:
                        selected_count = sum(1 for f in structural_features if safe_session_get(f"feat_{f}", False))
                        st.markdown(f"**Структурные:** ({selected_count}/{len(structural_features)} выбрано)")
                        for feat in structural_features:
                            checkbox_key = f"feat_{feat}"
                            if not safe_session_has(checkbox_key):
                                initial_value = feat in saved_selected_features_set
                                safe_session_set(checkbox_key, initial_value)
                            checkbox_value = st.checkbox(feat, key=checkbox_key)
                            selected_features_dict[feat] = checkbox_value
                    else:
                        st.markdown("**Структурные:** (нет признаков)")
                    
                    if other_features:
                        selected_count = sum(1 for f in other_features if safe_session_get(f"feat_{f}", False))
                        st.markdown(f"**Другие:** ({selected_count}/{len(other_features)} выбрано)")
                        for feat in other_features:
                            checkbox_key = f"feat_{feat}"
                            if not safe_session_has(checkbox_key):
                                initial_value = feat in saved_selected_features_set
                                safe_session_set(checkbox_key, initial_value)
                            checkbox_value = st.checkbox(feat, key=checkbox_key)
                            selected_features_dict[feat] = checkbox_value
                    else:
                        st.markdown("**Другие:** (нет признаков)")
                    
                # Показываем информацию о всех признаках
                st.markdown("---")
                # ОПТИМИЗАЦИЯ: Используем уже вычисленный selected_features_dict вместо повторного обхода всех признаков
                # Получаем текущие выбранные признаки из словаря, который уже был заполнен при создании checkbox'ов
                current_selected_from_checkboxes = [feat for feat, is_selected in selected_features_dict.items() if is_selected]
                
                total_selected = len(current_selected_from_checkboxes)
                st.caption(f"📊 Всего признаков: {len(feature_cols)}, Выбрано: {total_selected}, Не выбрано: {len(feature_cols) - total_selected}")
                
                # Показываем невыбранные признаки для удобства
                unselected_features = [f for f in feature_cols if f not in current_selected_from_checkboxes]
                if unselected_features:
                    with st.expander(f"👁️ Показать невыбранные признаки ({len(unselected_features)})"):
                        for feat in sorted(unselected_features):
                            st.text(f"  ☐ {feat}")
                
                # ОПТИМИЗАЦИЯ: Проверяем, изменились ли признаки сравнивая списки
                current_selected_sorted = sorted(current_selected_from_checkboxes)
                
                # Получаем сохраненные признаки
                saved_selected_features = safe_session_get("selected_features", [])
                saved_selected_sorted = sorted(saved_selected_features) if saved_selected_features else []
                
                # КРИТИЧНО: Сравниваем сами списки признаков
                lists_are_different = (current_selected_sorted != saved_selected_sorted)
                
                # КРИТИЧНО: Признаки изменились ТОЛЬКО если списки различаются
                if not saved_selected_sorted:
                    # Нет сохраненных признаков - считаем что признаки изменились (первая загрузка)
                    features_changed = True
                elif lists_are_different:
                    # Списки различаются - признаки точно изменились
                    features_changed = True
                else:
                    # Списки совпадают - признаки не изменились
                    features_changed = False
                
                
                # Показываем статус изменения признаков
                if features_changed:
                    st.info(f"ℹ️ Выбрано {len(current_selected_sorted)} признаков. Нажмите 'Перерасчет метрик' для применения изменений.")
                else:
                    st.success(f"✅ Признаки не изменены ({len(current_selected_sorted)} выбрано). Перерасчет не требуется.")
                
                # Кнопка перерасчета - ВСЕГДА показываем и всегда активна
                # При нажатии проверяем, изменились ли чекбоксы
                # Если не изменились - показываем warning и не выполняем пересчет
                st.markdown("---")
                apply_button = st.button("🔄 Перерасчет метрик", use_container_width=True, type="primary")
                
                if apply_button:
                    # Проверяем, изменились ли признаки
                    if not features_changed:
                        # Признаки не изменились - показываем warning и не выполняем пересчет
                        st.warning("⚠️ Признаки не изменились. Пересчет не требуется.")
                    elif len(current_selected_sorted) == 0:
                        st.error("❌ Необходимо выбрать хотя бы один признак!")
                    else:
                        # Признаки изменились - выполняем пересчет
                        selected_features_list = current_selected_sorted.copy()
                        
                        # Сохраняем выбранные признаки
                        safe_session_set("selected_features", selected_features_list)
                        
                        # КРИТИЧНО: Проверяем, совпадают ли признаки с экспериментальными
                        # Если совпадают - используем метрики из эксперимента, не пересчитываем
                        use_experiment_metrics = False
                        if use_experiment_data and safe_session_has("experiment_config_cache"):
                            experiment_config = safe_session_get("experiment_config_cache")
                            if experiment_config and experiment_config.get("features"):
                                experiment_features = sorted(experiment_config.get("features", []))
                                current_features_sorted = sorted(selected_features_list)
                                # Если признаки совпадают с экспериментальными - используем метрики из эксперимента
                                if experiment_features == current_features_sorted:
                                    use_experiment_metrics = True
                                    if experiment_config.get("metrics"):
                                        experiment_metrics = experiment_config.get("metrics", {})
                                        safe_session_set("current_metrics", experiment_metrics)
                                        safe_session_set("metrics_features", selected_features_list.copy())
                        
                        # Устанавливаем флаг features_applied только если метрики нужно пересчитать
                        if not use_experiment_metrics:
                            safe_session_set("features_applied", True)
                            # Очищаем metrics_features, чтобы принудительно пересчитать метрики
                            safe_session_del("metrics_features")
                            safe_session_del("current_metrics")
                        else:
                            safe_session_set("features_applied", False)
                            # НЕ очищаем метрики - используем из эксперимента
                        
                        # Сохраняем хэш признаков для будущих сравнений
                        selected_features_sorted = sorted(selected_features_list)
                        new_features_hash = hash(tuple(selected_features_sorted))
                        safe_session_set("features_key", new_features_hash)
                        
                        
                        # Очищаем GMM и спектр, если они были обучены (чтобы пересчитались с новыми признаками)
                        analyzer_val = safe_session_get("analyzer")
                        if analyzer_val is not None and analyzer_val.gmm is not None:
                            # Очищаем GMM из анализатора
                            analyzer_val.gmm = None
                            safe_session_set("analyzer", analyzer_val)
                        # Очищаем кэш спектра
                        safe_session_del("df_spectrum")
                        safe_session_del("spectrum_cache_key")
                        # Очищаем кэш качества GMM
                        try:
                            all_keys = list(st.session_state.keys()) if hasattr(st, 'session_state') else []
                            cache_keys_to_remove = [key for key in all_keys if key.startswith("gmm_quality_")]
                            for key in cache_keys_to_remove:
                                safe_session_del(key)
                        except (RuntimeError, AttributeError):
                            pass
                        
                        # Очищаем старые метрики, чтобы они пересчитались с новыми признаками
                        if safe_session_has("current_metrics"):
                            safe_session_del("current_metrics")
                        
                        # Сохраняем в конфигурационный файл
                        if save_feature_config(selected_features_list):
                            st.success("✅ Конфигурация сохранена в файл")
                            # Показываем информацию об исходном эксперименте (если есть)
                            try:
                                with open(config_file, 'r', encoding='utf-8') as f:
                                    saved_config = json.load(f)
                                if saved_config.get("source_experiment"):
                                    st.info(f"💡 Исходный эксперимент: **{saved_config['source_experiment']}** (не изменен)")
                            except Exception:
                                pass
                        
                        # Сбрасываем флаг изменения checkbox'ов после применения признаков
                        safe_session_set("features_applied", False)  # Сбрасываем флаг применения
                        st.success("✅ Признаки применены. Метрики будут пересчитаны.")
                        st.rerun()
                
                # Показываем текущий статус
                st.markdown("---")
                
                # Показываем информацию об источнике конфигурации
                # Приоритет: если используется эксперимент, показываем его данные из session_state
                # Иначе показываем данные из конфига
                if use_experiment_data and safe_session_has("experiment_name") and safe_session_has("experiment_config_cache"):
                    # Используем данные из текущего эксперимента
                    current_exp_name = safe_session_get("experiment_name")
                    experiment_config = safe_session_get("experiment_config_cache")
                    
                    st.success(f"📊 Конфигурация из эксперимента: **{current_exp_name}**")
                    
                    # Показываем метрики текущего эксперимента
                    if experiment_config.get("metrics"):
                        metrics = experiment_config.get("metrics", {})
                        with st.expander("📈 Метрики исходного эксперимента"):
                            score_val = metrics.get('score', 0)
                            separation_val = metrics.get('separation', 0)
                            mean_pc1_norm_mod_val = metrics.get('mean_pc1_norm_mod', 0)
                            explained_variance_val = metrics.get('explained_variance', 0)
                            
                            # Score
                            st.markdown("### Score (комплексная оценка)")
                            st.metric("Score", f"{score_val:.4f}")
                            show_score_exp_1 = st.checkbox("📚 Показать подробное объяснение Score", key="score_exp_1", value=False)
                            if show_score_exp_1:
                                st.markdown(f"""
                                **📌 ЧТО ТАКОЕ SCORE?**
                                
                                Score - это **комплексная метрика качества** набора признаков, которая объединяет три важных аспекта:
                                
                                ---
                                
                                **📐 ФОРМУЛА:**
                                
                                ```
                                Score = 0.4 × separation + 0.3 × mean_pc1_norm_mod + 0.3 × explained_variance
                                ```
                                
                                **Текущие значения:**
                                - Separation: {separation_val:.4f} (вес 40%)
                                - Mod (норм. PC1): {mean_pc1_norm_mod_val:.4f} (вес 30%)
                                - Объясненная дисперсия: {explained_variance_val:.4f} (вес 30%)
                                
                                **Вычисление:**
                                ```
                                Score = 0.4 × {separation_val:.4f} + 0.3 × {mean_pc1_norm_mod_val:.4f} + 0.3 × {explained_variance_val:.4f}
                                Score = {0.4 * separation_val:.4f} + {0.3 * mean_pc1_norm_mod_val:.4f} + {0.3 * explained_variance_val:.4f}
                                Score = {score_val:.4f}
                                ```
                                
                                ---
                                
                                **⚖️ ВЕСА КОМПОНЕНТОВ:**
                                
                                1. **40% - Separation** (самый важный фактор)
                                   - Показывает, насколько хорошо разделяются группы mod и normal
                                   - Чем больше разница между средними PC1, тем лучше
                                
                                2. **30% - Mean PC1 norm mod** (позиция патологических образцов)
                                   - Показывает, где находятся патологические образцы на шкале 0-1
                                   - Цель: близко к 1.0 (высокие значения PC1)
                                
                                3. **30% - Explained variance** (качество PCA)
                                   - Показывает, насколько хорошо PC1 описывает данные
                                   - Чем больше, тем информативнее PC1
                                
                                ---
                                
                                **📊 ИНТЕРПРЕТАЦИЯ ЗНАЧЕНИЙ:**
                                
                                - **< 1.0** → Плохое качество набора признаков
                                - **1.0 - 2.0** → Удовлетворительное качество
                                - **2.0 - 3.0** → Хорошее качество
                                - **3.0 - 3.5** → Отличное качество
                                - **> 3.5** → Превосходное качество
                                
                                ---
                                
                                **💡 ПОЧЕМУ ИМЕННО ТАКИЕ ВЕСА?**
                                
                                - **Separation (40%)** - самый важный, потому что главная цель - разделить нормальные и патологические образцы
                                - **Mod position (30%)** - важно, чтобы патологические образцы были справа (высокие PC1)
                                - **Explained variance (30%)** - важно, чтобы PC1 была информативной и стабильной
                                
                                ---
                                
                                **🎯 КАК УЛУЧШИТЬ SCORE?**
                                
                                1. **Увеличить separation:**
                                   - Выбрать признаки, которые лучше разделяют группы
                                   - Убрать признаки, которые "смазывают" разделение
                                
                                2. **Увеличить mean_pc1_norm_mod:**
                                   - Выбрать признаки, которые дают высокие PC1 для патологических образцов
                                   - Убедиться, что патологические образцы действительно справа на шкале
                                
                                3. **Увеличить explained_variance:**
                                   - Выбрать признаки, которые коррелируют между собой
                                   - Убрать шумовые признаки, которые не вносят структуру
                                """)
                            st.info(
                                "**Score** - комплексная оценка качества набора признаков:\n\n"
                                "• 40% - разделение между группами (separation)\n"
                                "• 30% - позиция mod образцов на нормализованной шкале (ближе к 1)\n"
                                "• 30% - объясненная дисперсия PC1\n\n"
                                "**Хорошие значения:** > 1.0"
                            )
                            
                            st.markdown("---")
                            
                            # Separation
                            st.markdown("### Separation (разделение групп)")
                            st.metric("Separation", f"{separation_val:.4f}")
                            show_separation_exp_1 = st.checkbox("📚 Показать подробное объяснение Separation", key="separation_exp_1", value=False)
                            if show_separation_exp_1:
                                st.markdown(f"""
                                **📌 ЧТО ТАКОЕ SEPARATION?**
                                
                                Separation показывает, **насколько хорошо разделяются** патологические (mod) и нормальные (normal) образцы на шкале PC1.
                                
                                ---
                                
                                **📐 ФОРМУЛА:**
                                
                                ```
                                separation = mean(PC1_mod) - mean(PC1_normal)
                                ```
                                
                                где:
                                - `mean(PC1_mod)` = среднее значение PC1 для патологических образцов
                                - `mean(PC1_normal)` = среднее значение PC1 для нормальных образцов
                                
                                ---
                                
                                **🔧 КАК ВЫЧИСЛЯЕТСЯ:**
                                
                                1. После обучения PCA на выбранных признаках вычисляется PC1 для всех образцов
                                2. Образцы разделяются на две группы: mod (патологические) и normal (нормальные)
                                3. Вычисляется среднее PC1 для каждой группы
                                4. Separation = разница между средними
                                
                                ```python
                                # Пример вычисления
                                pc1_mod = [2.5, 3.1, 2.8, 3.5, ...]  # PC1 для патологических
                                pc1_normal = [-1.2, -0.8, -1.5, ...]  # PC1 для нормальных
                                
                                mean_pc1_mod = np.mean(pc1_mod)  # например, 3.0
                                mean_pc1_normal = np.mean(pc1_normal)  # например, -1.0
                                
                                separation = 3.0 - (-1.0) = 4.0
                                ```
                                
                                ---
                                
                                **📊 ИНТЕРПРЕТАЦИЯ ЗНАЧЕНИЙ:**
                                
                                - **Отрицательное значение** → патологические образцы имеют МЕНЬШИЕ PC1, чем нормальные (плохо!)
                                - **0.0** → группы не разделяются (очень плохо)
                                - **0.0 - 2.0** → слабое разделение
                                - **2.0 - 4.0** → хорошее разделение
                                - **4.0 - 6.0** → отличное разделение
                                - **> 6.0** → превосходное разделение
                                
                                ---
                                
                                **🎯 ЦЕЛЕВЫЕ ЗНАЧЕНИЯ:**
                                
                                - **Хорошие значения:** > 2.0
                                  - Группы достаточно разделены для классификации
                                  - Можно различить патологические и нормальные образцы
                                
                                - **Отличные значения:** > 4.0
                                  - Группы хорошо разделены
                                  - Минимальное перекрытие между группами
                                  - Высокая точность классификации
                                
                                ---
                                
                                **💡 ВИЗУАЛЬНАЯ ИНТЕРПРЕТАЦИЯ:**
                                
                                Представьте распределение PC1 на графике:
                                
                                ```
                                Нормальные образцы:     |----|  (низкие PC1, слева)
                                Патологические:                    |----|  (высокие PC1, справа)
                                
                                Разрыв между группами = separation
                                ```
                                
                                Чем больше разрыв, тем лучше разделение!
                                
                                ---
                                
                                **⚠️ ВАЖНЫЕ МОМЕНТЫ:**
                                
                                1. **Separation - самая важная метрика** (вес 40% в Score)
                                   - Если группы не разделяются, другие метрики не помогут
                                
                                2. **Высокое separation не гарантирует высокий Score**
                                   - Нужно также хорошее explained_variance и правильная позиция mod
                                
                                3. **Низкое separation может быть из-за:**
                                   - Неправильно выбранных признаков
                                   - Смешанных данных (нормальные и патологические похожи)
                                   - Шумовых признаков, которые "смазывают" разделение
                                """)
                            st.info(
                                "**Separation** - разница между средними значениями PC1 для патологических (mod) "
                                "и нормальных (normal) образцов.\n\n"
                                "• Чем больше значение, тем лучше разделение между группами\n"
                                "• **Хорошие значения:** > 2.0\n"
                                "• **Отличные значения:** > 4.0"
                            )
                            
                            st.markdown("---")
                            
                            # Mod (норм. PC1)
                            st.markdown("### Mod (норм. PC1)")
                            st.metric("Mod (норм. PC1)", f"{mean_pc1_norm_mod_val:.4f}")
                            show_mod_exp_1 = st.checkbox("📚 Показать подробное объяснение Mod (норм. PC1)", key="mod_exp_1", value=False)
                            if show_mod_exp_1:
                                st.markdown(f"""
                                **📌 ЧТО ТАКОЕ MOD (НОРМ. PC1)?**
                                
                                Mod (норм. PC1) показывает, **где находятся патологические образцы** на нормализованной шкале PC1 от 0 до 1.
                                
                                ---
                                
                                **📐 ФОРМУЛА:**
                                
                                ```
                                mean_pc1_norm_mod = mean((PC1_mod - PC1_min) / (PC1_max - PC1_min))
                                ```
                                
                                где:
                                - `PC1_mod` = значения PC1 для патологических образцов
                                - `PC1_min` = минимальное значение PC1 среди ВСЕХ образцов
                                - `PC1_max` = максимальное значение PC1 среди ВСЕХ образцов
                                
                                ---
                                
                                **🔧 КАК ВЫЧИСЛЯЕТСЯ:**
                                
                                1. Вычисляется PC1 для всех образцов (нормальных и патологических)
                                2. Находятся минимум и максимум PC1 среди всех образцов
                                3. PC1 патологических образцов нормализуется в диапазон [0, 1]:
                                   ```
                                   PC1_norm_mod = (PC1_mod - PC1_min) / (PC1_max - PC1_min)
                                   ```
                                4. Вычисляется среднее нормализованных значений
                                
                                ```python
                                # Пример вычисления
                                pc1_all = [-2.0, -1.0, 0.5, 1.5, 2.5, 3.5, 4.0]  # Все образцы
                                pc1_mod = [2.5, 3.5, 4.0]  # Только патологические
                                
                                pc1_min = -2.0  # Минимум среди всех
                                pc1_max = 4.0   # Максимум среди всех
                                
                                # Нормализация патологических
                                pc1_norm_mod = [(2.5 - (-2.0)) / (4.0 - (-2.0)),
                                               (3.5 - (-2.0)) / (4.0 - (-2.0)),
                                               (4.0 - (-2.0)) / (4.0 - (-2.0))]
                                pc1_norm_mod = [0.75, 0.92, 1.0]
                                
                                mean_pc1_norm_mod = 0.89  # Среднее
                                ```
                                
                                ---
                                
                                **📊 ИНТЕРПРЕТАЦИЯ ЗНАЧЕНИЙ:**
                                
                                - **0.0** → патологические образцы имеют минимальные PC1 (очень плохо!)
                                  - Они находятся слева на шкале, как нормальные
                                  - Нельзя различить патологию
                                
                                - **0.0 - 0.5** → патологические образцы в нижней половине шкалы (плохо)
                                  - Слабое разделение с нормальными
                                
                                - **0.5 - 0.7** → патологические образцы в верхней половине (удовлетворительно)
                                  - Есть разделение, но не идеальное
                                
                                - **0.7 - 0.85** → патологические образцы в верхней части (хорошо)
                                  - Хорошее разделение с нормальными
                                  - Патология ассоциирована с высокими PC1
                                
                                - **0.85 - 1.0** → патологические образцы в верхней части (отлично)
                                  - Отличное разделение
                                  - Патология четко ассоциирована с высокими PC1
                                
                                ---
                                
                                **🎯 ЦЕЛЕВЫЕ ЗНАЧЕНИЯ:**
                                
                                - **Хорошие значения:** > 0.7
                                  - Патологические образцы находятся в верхней части шкалы
                                  - Можно различить патологию по высоким PC1
                                
                                - **Отличные значения:** > 0.85
                                  - Патологические образцы находятся близко к максимуму
                                  - Четкая ассоциация патологии с высокими PC1
                                  - Высокая точность классификации
                                
                                ---
                                
                                **💡 ВИЗУАЛЬНАЯ ИНТЕРПРЕТАЦИЯ:**
                                
                                Представьте нормализованную шкалу PC1 от 0 до 1:
                                
                                ```
                                0.0 ──────── 0.5 ──────── 1.0
                                 │           │           │
                                 │           │           │
                            Нормальные    Смешанные   Патологические
                            (низкие PC1)              (высокие PC1)
                                
                                mean_pc1_norm_mod показывает, где в среднем находятся патологические
                                ```
                                
                                **Цель:** патологические образцы должны быть справа (близко к 1.0)!
                                
                                ---
                                
                                **⚠️ ВАЖНЫЕ МОМЕНТЫ:**
                                
                                1. **Связь с Separation:**
                                   - Высокое separation + высокий mean_pc1_norm_mod = отличное разделение
                                   - Но можно иметь высокое separation при низком mean_pc1_norm_mod (если нормальные очень низкие)
                                
                                2. **Нормализация важна:**
                                   - Без нормализации нельзя сравнивать разные наборы признаков
                                   - Нормализация позволяет интерпретировать позицию на универсальной шкале 0-1
                                
                                3. **Низкое значение может означать:**
                                   - Патологические образцы не имеют высоких PC1
                                   - Возможно, признаки выбраны неправильно
                                   - Или патология не ассоциирована с выбранными признаками
                                """)
                            st.info(
                                "**Mod (норм. PC1)** - среднее нормализованное значение PC1 для патологических образцов.\n\n"
                                "• Значение от 0 до 1 на нормализованной шкале\n"
                                "• **Цель:** близко к 1.0 (патологические образцы должны иметь высокие значения PC1)\n"
                                "• **Хорошие значения:** > 0.7\n"
                                "• **Отличные значения:** > 0.85"
                            )
                            
                            st.markdown("---")
                            
                            # Объясненная дисперсия
                            st.markdown("### Объясненная дисперсия (Explained Variance)")
                            st.metric("Объясненная дисперсия", f"{explained_variance_val:.4f}")
                            show_variance_exp_1 = st.checkbox("📚 Показать подробное объяснение объясненной дисперсии", key="variance_exp_1", value=False)
                            if show_variance_exp_1:
                                st.markdown("""
                                **📌 ЧТО ТАКОЕ ОБЪЯСНЕННАЯ ДИСПЕРСИЯ?**
                                
                                Объясненная дисперсия показывает, **какую долю общей вариации данных** объясняет первая главная компонента (PC1).
                                
                                ---
                                
                                **📐 МАТЕМАТИЧЕСКАЯ ФОРМУЛА:**
                                
                                ```
                                explained_variance = λ₁ / Σλᵢ
                                ```
                                
                                где:
                                - `λ₁` = собственное значение (eigenvalue) первой главной компоненты PC1
                                - `Σλᵢ` = сумма всех собственных значений (общая дисперсия всех компонент)
                                
                                ---
                                
                                **🔧 КАК ВЫЧИСЛЯЕТСЯ:**
                                
                                1. **PCA находит собственные значения** матрицы ковариации:
                                   - Каждое собственное значение `λᵢ` = дисперсия вдоль соответствующей главной компоненты
                                   - PC1 имеет максимальное собственное значение `λ₁`
                                
                                2. **Вычисляется доля:**
                                   ```python
                                   explained_variance = pca.explained_variance_ratio_[0]
                                   # Это эквивалентно: λ₁ / (λ₁ + λ₂ + ... + λₙ)
                                   ```
                                
                                3. **Результат:** число от 0 до 1 (или от 0% до 100%)
                                
                                ---
                                
                                **💡 ЧТО ТАКОЕ ДИСПЕРСИЯ?**
                                
                                **Дисперсия** - это мера разброса данных:
                                - Большая дисперсия = данные сильно различаются
                                - Малая дисперсия = данные похожи друг на друга
                                
                                **В контексте PCA:**
                                - PCA находит направления, вдоль которых данные имеют максимальную дисперсию
                                - PC1 = направление с максимальной дисперсией
                                - Объясненная дисперсия = сколько этой максимальной дисперсии "захватывает" PC1
                                
                                ---
                                
                                **📊 ИНТЕРПРЕТАЦИЯ ЗНАЧЕНИЙ:**
                                
                                - **0.0 (0%)** → PC1 не объясняет вариацию (плохо)
                                - **0.3 (30%)** → PC1 объясняет 30% вариации (хорошо)
                                - **0.5 (50%)** → PC1 объясняет половину вариации (отлично)
                                - **0.7 (70%)** → PC1 объясняет большую часть вариации (очень хорошо)
                                - **1.0 (100%)** → PC1 объясняет всю вариацию (все данные лежат на одной линии)
                                
                                ---
                                
                                **🎯 ЦЕЛЕВЫЕ ЗНАЧЕНИЯ:**
                                
                                - **Хорошие значения:** > 0.3 (30%)
                                  - PC1 достаточно информативна для описания данных
                                  - Можно использовать для классификации
                                
                                - **Отличные значения:** > 0.5 (50%)
                                  - PC1 объясняет большую часть вариации
                                  - Данные хорошо структурированы вдоль PC1
                                  - Высокое качество разделения
                                
                                ---
                                
                                **⚠️ ВАЖНЫЕ МОМЕНТЫ:**
                                
                                1. **Низкая объясненная дисперсия (< 0.3)** может означать:
                                   - Данные многомерны и не могут быть сведены к одной компоненте
                                   - Нужны дополнительные компоненты (PC2, PC3...)
                                   - Признаки слабо коррелируют между собой
                                
                                2. **Высокая объясненная дисперсия (> 0.7)** означает:
                                   - Данные хорошо структурированы
                                   - Большая часть информации содержится в PC1
                                   - Можно эффективно использовать одномерную шкалу
                                
                                3. **Связь с другими метриками:**
                                   - Высокая объясненная дисперсия + высокое separation = отличный набор признаков
                                   - Низкая объясненная дисперсия может компенсироваться хорошим separation
                                
                                ---
                                
                                **📈 ПРИМЕР:**
                                
                                Если объясненная дисперсия = 0.357 (35.7%):
                                - PC1 объясняет 35.7% общей вариации данных
                                - Остальные 64.3% вариации объясняются другими компонентами
                                - Это хорошее значение для биомедицинских данных
                                """)
                            st.info(
                                "**Объясненная дисперсия** - доля дисперсии данных, объясняемая первой главной компонентой (PC1).\n\n"
                                "• Показывает, насколько хорошо PC1 описывает вариативность данных\n"
                                "• Значение от 0 до 1 (или от 0% до 100%)\n"
                                "• **Хорошие значения:** > 0.3 (30%)\n"
                                "• **Отличные значения:** > 0.5 (50%)"
                            )
                else:
                    # Используем данные из конфига
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            current_config_info = json.load(f)
                        
                        if current_config_info.get("source_experiment"):
                            source_exp = current_config_info["source_experiment"]
                            user_modified = current_config_info.get("user_modified", False)
                            
                            if user_modified:
                                st.info(f"📊 Конфигурация из эксперимента: **{source_exp}** (изменено пользователем)")
                            else:
                                st.success(f"📊 Конфигурация из эксперимента: **{source_exp}**")
                            
                            # Показываем метрики исходного эксперимента
                            if current_config_info.get("original_metrics") or current_config_info.get("metrics"):
                                metrics = current_config_info.get("original_metrics") or current_config_info.get("metrics", {})
                                with st.expander("📈 Метрики исходного эксперимента"):
                                    score_val = metrics.get('score', 0)
                                    separation_val = metrics.get('separation', 0)
                                    mean_pc1_norm_mod_val = metrics.get('mean_pc1_norm_mod', 0)
                                    explained_variance_val = metrics.get('explained_variance', 0)
                                    
                                    # Score
                                    st.markdown("### Score (комплексная оценка)")
                                    st.metric("Score", f"{score_val:.4f}")
                                    st.info(
                                        "**Score** - комплексная оценка качества набора признаков:\n\n"
                                        "• 40% - разделение между группами (separation)\n"
                                        "• 30% - позиция mod образцов на нормализованной шкале (ближе к 1)\n"
                                        "• 30% - объясненная дисперсия PC1\n\n"
                                        "**Хорошие значения:** > 1.0"
                                    )
                                    
                                    st.markdown("---")
                                    
                                    # Separation
                                    st.markdown("### Separation (разделение групп)")
                                    st.metric("Separation", f"{separation_val:.4f}")
                                    st.info(
                                        "**Separation** - разница между средними значениями PC1 для патологических (mod) "
                                        "и нормальных (normal) образцов.\n\n"
                                        "• Чем больше значение, тем лучше разделение между группами\n"
                                        "• **Хорошие значения:** > 2.0\n"
                                        "• **Отличные значения:** > 4.0"
                                    )
                                    
                                    st.markdown("---")
                                    
                                    # Mod (норм. PC1)
                                    st.markdown("### Mod (норм. PC1)")
                                    st.metric("Mod (норм. PC1)", f"{mean_pc1_norm_mod_val:.4f}")
                                    st.info(
                                        "**Mod (норм. PC1)** - среднее нормализованное значение PC1 для патологических образцов.\n\n"
                                        "• Значение от 0 до 1 на нормализованной шкале\n"
                                        "• **Цель:** близко к 1.0 (патологические образцы должны иметь высокие значения PC1)\n"
                                        "• **Хорошие значения:** > 0.7\n"
                                        "• **Отличные значения:** > 0.85"
                                    )
                                    
                                    st.markdown("---")
                                    
                                    # Объясненная дисперсия
                                    st.markdown("### Объясненная дисперсия")
                                    st.metric("Объясненная дисперсия", f"{explained_variance_val:.4f}")
                                    st.info(
                                        "**Объясненная дисперсия** - доля дисперсии данных, объясняемая первой главной компонентой (PC1).\n\n"
                                        "• Показывает, насколько хорошо PC1 описывает вариативность данных\n"
                                        "• Значение от 0 до 1 (или от 0% до 100%)\n"
                                        "• **Хорошие значения:** > 0.3 (30%)\n"
                                        "• **Отличные значения:** > 0.5 (50%)"
                                    )
                    except Exception:
                        pass
                
                # Загружаем признаки из session_state или из эксперимента
                selected_features_val = safe_session_get("selected_features", [])
                if selected_features_val:
                    # Используем сохраненные признаки, фильтруя только существующие
                    current_selected = [f for f in selected_features_val if f in feature_cols]
                    
                    # Показываем информацию о загруженных признаках
                    total_requested = len(selected_features_val)
                    total_available = len(current_selected)
                    
                    if total_available > 0:
                        if total_available < total_requested:
                            st.warning(f"⚠️ Доступно {total_available} из {total_requested} выбранных признаков. Некоторые признаки отсутствуют в данных.")
                        else:
                            st.success(f"✅ Выбрано {total_available} признаков")
                        with st.expander("📋 Показать выбранные признаки"):
                            for feat in sorted(current_selected):
                                st.text(f"  • {feat}")
                    else:
                        st.error(f"❌ Ни один из {total_requested} выбранных признаков не найден в данных!")
                        # Fallback: используем все доступные признаки
                        current_selected = feature_cols.copy()
                        st.info("💡 Используются все доступные признаки как fallback")
                else:
                    st.warning("⚠️ Не выбрано ни одного признака! Будут использованы все признаки.")
                    current_selected = feature_cols.copy()
                
                # ОПТИМИЗАЦИЯ: Применяем выбранные признаки к df_features только если они изменились
                # Кэшируем последний набор примененных признаков
                last_applied_features_key = "last_applied_features_for_df"
                last_applied_features = safe_session_get(last_applied_features_key, [])
                
                if current_selected != last_applied_features:
                    # Признаки изменились - применяем фильтрацию
                    if current_selected:
                        cols_to_keep = ["image"] + current_selected
                        available_cols = [col for col in cols_to_keep if col in df_features.columns]
                        df_features = df_features[available_cols]
                    safe_session_set(last_applied_features_key, current_selected.copy())
                else:
                    # Признаки не изменились - используем уже отфильтрованный df_features из кэша
                    # Применяем фильтрацию для совместимости (но это быстрая операция)
                    if current_selected:
                        cols_to_keep = ["image"] + current_selected
                        available_cols = [col for col in cols_to_keep if col in df_features.columns]
                        df_features = df_features[available_cols]
                
                # Вычисляем метрики для текущего набора признаков
                # Метрики пересчитываются при изменении признаков (даже если используется эксперимент)
                # Проверяем, были ли изменены признаки (через флаг или сравнение с сохраненными)
                features_were_changed = False
                
                # Проверка 1: флаг features_applied
                if safe_session_get("features_applied", False):
                    features_were_changed = True
                
                # Проверка 2: сравнение текущих признаков с сохраненными в эксперименте
                # Эта проверка нужна, чтобы определить, изменились ли признаки относительно оригинала из эксперимента
                if not features_were_changed and use_experiment_data:
                    experiment_config_cache = safe_session_get("experiment_config_cache")
                    if experiment_config_cache:
                        # experiment_config_cache может содержать 'features' или 'selected_features'
                        # Берем оригинальные признаки из эксперимента (до фильтрации)
                        experiment_original_features = set(
                            experiment_config_cache.get("selected_features", []) or
                            experiment_config_cache.get("features", [])
                        )
                        current_features_set = set(current_selected)
                        # Сравниваем с оригинальными признаками из эксперимента
                        if experiment_original_features and current_features_set != experiment_original_features:
                            features_were_changed = True
                
                # Проверка 3: сравнение с последними сохраненными метриками (если они есть)
                # Это самая важная проверка - сравниваем текущие признаки с теми, для которых были вычислены метрики
                if not features_were_changed:
                    metrics_features_val = safe_session_get("metrics_features", [])
                    if metrics_features_val:
                        # Сравниваем множества признаков (порядок не важен)
                        if set(current_selected) != set(metrics_features_val):
                            features_were_changed = True
                    elif safe_session_has("current_metrics"):
                        # Если метрики есть, но metrics_features нет - значит признаки могли измениться
                        # Пересчитываем для безопасности
                        features_were_changed = True
                
                # КРИТИЧНО: Метрики пересчитываются ТОЛЬКО если:
                # 1. Признаки были применены через кнопку (features_applied = True), ИЛИ
                # 2. Метрики еще не вычислены (первая загрузка), ИЛИ
                # 3. Не используется эксперимент (обычная загрузка данных)
                # НЕ пересчитываем метрики при простом изменении checkbox'ов без нажатия кнопки!
                should_recalculate_metrics = (
                    len(current_selected) > 0 and 
                    "image" in df_features.columns and
                    (
                        not use_experiment_data or  # Не используется эксперимент
                        features_were_changed or  # Или признаки были применены через кнопку
                        not safe_session_has("current_metrics")  # Или метрики еще не вычислены (первая загрузка)
                    )
                )
                
                
                if should_recalculate_metrics:
                    # Определяем mod и normal образцы из имен файлов
                    mod_samples = []
                    normal_samples = []
                    
                    for img_name in df_features["image"].unique():
                        sample_type = identify_sample_type(str(img_name))
                        if sample_type == 'mod':
                            mod_samples.append(img_name)
                        elif sample_type == 'normal':
                            normal_samples.append(img_name)
                    
                    # Вычисляем метрики только если есть и mod, и normal образцы
                    if len(mod_samples) > 0 and len(normal_samples) > 0:
                        try:
                            feature_cols_for_metrics = [col for col in current_selected if col in df_features.columns]
                            if len(feature_cols_for_metrics) > 0:
                                # КРИТИЧНО: Сортируем признаки для стабильности PCA
                                # Порядок признаков может влиять на PCA из-за численной нестабильности
                                sorted_feature_cols = sorted(feature_cols_for_metrics)
                                
                                
                                current_metrics = evaluate_feature_set(
                                    df_features,
                                    sorted_feature_cols,  # Используем отсортированные признаки
                                    mod_samples,
                                    normal_samples
                                )
                                
                                # Проверяем, что метрики валидны (не -inf)
                                if (current_metrics.get('score', -np.inf) != -np.inf and 
                                    current_metrics.get('separation', -np.inf) != -np.inf):
                                    
                                    # Сохраняем метрики в session_state для использования при сохранении эксперимента
                                    safe_session_set("current_metrics", current_metrics)
                                    # Сохраняем признаки, для которых были вычислены метрики
                                    safe_session_set("metrics_features", current_selected.copy())
                                    
                                    # Сбрасываем флаг features_applied после успешного пересчета метрик
                                    # Теперь проверка изменений будет работать через сравнение metrics_features
                                    if safe_session_has("features_applied"):
                                        safe_session_del("features_applied")
                                    
                                    # Определяем заголовок в зависимости от того, используется ли эксперимент
                                    if use_experiment_data:
                                        expander_title = "📊 Метрики качества текущего набора признаков (после изменений)"
                                    else:
                                        expander_title = "📊 Метрики качества текущего набора признаков"
                                    
                                    with st.expander(expander_title, expanded=True):
                                        score_val = current_metrics.get('score', 0)
                                        separation_val = current_metrics.get('separation', 0)
                                        mean_pc1_norm_mod_val = current_metrics.get('mean_pc1_norm_mod', 0)
                                        explained_variance_val = current_metrics.get('explained_variance', 0)
                                        
                                        st.info(
                                            f"**Статистика данных:**\n\n"
                                            f"• Всего образцов: {len(df_features)}\n"
                                            f"• Патологических (mod): {len(mod_samples)}\n"
                                            f"• Нормальных (normal): {len(normal_samples)}\n"
                                            f"• Выбрано признаков: {len(feature_cols_for_metrics)}"
                                        )
                                        
                                        st.markdown("---")
                                        
                                        # Score
                                        st.markdown("### Score (комплексная оценка)")
                                        st.metric("Score", f"{score_val:.4f}")
                                        st.info(
                                            "**Score** - комплексная оценка качества набора признаков:\n\n"
                                            "• 40% - разделение между группами (separation)\n"
                                            "• 30% - позиция mod образцов на нормализованной шкале (ближе к 1)\n"
                                            "• 30% - объясненная дисперсия PC1\n\n"
                                            "**Хорошие значения:** > 1.0"
                                        )
                                        
                                        st.markdown("---")
                                        
                                        # Separation
                                        st.markdown("### Separation (разделение групп)")
                                        st.metric("Separation", f"{separation_val:.4f}")
                                        st.info(
                                            "**Separation** - разница между средними значениями PC1 для патологических (mod) "
                                            "и нормальных (normal) образцов.\n\n"
                                            "• Чем больше значение, тем лучше разделение между группами\n"
                                            "• **Хорошие значения:** > 2.0\n"
                                            "• **Отличные значения:** > 4.0"
                                        )
                                        
                                        st.markdown("---")
                                        
                                        # Mod (норм. PC1)
                                        st.markdown("### Mod (норм. PC1)")
                                        st.metric("Mod (норм. PC1)", f"{mean_pc1_norm_mod_val:.4f}")
                                        st.info(
                                            "**Mod (норм. PC1)** - среднее нормализованное значение PC1 для патологических образцов.\n\n"
                                            "• Значение от 0 до 1 на нормализованной шкале\n"
                                            "• **Цель:** близко к 1.0 (патологические образцы должны иметь высокие значения PC1)\n"
                                            "• **Хорошие значения:** > 0.7\n"
                                            "• **Отличные значения:** > 0.85"
                                        )
                                        
                                        st.markdown("---")
                                        
                                        # Объясненная дисперсия
                                        st.markdown("### Объясненная дисперсия")
                                        st.metric("Объясненная дисперсия", f"{explained_variance_val:.4f}")
                                        st.info(
                                            "**Объясненная дисперсия** - доля дисперсии данных, объясняемая первой главной компонентой (PC1).\n\n"
                                            "• Показывает, насколько хорошо PC1 описывает вариативность данных\n"
                                            "• Значение от 0 до 1 (или от 0% до 100%)\n"
                                            "• **Хорошие значения:** > 0.3 (30%)\n"
                                            "• **Отличные значения:** > 0.5 (50%)"
                                        )
                        except Exception as e:
                            # Не показываем ошибку, просто не отображаем метрики
                            pass
                else:
                    # Метрики не пересчитываются - используем метрики из эксперимента, если они есть
                    if use_experiment_data and safe_session_has("experiment_config_cache"):
                        experiment_config = safe_session_get("experiment_config_cache")
                        if experiment_config.get("metrics") and not features_were_changed:
                            # Используем метрики из эксперимента
                            experiment_metrics = experiment_config.get("metrics", {})
                            safe_session_set("current_metrics", experiment_metrics)
                            safe_session_set("metrics_features", current_selected.copy())
                            
                            # Отображаем метрики из эксперимента
                            with st.expander("📊 Метрики качества набора признаков (из эксперимента)", expanded=True):
                                score_val = experiment_metrics.get('score', 0)
                                separation_val = experiment_metrics.get('separation', 0)
                                mean_pc1_norm_mod_val = experiment_metrics.get('mean_pc1_norm_mod', 0)
                                explained_variance_val = experiment_metrics.get('explained_variance', 0)
                                
                                st.info(
                                    f"**Статистика данных:**\n\n"
                                    f"• Всего образцов: {len(df_features)}\n"
                                    f"• Выбрано признаков: {len(current_selected)}\n"
                                    f"• Метрики из эксперимента (признаки не изменены)"
                                )
                                
                                st.markdown("---")
                                
                                # Score
                                st.markdown("### Score (комплексная оценка)")
                                st.metric("Score", f"{score_val:.4f}")
                                st.info(
                                    "**Score** - комплексная оценка качества набора признаков:\n\n"
                                    "• 40% - разделение между группами (separation)\n"
                                    "• 30% - позиция mod образцов на нормализованной шкале (ближе к 1)\n"
                                    "• 30% - объясненная дисперсия PC1\n\n"
                                    "**Хорошие значения:** > 1.0"
                                )
                                
                                st.markdown("---")
                                
                                # Separation
                                st.markdown("### Separation (разделение групп)")
                                st.metric("Separation", f"{separation_val:.4f}")
                                st.info(
                                    "**Separation** - разница между средними значениями PC1 для патологических (mod) "
                                    "и нормальных (normal) образцов.\n\n"
                                    "• Чем больше значение, тем лучше разделение между группами\n"
                                    "• **Хорошие значения:** > 2.0\n"
                                    "• **Отличные значения:** > 4.0"
                                )
                                
                                st.markdown("---")
                                
                                # Mod (норм. PC1)
                                st.markdown("### Mod (норм. PC1)")
                                st.metric("Mod (норм. PC1)", f"{mean_pc1_norm_mod_val:.4f}")
                                st.info(
                                    "**Mod (норм. PC1)** - среднее нормализованное значение PC1 для патологических образцов.\n\n"
                                    "• Значение от 0 до 1 на нормализованной шкале\n"
                                    "• **Цель:** близко к 1.0 (патологические образцы должны иметь высокие значения PC1)\n"
                                    "• **Хорошие значения:** > 0.7\n"
                                    "• **Отличные значения:** > 0.85"
                                )
                                
                                st.markdown("---")
                                
                                # Объясненная дисперсия
                                st.markdown("### Объясненная дисперсия")
                                st.metric("Объясненная дисперсия", f"{explained_variance_val:.4f}")
                                st.info(
                                    "**Объясненная дисперсия** - доля дисперсии данных, объясняемая первой главной компонентой (PC1).\n\n"
                                    "• Показывает, насколько хорошо PC1 описывает вариативность данных\n"
                                    "• Значение от 0 до 1 (или от 0% до 100%)\n"
                                    "• **Хорошие значения:** > 0.3 (30%)\n"
                                    "• **Отличные значения:** > 0.5 (50%)"
                                )
                
                # Рекомендации
                with st.expander("💡 Рекомендации по выбору признаков"):
                    st.markdown(f"""
                    **О количестве признаков:**
                    - **Относительные признаки:** ожидается 30 признаков (10 классов × 3 типа признаков, без Crypts)
                    - **Абсолютные признаки:** ожидается 22 признака (11 классов × 2 типа признаков: 10 патологических + 1 Crypts)
                    - Если вы видите другое количество, возможно, присутствуют дополнительные колонки из предыдущего анализа
                    - Подробнее см. [docs/FEATURES.md](docs/FEATURES.md)
                    
                    **Когда стоит исключать признаки (blacklist):**
                    
                    1. **Аномально высокие значения** (например, Meta_relative_count > 50)
                       - Могут доминировать в PCA и "перетягивать" образец в неправильную сторону
                    
                    2. **Признаки, которые мешают классификации**
                       - Если образец явно патологический, но получает низкий score
                    
                    **Когда использовать whitelist (только выбранные признаки):**
                    
                    1. **Фокус на патологических признаках**
                       - Используйте только Dysplasia, Mild, Moderate признаки
                       - Это может помочь, если другие признаки (Meta, Neutrophils) мешают
                    
                    2. **Для образца 6mod:**
                       - Нажмите кнопку "Только патология" для быстрого выбора
                       - Или выберите вручную: Dysplasia, Mild, Moderate признаки
                    
                    **После изменения признаков** кнопка "Применить признаки" появится автоматически. PCA пересчитывается при загрузке данных и после применения изменений.
                    """)
            else:
                st.info("Загрузите данные, чтобы выбрать признаки")

        with tab2:
            st.header("Распределения признаков")
            st.info("💡 Отображаются распределения только для признаков, выбранных в секции '🎯 Выбор признаков'.")

            if len(df_features) > 0:
                # Используем только признаки, выбранные в секции "🎯 Выбор признаков"
                selected_features_val = safe_session_get("selected_features", [])
                if selected_features_val:
                    selected_features = [
                        f for f in selected_features_val 
                        if f in df_features.columns
                    ]
                else:
                    selected_features = []

                if selected_features:
                    st.markdown(f"**Выбрано признаков для анализа: {len(selected_features)}**")
                    
                    cols = st.columns(2)

                    for idx, feature in enumerate(selected_features):
                        col = cols[idx % 2]

                        with col:
                            st.subheader(feature)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            # Дополнительная проверка на случай если колонка все еще отсутствует
                            if feature in df_features.columns:
                                ax.hist(
                                    df_features[feature].dropna(),
                                    bins=20,
                                    alpha=0.7,
                                    edgecolor="black",
                                )
                                ax.set_xlabel(feature)
                                ax.set_ylabel("Frequency")
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)  # Закрываем фигуру для предотвращения утечки памяти
                            else:
                                st.warning(f"Признак '{feature}' отсутствует в данных")
                else:
                    st.warning("⚠️ Не выбрано ни одного признака для анализа.")
                    st.info("💡 Перейдите в секцию '🎯 Выбор признаков' и выберите признаки для анализа.")

        with tab3:
            st.header("Спектральный анализ")
            
            # Подробное описание методов
            with st.expander("📚 Методы спектрального анализа: GMM и BIC", expanded=False):
                st.markdown("""
                ## 🔬 Методы аппроксимации распределения патологий
                
                ### 1. GMM (Gaussian Mixture Model) - Метод аппроксимации
                
                **GMM** - это параметрическая модель, которая аппроксимирует распределение данных как **смесь нескольких гауссовых (нормальных) распределений**.
                
                #### Формула:
                ```
                p(x) = Σ(i=1 to k) w_i × N(x | μ_i, σ_i²)
                ```
                
                где:
                - `k` - число компонентов (гауссовых распределений)
                - `w_i` - вес i-го компонента (Σw_i = 1)
                - `μ_i` - среднее значение i-го компонента
                - `σ_i` - стандартное отклонение i-го компонента
                - `N(x | μ_i, σ_i²)` - плотность нормального распределения
                
                #### Алгоритм обучения: EM (Expectation-Maximization)
                
                1. **E-шаг (Expectation)**: Вычисление вероятностей принадлежности каждой точки к каждому компоненту
                2. **M-шаг (Maximization)**: Обновление параметров (μ, σ, w) для максимизации правдоподобия
                3. **Итерации**: Алгоритм повторяет E и M шаги до сходимости
                
                #### Ссылки:
                - **Scikit-learn**: https://scikit-learn.org/stable/modules/mixture.html
                - **Bishop, C. M. (2006)**: *Pattern Recognition and Machine Learning*. Chapter 9
                - **Dempster et al. (1977)**: Maximum likelihood from incomplete data via the EM algorithm
                
                ---
                
                ### 2. BIC (Bayesian Information Criterion) - Критерий выбора числа компонентов
                
                **BIC** - это критерий для выбора оптимального числа компонентов в GMM, который **балансирует точность модели и её сложность**.
                
                #### Формула:
                ```
                BIC = -2 × log_likelihood + k × log(n)
                ```
                
                где:
                - `log_likelihood` - логарифм правдоподобия модели (чем больше, тем лучше)
                - `k` - число параметров модели (для GMM: 3k - 1)
                - `n` - число образцов (точек данных)
                - `log(n)` - штраф за сложность
                
                #### Интерпретация:
                
                - **Меньше BIC = лучше модель**
                - Первое слагаемое: штраф за плохое соответствие данным
                - Второе слагаемое: штраф за сложность модели (больше параметров = больше штраф)
                - BIC склонен выбирать **более простые модели**, чтобы избежать переобучения
                
                #### Алгоритм выбора:
                
                ```
                Для каждого числа компонентов k от 1 до max_components:
                    1. Обучить GMM с k компонентами (EM-алгоритм)
                    2. Вычислить BIC для этой модели
                    3. Сохранить k с минимальным BIC
                
                Выбрать k с минимальным BIC
                ```
                
                #### Ссылки:
                - **Schwarz, G. (1978)**: Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464
                - **Burnham & Anderson (2004)**: *Model Selection and Multimodel Inference*
                - Подробное объяснение: см. `GMM_BIC_EXPLANATION.md`
                
                ---
                
                ### 3. Связь между GMM и BIC
                
                - **GMM** - это метод аппроксимации (как аппроксимировать данные)
                - **BIC** - это критерий выбора (сколько компонентов использовать)
                
                ### 4. Сравнение с альтернативными методами
                
                - **KDE (Kernel Density Estimation)**: Непараметрический метод, используется для сравнения с GMM
                - **AIC (Akaike Information Criterion)**: Меньший штраф за сложность, склонен выбирать больше компонентов
                - **Cross-Validation**: Разделение на train/validation для оценки качества
                
                ---
                
                ### 📊 Метрики качества аппроксимации:
                
                - **RMSE**: Среднеквадратичная ошибка между KDE и GMM (меньше = лучше)
                - **R²**: Коэффициент детерминации (ближе к 1 = лучше)
                - **BIC**: Баланс точности и сложности (меньше = лучше)
                - **Max Error**: Максимальная локальная ошибка (показывает худший случай)
                
                **Рекомендация**: Для практических целей (классификация, шкала) используйте число компонентов с **лучшим RMSE**, так как это означает более точное описание реальной структуры данных.
                """)

            if use_spectral_analysis and len(df_features) > 0:
                # ОПТИМИЗАЦИЯ: Используем стабильный ключ на основе выбранных признаков и настроек
                # вместо хэша всего DataFrame, который может изменяться при каждом RUN
                
                
                # Получаем список признаков, которые будут использоваться для PCA
                # ПРИОРИТЕТ: 1) Признаки из эксперимента, 2) selected_features из session_state, 3) Все числовые колонки
                feature_columns_for_key = None
                
                if use_experiment_data and safe_session_has("experiment_config_cache"):
                    experiment_config = safe_session_get("experiment_config_cache")
                    if experiment_config:
                        experiment_features = experiment_config.get('features', [])
                        if experiment_features:
                            # Используем признаки из эксперимента - это самый стабильный вариант
                            feature_columns_for_key = sorted(experiment_features)
                
                # Если не нашли в эксперименте, проверяем selected_features
                if feature_columns_for_key is None:
                    selected_features_val = safe_session_get("selected_features")
                    if selected_features_val:
                        feature_columns_for_key = sorted(selected_features_val)
                
                # Если все еще нет, используем все числовые колонки из df_features
                if feature_columns_for_key is None:
                    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
                    if "image" in numeric_cols:
                        numeric_cols.remove("image")
                    feature_columns_for_key = sorted(numeric_cols) if numeric_cols else []
                
                # КРИТИЧНО: Используем стабильный ключ на основе признаков
                # НИКОГДА не используем len(df_features) или другие нестабильные метрики
                if feature_columns_for_key:
                    features_hash = hash(tuple(feature_columns_for_key))
                else:
                    # Fallback: используем имя эксперимента как идентификатор (если есть)
                    if use_experiment_data and safe_session_has("experiment_name"):
                        exp_name = safe_session_get("experiment_name")
                        features_hash = hash(f"experiment_{exp_name}")
                    else:
                        # Последний fallback: используем фиксированное значение
                        features_hash = hash("default_features")
                
                spectral_settings_key = f"spectral_{features_hash}_{percentile_low}_{percentile_high}"
                
                # Проверяем, нужно ли переобучить анализатор
                # Переобучаем только если:
                # 1. Анализатора нет в session_state
                # 2. Изменился ключ настроек (признаки или процентили)
                # 3. Явно установлен флаг features_applied (признаки были изменены через кнопку)
                has_analyzer = safe_session_has("analyzer")
                has_settings_key = safe_session_has("spectral_settings_key")
                settings_match = safe_session_get("spectral_settings_key") == spectral_settings_key if has_settings_key else False
                features_applied = safe_session_get("features_applied", False)
                
                need_retrain = (
                    not has_analyzer or
                    not has_settings_key or
                    not settings_match or
                    features_applied
                )
                
                debug_info = []
                if not has_analyzer:
                    debug_info.append("нет анализатора")
                    if not has_settings_key:
                        debug_info.append("нет ключа настроек")
                    if has_settings_key and not settings_match:
                        cached_key = safe_session_get("spectral_settings_key")
                        debug_info.append(f"ключ не совпадает: {cached_key} != {spectral_settings_key}")
                    if features_applied:
                        debug_info.append("признаки изменены")
                    
                if DEBUG_MODE:
                    if need_retrain:
                        logger.debug(f"⚠️ ПЕРЕОБУЧЕНИЕ НЕОБХОДИМО: {', '.join(debug_info) if debug_info else 'неизвестная причина'}")
                    else:
                        logger.debug(f"✅ Переобучение НЕ необходимо. Ключ совпадает: {spectral_settings_key}")
                
                # Если загружаем данные из эксперимента, пытаемся загрузить сохраненную модель
                analyzer_loaded_from_experiment = False
                if use_experiment_data and safe_session_has("experiment_dir") and need_retrain:
                    experiment_dir = Path(safe_session_get("experiment_dir"))
                    model_path = experiment_dir / "spectral_analyzer.pkl"
                    if model_path.exists():
                        try:
                            analyzer = spectral_analysis.SpectralAnalyzer()
                            analyzer.load(model_path)
                            # Проверяем, что модель совместима с текущими данными
                            if analyzer.feature_columns is not None:
                                # Проверяем, что все признаки модели есть в текущих данных
                                missing_features = [f for f in analyzer.feature_columns if f not in df_features.columns]
                                if not missing_features:
                                    analyzer_loaded_from_experiment = True
                                    st.info(f"✅ Загружена модель PCA из эксперимента (использовано {len(analyzer.feature_columns)} признаков)")
                                else:
                                    st.warning(f"⚠️ Модель из эксперимента использует признаки, которых нет в данных: {missing_features}")
                        except Exception as e:
                            st.warning(f"⚠️ Не удалось загрузить модель из эксперимента: {e}")
                
                if analyzer_loaded_from_experiment:
                    # ОПТИМИЗАЦИЯ: Проверяем, можно ли использовать уже вычисленный спектр
                    # Если модель загружена и настройки совпадают, можно пропустить fit_spectrum
                    spectrum_already_fitted = (
                        analyzer.modes is not None and 
                        analyzer.pc1_percentiles is not None and
                        len(analyzer.modes) > 0
                    )
                    
                    # Модель загружена из эксперимента, нужно вычислить df_pca и fit_spectrum
                    # Используем полный набор данных, чтобы все признаки из модели были доступны
                    if use_relative_features and safe_session_has("df_features_full") and safe_session_get("df_features_full") is not None:
                        df_for_transform = safe_session_get("df_features_full")
                    elif safe_session_has("df_features_for_selection") and safe_session_get("df_features_for_selection") is not None:
                        df_for_transform = safe_session_get("df_features_for_selection")
                    else:
                        df_for_transform = df_features
                    
                    df_pca = analyzer.transform_pca(df_for_transform)
                    
                    # ОПТИМИЗАЦИЯ: Пересчитываем спектр только если он не загружен или настройки изменились
                    if not spectrum_already_fitted:
                        analyzer.fit_spectrum(
                            df_pca,
                            percentile_low=percentile_low,
                            percentile_high=percentile_high,
                        )
                    else:
                        # Используем загруженный спектр из модели
                        pass
                    
                    # Сохраняем в session_state
                    safe_session_set("analyzer", analyzer)
                    safe_session_set("df_pca", df_pca)
                    safe_session_set("spectral_settings_key", spectral_settings_key)
                    # Сохраняем ключ признаков для отслеживания изменений
                    if feature_columns_for_key:
                        safe_session_set("features_key", hash(tuple(feature_columns_for_key)))
                    # Очищаем кэш GMM качества, так как PCA изменился
                    try:
                        # Используем безопасный способ получения всех ключей
                        all_keys = []
                        if safe_session_has("_temp_keys_check"):
                            # Если session_state доступен, получаем ключи безопасно
                            try:
                                if hasattr(st, 'session_state'):
                                    _ = st.session_state  # Проверка доступности
                                    all_keys = list(st.session_state.keys())
                            except (RuntimeError, AttributeError):
                                pass
                        cache_keys_to_remove = [key for key in all_keys if key.startswith("gmm_quality_")]
                        for key in cache_keys_to_remove:
                            safe_session_del(key)
                    except (RuntimeError, AttributeError):
                        pass
                    
                    # Вычисляем спектр для сохранения в кэш (даже если он уже был загружен)
                    # Это нужно для того, чтобы при переходе на вкладку спектр был доступен
                    use_gmm_for_spectrum = safe_session_get("use_gmm_spectral", True)
                    use_gmm_classification_for_spectrum = safe_session_get("use_gmm_classification", False) if use_gmm_for_spectrum and analyzer.gmm is not None else False
                    
                    # Вычисляем спектр сразу после загрузки модели
                    df_spectrum_loaded = analyzer.transform_to_spectrum(df_pca, use_gmm_classification=use_gmm_classification_for_spectrum)
                    
                    # Сохраняем спектр в кэш сразу после загрузки модели
                    gmm_n_components_for_cache = analyzer.gmm.n_components if analyzer.gmm is not None else 0
                    spectrum_cache_key_loaded = f"spectrum_{spectral_settings_key}_{use_gmm_for_spectrum}_{use_gmm_classification_for_spectrum}_{gmm_n_components_for_cache}"
                    safe_session_set("df_spectrum", df_spectrum_loaded)
                    safe_session_set("spectrum_cache_key", spectrum_cache_key_loaded)
                    
                
                if need_retrain and not analyzer_loaded_from_experiment:
                    # Обучение спектрального анализатора
                    with st.spinner("Обучение спектрального анализатора..."):
                        analyzer = spectral_analysis.SpectralAnalyzer()

                        # Если загружаем данные из эксперимента, используем признаки из конфигурации эксперимента
                        if use_experiment_data and safe_session_has("experiment_config_cache"):
                            experiment_config = safe_session_get("experiment_config_cache")
                            experiment_features = experiment_config.get('features', [])
                            if experiment_features:
                                # Используем ТОЧНО те же признаки, что были в эксперименте
                                # Проверяем против полного набора признаков (df_features_full или df_features_for_selection),
                                # а не против уже отфильтрованного df_features
                                df_features_full_val = safe_session_get("df_features_full")
                                df_features_for_selection_val = safe_session_get("df_features_for_selection")
                                if use_relative_features and df_features_full_val is not None:
                                    check_against_df = df_features_full_val
                                elif df_features_for_selection_val is not None:
                                    check_against_df = df_features_for_selection_val
                                else:
                                    check_against_df = df_features
                                
                                available_experiment_features = [
                                    f for f in experiment_features 
                                    if f in check_against_df.columns
                                ]
                                if available_experiment_features:
                                    feature_columns_for_pca = available_experiment_features
                                    st.info(f"💡 Используются признаки из эксперимента: {len(feature_columns_for_pca)} признаков")
                                    # Показываем, какие признаки используются (включая структурные, если есть)
                                    structural_in_pca = [f for f in feature_columns_for_pca if any(x in f.lower() for x in ['surface epithelium', 'muscularis mucosae'])]
                                    if structural_in_pca:
                                        st.info(f"   Включая структурные признаки: {', '.join(structural_in_pca)}")
                                else:
                                    # Fallback: используем все числовые колонки
                                    feature_columns_for_pca = [
                                        col for col in df_features.select_dtypes(include=[np.number]).columns
                                        if col != "image"
                                    ]
                                    st.warning("⚠️ Признаки из эксперимента не найдены в данных, используются все доступные")
                            else:
                                # Fallback: используем все числовые колонки
                                feature_columns_for_pca = [
                                    col for col in df_features.select_dtypes(include=[np.number]).columns
                                    if col != "image"
                                ]
                        else:
                            # Обычный режим: используем все числовые колонки из df_features (включая структурные, если они выбраны)
                            feature_columns_for_pca = [
                                col for col in df_features.select_dtypes(include=[np.number]).columns
                                if col != "image"
                            ]
                        
                        # Для обучения PCA используем полный набор данных, если загружаем эксперимент
                        # Это гарантирует, что все признаки из эксперимента будут доступны
                        if use_experiment_data and safe_session_has("experiment_config_cache"):
                            df_features_full_val = safe_session_get("df_features_full")
                            df_features_for_selection_val = safe_session_get("df_features_for_selection")
                            if use_relative_features and df_features_full_val is not None:
                                df_for_pca = df_features_full_val
                            elif df_features_for_selection_val is not None:
                                df_for_pca = df_features_for_selection_val
                            else:
                                df_for_pca = df_features
                        else:
                            df_for_pca = df_features
                        
                        analyzer.fit_pca(df_for_pca, feature_columns=feature_columns_for_pca)

                        # Преобразование через PCA - используем тот же DataFrame, что и при обучении
                        # чтобы все признаки были доступны
                        df_pca = analyzer.transform_pca(df_for_pca)

                        # Анализ спектра
                        analyzer.fit_spectrum(
                            df_pca,
                            percentile_low=percentile_low,
                            percentile_high=percentile_high,
                        )
                        
                        # Сохраняем в session_state
                        safe_session_set("analyzer", analyzer)
                        safe_session_set("df_pca", df_pca)
                        safe_session_set("spectral_settings_key", spectral_settings_key)
                        # Сохраняем ключ признаков для отслеживания изменений
                        if feature_columns_for_key:
                            safe_session_set("features_key", hash(tuple(feature_columns_for_key)))
                        # Очищаем флаг принудительного пересчета
                        # Очищаем флаг применения признаков после пересчета PCA
                        safe_session_del("features_applied")
                        # Очищаем кэш GMM качества, так как PCA изменился
                        try:
                            # Безопасное получение списка ключей
                            all_keys = list(st.session_state.keys()) if hasattr(st, 'session_state') else []
                            cache_keys_to_remove = [key for key in all_keys if key.startswith("gmm_quality_")]
                            for key in cache_keys_to_remove:
                                safe_session_del(key)
                        except (RuntimeError, AttributeError):
                            pass
                        # Очищаем сохраненный спектр
                        safe_session_del("df_spectrum")
                else:
                    # Используем сохраненный анализатор
                    analyzer = safe_session_get("analyzer")
                    df_pca = safe_session_get("df_pca")
                    
                    if analyzer is None or df_pca is None:
                        st.error("❌ Ошибка: анализатор или df_pca отсутствуют в session_state")
                        st.stop()
                    
                    # КРИТИЧНО: Проверяем, что ключ совпадает с сохраненным
                    # Если не совпадает, это означает, что что-то изменилось и нужно пересчитать
                    if not safe_session_has("spectral_settings_key"):
                        # Если ключа нет, устанавливаем текущий (не должно происходить в нормальной работе)
                        safe_session_set("spectral_settings_key", spectral_settings_key)
                    elif safe_session_get("spectral_settings_key") != spectral_settings_key:
                        # Если ключ не совпадает, но анализатор есть - это проблема
                        # Это может означать, что признаки изменились, но анализатор не пересчитан
                        # Для безопасности: если нет явного флага features_applied, просто обновляем ключ
                        # Это предотвратит бесконечный цикл rerun
                        if not safe_session_get("features_applied", False):
                            # Обновляем ключ без пересчета - это предотвратит rerun
                            safe_session_set("spectral_settings_key", spectral_settings_key)
                        # Если features_applied установлен, need_retrain уже True и пересчет произойдет
                
                # ОПТИМИЗАЦИЯ: Используем стабильный ключ для кэширования GMM качества
                # Вместо хэша данных используем настройки и количество точек
                # Оценка качества GMM (BIC) - выполняется автоматически при спектральном анализе
                # Кэширование результатов оценки качества
                n_samples = len(df_pca)
                gmm_quality_key = f"gmm_quality_{n_samples}_{percentile_low}_{percentile_high}_{spectral_settings_key}"
                cache_key = gmm_quality_key
                if not safe_session_has(cache_key):
                    with st.spinner("Вычисление метрик качества GMM (BIC) для определения оптимального числа компонентов..."):
                        try:
                            # Ограничиваем max_components для ускорения (5 вместо 10)
                            quality_df = analyzer.evaluate_gmm_quality(df_pca, max_components=5)
                            safe_session_set(cache_key, quality_df)
                        except Exception as e:
                            st.warning(f"Не удалось оценить качество: {e}")
                            quality_df = pd.DataFrame()
                else:
                    quality_df = safe_session_get(cache_key)
                
                # Показываем результаты BIC для информации
                optimal_components = 2  # Значение по умолчанию
                if not quality_df.empty:
                    best_bic_idx = quality_df["BIC"].idxmin()
                    optimal_components = int(quality_df.loc[best_bic_idx, "Число компонентов"])
                    optimal_bic = quality_df.loc[best_bic_idx, "BIC"]
                    
                    st.info(f"📊 **BIC анализ:** Оптимальное число компонентов GMM = **{optimal_components}** (BIC={optimal_bic:.1f})")
                    
                    with st.expander("🔍 Подробный анализ качества аппроксимации GMM (BIC, RMSE, R²)"):
                        st.markdown("**Оцените качество аппроксимации для разного числа компонентов:**")
                        
                        try:
                            if not quality_df.empty:
                                st.dataframe(quality_df, use_container_width=True, hide_index=True)
                                
                                # График метрик качества
                                fig_quality, axes = plt.subplots(2, 2, figsize=(14, 10))
                                
                                n_components = quality_df["Число компонентов"]
                                
                                # BIC
                                axes[0, 0].plot(n_components, quality_df["BIC"], 'o-', linewidth=2, markersize=8)
                                axes[0, 0].set_xlabel("Число компонентов")
                                axes[0, 0].set_ylabel("BIC")
                                axes[0, 0].set_title("BIC (меньше = лучше)")
                                axes[0, 0].grid(True, alpha=0.3)
                                
                                # RMSE
                                axes[0, 1].plot(n_components, quality_df["RMSE"], 'o-', linewidth=2, markersize=8, color='red')
                                axes[0, 1].set_xlabel("Число компонентов")
                                axes[0, 1].set_ylabel("RMSE")
                                axes[0, 1].set_title("RMSE (меньше = лучше)")
                                axes[0, 1].grid(True, alpha=0.3)
                                
                                # R²
                                axes[1, 0].plot(n_components, quality_df["R²"], 'o-', linewidth=2, markersize=8, color='green')
                                axes[1, 0].set_xlabel("Число компонентов")
                                axes[1, 0].set_ylabel("R²")
                                axes[1, 0].set_title("R² (больше = лучше)")
                                axes[1, 0].grid(True, alpha=0.3)
                                
                                # Max Error
                                axes[1, 1].plot(n_components, quality_df["Max Error"], 'o-', linewidth=2, markersize=8, color='orange')
                                axes[1, 1].set_xlabel("Число компонентов")
                                axes[1, 1].set_ylabel("Max Error")
                                axes[1, 1].set_title("Максимальная ошибка")
                                axes[1, 1].grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig_quality)
                                plt.close(fig_quality)
                                
                                # Рекомендация
                                best_rmse_idx = quality_df["RMSE"].idxmin()
                                best_bic_idx = quality_df["BIC"].idxmin()
                                best_r2_idx = quality_df["R²"].idxmax()
                                
                                # Сравнение RMSE для 2 и 3 компонентов
                                rmse_2 = None
                                rmse_3 = None
                                if 2 in quality_df["Число компонентов"].values:
                                    rmse_2 = quality_df[quality_df["Число компонентов"] == 2]["RMSE"].values[0]
                                if 3 in quality_df["Число компонентов"].values:
                                    rmse_3 = quality_df[quality_df["Число компонентов"] == 3]["RMSE"].values[0]
                                
                                comparison_text = ""
                                if rmse_2 is not None and rmse_3 is not None:
                                    improvement = ((rmse_2 - rmse_3) / rmse_2) * 100
                                    if rmse_3 < rmse_2:
                                        comparison_text = f"\n\n**Сравнение 2 vs 3 компонентов:**\n- 2 компонента: RMSE={rmse_2:.4f}\n- 3 компонента: RMSE={rmse_3:.4f}\n- **Улучшение на {improvement:.1f}%** при использовании 3 компонентов ✅"
                                    else:
                                        comparison_text = f"\n\n**Сравнение 2 vs 3 компонентов:**\n- 2 компонента: RMSE={rmse_2:.4f}\n- 3 компонента: RMSE={rmse_3:.4f}\n- 2 компонента дают лучший RMSE, но BIC может выбрать другое число"
                                
                                st.info(f"""
                                **Рекомендации:**
                                - По RMSE: {int(quality_df.loc[best_rmse_idx, "Число компонентов"])} компонентов (RMSE={quality_df.loc[best_rmse_idx, "RMSE"]:.4f})
                                - По BIC: {int(quality_df.loc[best_bic_idx, "Число компонентов"])} компонентов (BIC={quality_df.loc[best_bic_idx, "BIC"]:.1f})
                                - По R²: {int(quality_df.loc[best_r2_idx, "Число компонентов"])} компонентов (R²={quality_df.loc[best_r2_idx, "R²"]:.4f})
                                {comparison_text}
                                
                                **Интерпретация:**
                                - **RMSE** показывает точность аппроксимации (меньше = лучше)
                                - **BIC** балансирует точность и сложность модели (меньше = лучше, но может выбрать меньше компонентов)
                                - Если RMSE лучше с 3 компонентами, но BIC выбрал 2 - это означает компромисс между точностью и простотой модели
                                - Для практических целей (классификация, шкала) можно использовать число компонентов с лучшим RMSE
                                """)
                        except Exception as e:
                            st.warning(f"Не удалось оценить качество: {e}")
                
                # GMM (опционально) - выполняется независимо от переобучения
                # КРИТИЧНО: Используем key для стабильного значения checkbox
                # Streamlit сам управляет значением через key, не нужно сохранять вручную
                use_gmm_key = "use_gmm_spectral"
                default_use_gmm = safe_session_get(use_gmm_key, True)
                use_gmm = st.checkbox("Использовать GMM для моделирования состояний", value=default_use_gmm, key=use_gmm_key)
                # Значение автоматически сохраняется Streamlit через key
                if use_gmm:
                    # Определяем оптимальное число компонентов по BIC (если доступно)
                    default_n_components = optimal_components if not quality_df.empty else 2
                    
                    # Выбор числа компонентов
                    auto_components = st.checkbox(
                        "Автоматический выбор числа компонентов (BIC)",
                        value=True,
                        help=f"Если включено, используется оптимальное число компонентов по BIC ({default_n_components}). Если выключено, можно задать число компонентов вручную"
                    )
                    
                    n_components = None
                    if not auto_components:
                        n_components = st.slider(
                            "Число компонентов GMM",
                            min_value=1,
                            max_value=min(10, len(df_pca) // 2),
                            value=default_n_components,
                            help="Увеличьте число компонентов для лучшей аппроксимации, но осторожно с переобучением"
                        )
                    
                    analyzer.fit_gmm(df_pca, n_components=n_components)
                    
                    # Показываем информацию о выбранном числе компонентов
                    if analyzer.gmm is not None:
                        st.success(f"✅ GMM обучен с {analyzer.gmm.n_components} компонентами")
                    # Обновляем анализатор в session_state после обучения GMM
                    safe_session_set("analyzer", analyzer)

                # Опция выбора метода классификации
                # КРИТИЧНО: Используем key для стабильного значения checkbox
                # Streamlit сам управляет значением через key, не нужно сохранять вручную
                use_gmm_classification_key = "use_gmm_classification"
                if use_gmm and analyzer.gmm is not None:
                    default_gmm_classification = safe_session_get(use_gmm_classification_key, False)
                    use_gmm_classification = st.checkbox(
                        "Использовать GMM компоненты для классификации образцов",
                        value=default_gmm_classification,
                        key=use_gmm_classification_key,
                        help="Если включено, образцы классифицируются по принадлежности к GMM компонентам. "
                             "Если выключено, используется фиксированное разделение на 4 категории (normal/mild/moderate/severe) "
                             "на основе позиции на спектральной шкале."
                    )
                    # Значение автоматически сохраняется Streamlit через key
                else:
                    use_gmm_classification = False
                    # Если GMM не используется, виджет не создан, поэтому можно установить значение напрямую
                    # Но только если виджет еще не был создан ранее
                    if not safe_session_has(use_gmm_classification_key):
                        # Если виджет еще не создан, можно установить значение
                        safe_session_set(use_gmm_classification_key, False)
                
                # ОПТИМИЗАЦИЯ: Используем стабильный ключ для кэширования спектра
                # Определяем ключ для кэширования спектра на основе настроек GMM и спектральных настроек
                gmm_n_components = analyzer.gmm.n_components if analyzer.gmm is not None else 0
                spectrum_cache_key = f"spectrum_{spectral_settings_key}_{use_gmm}_{use_gmm_classification}_{gmm_n_components}"
                
                
                # Пересчитываем спектр только если:
                # 1. Его нет в кэше
                # 2. Изменился ключ кэша (настройки GMM или спектральные настройки)
                cached_spectrum_key = safe_session_get("spectrum_cache_key")
                cached_df_spectrum = safe_session_get("df_spectrum")
                
                
                need_recalculate_spectrum = (
                    cached_df_spectrum is None or
                    cached_spectrum_key != spectrum_cache_key
                )
                
                if need_recalculate_spectrum:
                    # Преобразование в спектральную шкалу
                    with st.spinner("Вычисление спектральной шкалы..."):
                        df_spectrum = analyzer.transform_to_spectrum(df_pca, use_gmm_classification=use_gmm_classification if use_gmm else False)
                    
                    # Сохраняем в session_state с ключом
                    safe_session_set("df_spectrum", df_spectrum)
                    safe_session_set("spectrum_cache_key", spectrum_cache_key)
                else:
                    # Используем сохраненный спектр
                    df_spectrum = cached_df_spectrum
                
                # Обновляем анализатор в session_state (на случай если был обучен GMM)
                safe_session_set("analyzer", analyzer)

                # Информация о спектре
                spectrum_info = analyzer.get_spectrum_info()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Число мод", spectrum_info["n_modes"])
                with col2:
                    st.metric(
                        "PC1 медиана",
                        f"{spectrum_info['percentiles']['median']:.2f}",
                    )
                with col3:
                    st.metric(
                        "PC1 std",
                        f"{spectrum_info['percentiles']['std']:.2f}",
                    )
                with col4:
                    if "gmm_components" in spectrum_info:
                        st.metric("GMM компонентов", spectrum_info["gmm_components"])
                    else:
                        st.metric("GMM компонентов", "Не обучен")

                # Визуализация спектра
                st.subheader("Визуализация спектра")
                
                with st.expander("🔬 Как вычисляется PC1 для конкретного WSI?"):
                    st.markdown("""
                    ## Вычисление PC1 для конкретного WSI
                    
                    После обучения PCA модели, для каждого WSI вычисляется значение PC1 следующим образом:
                    
                    ### Шаг 1: Извлечение признаков WSI
                    ```
                    X_wsi = [признак₁, признак₂, ..., признакₙ]
                    ```
                    Например, для WSI "image_001.tif":
                    ```
                    X_wsi = [Mild_relative_count=0.5, Dysplasia_relative_area=1.2, ..., Paneth_mean_relative_area=0.3]
                    ```
                    
                    ### Шаг 2: Стандартизация признаков WSI
                    Используются те же параметры стандартизации (μ и σ), что были вычислены при обучении:
                    ```
                    X_wsi_scaled[i] = (X_wsi[i] - μᵢ) / σᵢ
                    ```
                    Где:
                    - `μᵢ` - среднее значение i-го признака из обучающей выборки
                    - `σᵢ` - стандартное отклонение i-го признака из обучающей выборки
                    
                    **Важно:** Используются параметры из обучения, а не пересчитываются заново!
                    
                    ### Шаг 3: Вычисление PC1
                    ```
                    PC1(wsi) = loading₁ × X_wsi_scaled[1] + loading₂ × X_wsi_scaled[2] + ... + loadingₙ × X_wsi_scaled[n]
                    ```
                    
                    Или в матричной форме:
                    ```
                    PC1(wsi) = loadings^T × X_wsi_scaled
                    ```
                    Где `loadings` - вектор loadings первой главной компоненты (из `pca.components_[0]`)
                    
                    ### Шаг 4: Нормализация PC1 (опционально)
                    Для получения шкалы от 0 до 1:
                    ```
                    PC1_norm(wsi) = (PC1(wsi) - PC1_min) / (PC1_max - PC1_min)
                    ```
                    Где `PC1_min` и `PC1_max` - минимальное и максимальное значения PC1 из обучающей выборки
                    
                    ---
                    
                    **Пример вычисления PC1 для конкретного WSI:**
                    
                    Предположим, у нас есть WSI со следующими признаками:
                    ```
                    Mild_relative_count = 0.8
                    Dysplasia_relative_area = 1.5
                    Crypts_count = 100
                    ```
                    
                    После стандартизации (используя μ и σ из обучения):
                    ```
                    Mild_relative_count_scaled = (0.8 - 0.5) / 0.3 = 1.0
                    Dysplasia_relative_area_scaled = (1.5 - 1.0) / 0.5 = 1.0
                    Crypts_count_scaled = (100 - 120) / 20 = -1.0
                    ```
                    
                    Если loadings:
                    ```
                    Mild_relative_count: loading = +0.25
                    Dysplasia_relative_area: loading = +0.30
                    Crypts_count: loading = -0.10
                    ```
                    
                    Тогда PC1 вычисляется как:
                    ```
                    PC1(wsi) = (0.25 × 1.0) + (0.30 × 1.0) + (-0.10 × -1.0)
                             = 0.25 + 0.30 + 0.10
                             = 0.65
                    ```
                    
                    Если PC1_min = -2.0 и PC1_max = 3.0 из обучающей выборки:
                    ```
                    PC1_norm(wsi) = (0.65 - (-2.0)) / (3.0 - (-2.0))
                                 = 2.65 / 5.0
                                 = 0.53
                    ```
                    
                    **Интерпретация:** WSI имеет PC1_norm = 0.53, что означает средний уровень патологии (ближе к середине шкалы).
                    
                    ---
                    
                    **Где это происходит в коде:**
                    
                    В `scale/pca_scoring.py`, метод `transform()`:
                    ```python
                    # 1. Извлечение признаков
                    X = df[feature_columns].fillna(0).values
                    
                    # 2. Стандартизация (используя параметры из обучения)
                    X_scaled = self.scaler.transform(X)  # ← использует self.scaler.mean_ и self.scaler.scale_
                    
                    # 3. Вычисление PC1 (используя loadings из обучения)
                    X_pca = self.pca.transform(X_scaled)  # ← использует self.pca.components_
                    PC1 = X_pca[:, 0]  # Первая колонка = PC1 для каждого образца
                    
                    # 4. Нормализация
                    PC1_norm = (PC1 - self.pc1_min) / (self.pc1_max - self.pc1_min)
                    ```
                    
                    **Ключевой момент:** Все параметры (μ, σ, loadings, PC1_min, PC1_max) фиксируются при обучении (`fit()`) и используются для всех последующих WSI (`transform()`).
                    
                    Подробнее см. [docs/PCA.md](docs/PCA.md)
                    """)
                
                with st.expander("ℹ️ Как интерпретировать графики спектра?"):
                    st.markdown("""
                    ## 🔗 Связь между двумя графиками
                    
                    **Это одно и то же распределение, но в разных масштабах:**
                    
                    1. **Верхний график** показывает **сырые значения PC1** (результат PCA анализа)
                    2. **Нижний график** показывает **нормализованные значения 0-1** (спектральная шкала)
                    
                    **Преобразование:**
                    ```
                    PC1_spectrum = (PC1 - P1) / (P99 - P1)
                    ```
                    - P1 (1-й процентиль) → становится 0
                    - P99 (99-й процентиль) → становится 1
                    - Все значения между P1 и P99 → масштабируются линейно в диапазон [0, 1]
                    - Значения за пределами P1-P99 → обрезаются до 0 или 1
                    
                    **Почему это нужно?**
                    - Сырые значения PC1 зависят от конкретных данных (могут быть, например, от -5 до +10)
                    - Нормализованная шкала 0-1 универсальна и интерпретируема:
                      - 0 = минимальная патология (ближе к норме)
                      - 1 = максимальная патология
                    
                    ---
                    
                    **Верхний график: Распределение PC1 (сырые значения)**
                    
                    - **Синяя линия (KDE)**: KDE = Kernel Density Estimation (Оценка плотности ядерным методом)
                      - 📌 **ОПРЕДЕЛЕНИЕ KDE**: Непараметрический метод оценки плотности распределения данных
                      - 📐 **ФОРМУЛА**: KDE(x) = (1/n) × Σ K((x - x_i) / h)
                      - 🔧 **КАК ВЫЧИСЛЯЕТСЯ**: На каждую точку PC1 размещается гауссово ядро, все ядра суммируются
                      - Пики = области с высокой концентрацией образцов
                      - Широкое распределение = большой разброс патологий
                      - Узкое распределение = образцы похожи друг на друга
                    
                    - **Гистограмма (серые столбцы)**: Реальное распределение ваших данных
                      - Показывает, сколько образцов попадает в каждый диапазон PC1
                      - Ось Y = **Density** (плотность, нормализованная)
                    
                    - **Красные пунктирные линии**: МОДЫ (стабильные состояния)
                      - 📌 **ОПРЕДЕЛЕНИЕ МОДЫ**: Мода = локальный максимум плотности распределения PC1
                      - 🔍 **МЕТОД ПОИСКА**: Peak Detection Algorithm (Алгоритм обнаружения пиков)
                      - 📐 **АЛГОРИТМ**: KDE (Kernel Density Estimation) → find_peaks (scipy.signal) → позиция максимума
                      - Мода = позиция на шкале PC1, где концентрация образцов максимальна
                      - ⚠️ **ВАЖНО**: Моды НЕ обязательно разделяют нормальные и патологические образцы
                      - Моды = локальные максимумы плотности = центры кластеров похожих образцов
                      - Каждая мода = группа образцов с похожими характеристиками
                      - Мода слева (низкие PC1) → обычно ближе к норме, справа (высокие PC1) → обычно патология
                      - Но граница между норма/патология определяется порогами (0.2, 0.5, 0.8), а не модами
                      - Моды помогают выявить основные патологические состояния в данных
                    
                    - **Зеленые пунктирные линии**: Процентили (P1, P99)
                      - **P1** (слева) → будет соответствовать 0 на нижнем графике
                      - **P99** (справа) → будет соответствовать 1 на нижнем графике
                      - Образцы за пределами P1-P99 → выбросы (обрезаются до 0 или 1)
                    
                    ---
                    
                    **Нижний график: Спектральная шкала 0-1 (нормализованные значения)**
                    
                    - **Гистограмма**: Распределение образцов на шкале от 0 до 1
                      - 0 = норма (минимальная патология) = соответствует P1 на верхнем графике
                      - 1 = максимальная патология = соответствует P99 на верхнем графике
                      - ⚠️ **ВАЖНО**: Гистограмма группирует данные в 30 bins (интервалов)
                      - **Один столбик НЕ равен одному WSI** - в одном столбике может быть несколько WSI
                      - Если у вас 10 WSI, они распределены по этим 30 bins
                      - Высота столбика = **Frequency** (частота, количество образцов в bin)
                    
                    - **Красные пунктирные линии**: Позиции МОД на шкале 0-1
                      - 📌 **МОДА** = локальный максимум плотности (пик распределения)
                      - Те же моды, что на верхнем графике, но пересчитанные в шкалу 0-1
                      - Показывают, где находятся стабильные состояния (группы похожих образцов) на шкале
                      - ⚠️ Это НЕ центр нормы, а локальные максимумы концентрации образцов
                      - Можно интерпретировать как "уровни патологии", но границы определяются порогами (0.2, 0.5, 0.8)
                    
                    ---
                    
                    **Интерпретация для ваших данных:**
                    - Если моды группируются слева (ближе к 0) → много нормальных образцов
                    - Если моды справа (ближе к 1) → много патологических образцов
                    - Равномерное распределение → плавный переход от нормы к патологии
                    - Два четких пика → бимодальное распределение (норма vs патология)
                    
                    **Чтобы увидеть каждый WSI отдельно**, посмотрите таблицу "Результаты спектрального анализа" ниже
                    
                    ---
                    
                    ## 🔬 GMM (Gaussian Mixture Model) - что добавляет?
                    
                    **GMM** - это модель смеси гауссовых распределений, которая:
                    
                    ⚖️ **КЛЮЧЕВЫЕ ОТЛИЧИЯ KDE vs GMM (оба используют гауссовы, но по-разному):**
                    
                    **📊 KDE (Kernel Density Estimation) - непараметрический метод:**
                    - Размещает гауссово ядро на **КАЖДОЙ точке данных** (n ядер = n точек)
                    - Все ядра имеют **одинаковый вес** (1/n)
                    - **Одинаковый bandwidth** (h) для всех ядер (или адаптивный по правилу Скотта)
                    - **Не требует обучения** - просто суммирует все ядра
                    - Показывает **РЕАЛЬНУЮ форму** распределения (точная копия данных)
                    - Чувствителен к выбросам (каждая точка влияет на результат)
                    - Формула: `KDE(x) = (1/n) × Σ K((x - x_i) / h)`
                    
                    **📊 GMM (Gaussian Mixture Model) - параметрический метод:**
                    - Использует **ФИКСИРОВАННОЕ число компонентов** (обычно 2-5, намного меньше числа точек)
                    - Каждый компонент имеет **свой вес** (w_i), **центр** (μ_i), **ширину** (σ_i)
                    - **Требует обучения** через EM-алгоритм для нахождения оптимальных параметров
                    - Автоматически определяет **число компонентов** через BIC критерий
                    - Аппроксимирует данные **параметрической моделью** (упрощенное представление)
                    - Устойчив к выбросам (компоненты группируют похожие данные)
                    - Формула: `GMM(x) = Σ w_i × N(x | μ_i, σ_i²)`
                    
                    **💡 ПРАКТИЧЕСКОЕ РАЗЛИЧИЕ:**
                    - **KDE** = "сфотографировать" распределение (точная копия, все детали)
                    - **GMM** = "нарисовать упрощенную модель" (компактное представление, основные состояния)
                    - **KDE** лучше для визуализации и понимания реальной формы данных
                    - **GMM** лучше для классификации, интерпретации и дальнейшего анализа
                    
                    ---
                    
                    - **Автоматически определяет оптимальное число состояний** (компонентов) через BIC критерий
                      - Не нужно вручную задавать число кластеров
                      - Модель сама выбирает, сколько "состояний патологии" есть в данных
                    
                    - **Моделирует распределение PC1 как смесь нескольких гауссовых распределений**
                      - Каждый компонент = одно "состояние патологии" (например: норма, mild, moderate)
                      - Каждый компонент имеет свой центр (mean) и вес (weight)
                    
                    - **Визуализация на графике:**
                      - **Фиолетовая пунктирная линия**: Плотность распределения, предсказанная GMM
                      - **Фиолетовые крестики (X)**: Центры компонентов GMM (состояния патологии)
                      - Число рядом с крестиком = вес компонента (доля образцов в этом состоянии)
                    
                    - **Сравнение с KDE:**
                      - KDE (синяя линия) = сглаженная оценка плотности из данных
                      - GMM (фиолетовая линия) = параметрическая модель, которая пытается объяснить данные через смесь гауссовых
                      - Если GMM хорошо соответствует KDE → модель хорошо описывает данные
                      - Если GMM сильно отличается от KDE → возможно, нужно больше компонентов или данные не гауссовы
                    
                    - **Практическое применение:**
                      - Валидация мод, найденных через KDE
                      - Автоматическое определение числа патологических состояний
                      - Более точное моделирование распределения для прогнозирования
                    
                    **Когда использовать GMM:**
                    - Когда хотите автоматически определить число состояний патологии
                    - Когда нужна параметрическая модель для дальнейшего анализа
                    - Для валидации результатов KDE анализа
                    """)

                label_column = None
                if "label" in df_spectrum.columns:
                    label_column = "label"

                # Сохранение графика
                plot_path = Path("temp_spectrum_plot.png")
                analyzer.visualize_spectrum(
                    df_pca, label_column=label_column, save_path=plot_path
                )

                if plot_path.exists():
                    st.image(str(plot_path))
                    plot_path.unlink()  # Удаление временного файла
                
                # Expander с пояснениями к графику
                with st.expander("ℹ️ Пояснения к графику спектрального анализа", expanded=False):
                    gmm_method_text = ""
                    if analyzer.gmm is not None:
                        gmm_method_text = (
                            f"GMM обучен с {analyzer.gmm.n_components} компонентами через EM-алгоритм. "
                            f"Число компонентов выбрано автоматически по BIC критерию. "
                        )
                    
                    st.markdown(f"""
                    **📌 НАЗВАНИЕ МЕТОДА СПЕКТРАЛЬНОГО АНАЛИЗА:**
                    'Non-parametric Density Estimation with Peak Detection'
                    (Непараметрическая оценка плотности с обнаружением пиков)
                    
                    ---
                    
                    **[KDE] (синяя линия):** KDE = Kernel Density Estimation (Оценка плотности ядерным методом)
                    
                    📌 **ОПРЕДЕЛЕНИЕ KDE:** Непараметрический метод оценки плотности распределения данных.
                    
                    📐 **ФОРМУЛА:** `KDE(x) = (1/n) × Σ K((x - x_i) / h)`
                    - где: n - число точек, K - ядро (гауссова функция), h - bandwidth (ширина окна), x_i - данные
                    
                    🔧 **КАК ВЫЧИСЛЯЕТСЯ:**
                    1. На каждую точку данных (PC1) размещается гауссово ядро
                    2. Все ядра суммируются → получается сглаженная кривая плотности
                    3. Используется правило Скотта для выбора bandwidth (h)
                    4. Результат: непрерывная оценка плотности распределения
                    
                    Показывает реальную форму распределения без предположений о модели. Пики = области с высокой концентрацией образцов.
                    
                    ---
                    
                    **[Mode] (красные пунктирные линии):** МОДЫ - стабильные состояния (локальные максимумы плотности)
                    
                    📌 **ОПРЕДЕЛЕНИЕ МОДЫ:** Мода = локальный максимум плотности распределения PC1.
                    
                    🔍 **МЕТОД ПОИСКА:** Peak Detection Algorithm (Алгоритм обнаружения пиков)
                    
                    📐 **АЛГОРИТМ:**
                    1. KDE (Kernel Density Estimation) - оценка плотности через гауссовы ядра
                    2. find_peaks (scipy.signal) - поиск локальных максимумов в оценке плотности
                       Параметры: height ≥ 5% от максимума, distance ≥ 1/20 длины сетки
                    3. Мода = позиция пика на шкале PC1, где концентрация образцов максимальна
                    
                    ⚠️ **ВАЖНО:** Это НЕ центр нормы, а локальные максимумы распределения (группы похожих образцов).
                    - Каждая мода = группа образцов с похожими характеристиками
                    - Мода слева (низкие PC1) обычно ближе к норме, справа (высокие PC1) - к патологии
                    - Но граница норма/патология определяется порогами (0.2, 0.5, 0.8), а не модами
                    - Моды помогают выявить основные патологические состояния в данных
                    
                    ---
                    
                    **[Границы интервалов] (зеленые/оранжевые/красные сплошные линии):** границы классификации на спектральной шкале
                    - 0.0-0.2 = normal (норма)
                    - 0.2-0.5 = mild (легкая патология)
                    - 0.5-0.8 = moderate (умеренная патология)
                    - 0.8-1.0 = severe (тяжелая патология)
                    
                    ---
                    
                    **[GMM] (фиолетовая линия):** GMM = Gaussian Mixture Model (Модель смеси гауссовых распределений)
                    
                    📌 **ОПРЕДЕЛЕНИЕ GMM:** Параметрическая модель, аппроксимирующая распределение как смесь нескольких гауссовых.
                    
                    📐 **ФОРМУЛА:** `GMM(x) = Σ w_i × N(x | μ_i, σ_i²)`
                    - где: w_i - вес компонента, μ_i - центр, σ_i - ширина, N - гауссова плотность
                    
                    🔧 **КАК ВЫЧИСЛЯЕТСЯ:**
                    1. EM-алгоритм находит оптимальное число компонентов (через BIC)
                    2. Для каждого компонента: μ (центр), σ (ширина), w (вес)
                    3. Суммируются несколько гауссовых с разными параметрами
                    
                    {gmm_method_text}
                    
                    Аппроксимирует распределение через несколько компонентов (состояний). Центры компонентов отмечены вертикальными линиями. RMSE показывает качество аппроксимации KDE.
                    
                    ---
                    
                    **⚖️ КЛЮЧЕВЫЕ ОТЛИЧИЯ KDE vs GMM (оба используют гауссовы, но по-разному):**
                    
                    **📊 KDE (непараметрический):**
                    - Размещает ядро на КАЖДОЙ точке данных (n ядер = n точек)
                    - Все ядра имеют одинаковый вес (1/n)
                    - Одинаковый bandwidth (h) для всех ядер (или адаптивный)
                    - Не требует обучения - просто суммирует ядра
                    - Показывает РЕАЛЬНУЮ форму распределения
                    - Чувствителен к выбросам (каждая точка влияет)
                    
                    **📊 GMM (параметрический):**
                    - Использует ФИКСИРОВАННОЕ число компонентов (обычно 2-5, меньше числа точек)
                    - Каждый компонент имеет свой вес (w_i), центр (μ_i), ширину (σ_i)
                    - Требует обучения через EM-алгоритм для нахождения параметров
                    - Автоматически определяет число компонентов через BIC
                    - Аппроксимирует данные параметрической моделью
                    - Устойчив к выбросам (компоненты группируют данные)
                    
                    **💡 ПРАКТИЧЕСКОЕ РАЗЛИЧИЕ:**
                    - KDE = 'сфотографировать' распределение (точная копия)
                    - GMM = 'нарисовать упрощенную модель' (компактное представление)
                    - KDE лучше для визуализации, GMM лучше для классификации и интерпретации
                    """)

                # Таблица с результатами
                st.subheader("Результаты спектрального анализа")
                st.markdown(
                    "**Эта таблица показывает каждый WSI отдельно** - здесь вы можете увидеть точное значение "
                    "спектральной шкалы для каждого образца."
                )
                display_cols = ["image", "PC1", "PC1_spectrum"]
                if "PC1_mode" in df_spectrum.columns:
                    display_cols.append("PC1_mode")

                st.dataframe(
                    df_spectrum[display_cols].sort_values(
                        by="PC1_spectrum", ascending=False
                    ),
                    use_container_width=True,
                )
                
                # Дополнительная визуализация: гистограмма с точками для каждого WSI
                st.subheader("📊 Распределение WSI на спектральной шкале (с точками)")
                st.markdown(
                    "**Этот график показывает гистограмму (как на нижнем графике выше) с наложенными точками для каждого WSI.** "
                    "Вы можете увидеть, где именно расположен каждый из ваших образцов."
                )
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Получаем значения спектра (те же, что используются в гистограмме)
                spectrum_values = df_spectrum["PC1_spectrum"].dropna().values
                image_names = df_spectrum.loc[df_spectrum["PC1_spectrum"].notna(), "image"].values
                
                # Строим гистограмму (как в нижнем графике visualize_spectrum)
                counts, bins, patches = ax.hist(
                    spectrum_values, 
                    bins=30, 
                    alpha=0.6, 
                    color='lightblue',
                    edgecolor='black',
                    linewidth=0.5,
                    label='Гистограмма (частота)'
                )
                
                # Добавляем точки для каждого WSI поверх гистограммы
                # Размещаем точки на высоте, соответствующей частоте в этом bin + небольшой отступ
                np.random.seed(42)  # Фиксированный seed для воспроизводимости
                
                # Для каждой точки находим соответствующий bin и размещаем на его высоте
                point_heights = []
                for val in spectrum_values:
                    # Находим индекс bin для этого значения
                    bin_idx = np.digitize(val, bins) - 1
                    bin_idx = np.clip(bin_idx, 0, len(counts) - 1)
                    # Высота = частота в этом bin + небольшой случайный отступ
                    height = counts[bin_idx] + np.random.uniform(0.1, 0.3)
                    point_heights.append(height)
                
                point_heights = np.array(point_heights)
                
                # Цвета в зависимости от значения (зеленый = норма, красный = патология)
                colors = plt.cm.RdYlGn_r(spectrum_values)  # Красный-желтый-зеленый (инвертированный)
                
                # Рисуем точки поверх гистограммы
                ax.scatter(spectrum_values, point_heights, alpha=0.8, s=120, c=colors, 
                          edgecolors='black', linewidth=1.5, zorder=5, label='WSI образцы')
                
                # Подписи для каждого образца (первые 20 символов имени)
                for i, (x, y, name) in enumerate(zip(spectrum_values, point_heights, image_names)):
                    short_name = name[:20] + "..." if len(name) > 20 else name
                    ax.annotate(short_name, (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.7,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
                
                # Отметка мод (как в оригинальном графике)
                if analyzer.modes:
                    for mode in analyzer.modes:
                        mode_spectrum = (mode["position"] - analyzer.pc1_p1) / (
                            analyzer.pc1_p99 - analyzer.pc1_p1
                        )
                        mode_spectrum = np.clip(mode_spectrum, 0.0, 1.0)
                        ax.axvline(
                            mode_spectrum,
                            color="r",
                            linestyle="--",
                            linewidth=2,
                            alpha=0.7,
                            label="Мода" if mode == analyzer.modes[0] else ""
                        )
                
                ax.set_xlabel("Спектральная шкала (0-1)", fontsize=12)
                ax.set_ylabel("Частота (количество образцов в bin)", fontsize=12)
                ax.set_title(
                    f"Распределение WSI на спектральной шкале (всего {len(spectrum_values)} образцов)\n"
                    "Гистограмма показывает частоту, точки - расположение каждого WSI",
                    fontsize=13
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(bottom=0)
                ax.grid(True, alpha=0.3, axis="both")
                ax.legend(loc='upper right')
                plt.tight_layout()
                st.pyplot(fig)

                # Важность признаков
                st.subheader("📊 Важность признаков (PC1 loadings)")
                st.markdown(
                    "**Loadings PC1** показывают вклад каждого признака в первую главную компоненту. "
                    "Чем больше абсолютное значение, тем важнее признак для разделения образцов."
                )
                
                # Подробное объяснение вычисления важности
                with st.expander("🔬 Как вычисляется важность признака?"):
                    st.markdown("""
                    ## Математический процесс вычисления важности признаков
                    
                    ### Шаг 1: Стандартизация данных
                    Перед применением PCA все признаки стандартизируются (нормализуются):
                    ```
                    X_scaled = (X - μ) / σ
                    ```
                    Где:
                    - `X` - исходные значения признаков
                    - `μ` - среднее значение признака
                    - `σ` - стандартное отклонение признака
                    
                    **Зачем это нужно:** Признаки имеют разные масштабы (например, count может быть 0-100, а area - 0-10000). 
                    Стандартизация приводит все признаки к одному масштабу, чтобы ни один признак не доминировал из-за больших числовых значений.
                    
                    ---
                    
                    ### Шаг 2: Применение PCA
                    PCA (Principal Component Analysis) находит главные компоненты - направления максимальной вариации в данных.
                    
                    **Первая главная компонента (PC1)** - это направление, вдоль которого данные варьируются больше всего.
                    Она максимизирует дисперсию и лучше всего разделяет образцы по степени патологии.
                    
                    #### 📐 Матрица ковариации и её роль в PCA
                    
                    **Математическая основа PCA:**
                    
                    PCA можно вычислить двумя эквивалентными способами:
                    
                    **Способ 1: Через матрицу ковариации (классический подход)**
                    
                    1. **Вычисление матрицы ковариации:**
                       ```
                       Cov = (1/(n-1)) × X_scaled^T × X_scaled
                       ```
                       Где:
                       - `X_scaled` - матрица стандартизированных данных (размер: n образцов × p признаков)
                       - `n` - число образцов
                       - `p` - число признаков
                       - `Cov` - матрица ковариации (размер: p × p)
                    
                    2. **Элементы матрицы ковариации:**
                       ```
                       Cov[i, j] = (1/(n-1)) × Σ (x_i - μ_i) × (x_j - μ_j)
                       ```
                       - `Cov[i, i]` - дисперсия i-го признака (диагональные элементы)
                       - `Cov[i, j]` - ковариация между признаками i и j (недиагональные элементы)
                       - Ковариация показывает, как два признака изменяются вместе
                    
                    3. **Собственные векторы и собственные значения:**
                       ```
                       Cov × v = λ × v
                       ```
                       Где:
                       - `v` - собственный вектор (eigenvector) = направление главной компоненты
                       - `λ` - собственное значение (eigenvalue) = дисперсия вдоль этого направления
                       - Собственные векторы упорядочены по убыванию собственных значений
                       - Первый собственный вектор (с наибольшим λ) = PC1
                    
                    **Способ 2: Через SVD (Singular Value Decomposition) - используется в sklearn**
                    
                    Sklearn использует более эффективный численный метод - SVD:
                    ```
                    X_scaled = U × Σ × V^T
                    ```
                    Где:
                    - `V^T` - транспонированная матрица правых сингулярных векторов = loadings (components_)
                    - `Σ` - диагональная матрица сингулярных значений (связана с собственными значениями)
                    - `U` - матрица левых сингулярных векторов
                    
                    **Связь между методами:**
                    - Собственные векторы матрицы ковариации = сингулярные векторы V из SVD
                    - Собственные значения = квадраты сингулярных значений, деленные на (n-1)
                    - Оба метода дают одинаковые результаты, но SVD численно более устойчив
                    
                    **Почему матрица ковариации важна?**
                    
                    1. **Диагональные элементы (дисперсии):**
                       - Показывают, насколько каждый признак варьируется
                       - Признаки с большой дисперсией потенциально важнее
                    
                    2. **Недиагональные элементы (ковариации):**
                       - Показывают корреляции между признаками
                       - Если два признака сильно коррелируют, PCA объединяет их в одну компоненту
                       - Это позволяет уменьшить размерность без потери информации
                    
                    3. **Собственные векторы:**
                       - Направления максимальной вариации в данных
                       - PC1 = направление наибольшей вариации
                       - PC2 = направление второй по величине вариации (ортогонально к PC1)
                    
                    **Пример вычисления матрицы ковариации:**
                    
                    Предположим, у нас есть 3 образца и 2 признака:
                    ```
                    X_scaled = [[1.0, 0.5],
                                [0.0, -0.5],
                                [-1.0, 0.0]]
                    ```
                    
                    Матрица ковариации:
                    ```
                    Cov = (1/(3-1)) × X_scaled^T × X_scaled
                        = 0.5 × [[1.0, 0.0, -1.0],    [[1.0, 0.5],
                                 [0.5, -0.5, 0.0]]  ×   [0.0, -0.5],
                                                         [-1.0, 0.0]]
                        = [[1.0, 0.25],
                           [0.25, 0.25]]
                    ```
                    
                    Диагональные элементы: дисперсии признаков (1.0 и 0.25)
                    Недиагональные элементы: ковариация между признаками (0.25)
                    
                    **Где это происходит в коде?**
                    
                    В нашем коде (`scale/pca_scoring.py` и `scale/spectral_analysis.py`):
                    ```python
                    # Стандартизация данных
                    X_scaled = scaler.fit_transform(X)
                    
                    # PCA обучение (внутри sklearn использует SVD)
                    pca = PCA(n_components=None)
                    pca.fit(X_scaled)  # ← Здесь вычисляется матрица ковариации (через SVD)
                    
                    # Loadings доступны через:
                    loadings = pca.components_[0]  # Первая главная компонента
                    ```
                    
                    Sklearn автоматически:
                    1. Вычисляет матрицу ковариации (или эквивалент через SVD)
                    2. Находит собственные векторы и собственные значения
                    3. Сохраняет их в `pca.components_` (loadings) и `pca.explained_variance_` (собственные значения)
                    
                    **Доступ к матрице ковариации:**
                    
                    Если нужно явно получить матрицу ковариации из обученной PCA модели:
                    ```python
                    # Матрица ковариации (если нужна явно)
                    covariance_matrix = pca.get_covariance()  # Доступна в sklearn PCA
                    
                    # Или можно вычислить вручную:
                    import numpy as np
                    covariance_matrix = np.cov(X_scaled.T)  # Транспонируем для правильной размерности
                    ```
                    
                    ---
                    
                    ### Шаг 3: Извлечение loadings
                    **Loadings (коэффициенты загрузки)** - это веса, которые показывают, как каждый признак вносит вклад в PC1.
                    
                    Loadings берутся из первой строки матрицы `components_` обученной PCA модели:
                    ```python
                    loadings = pca.components_[0]  # Первая строка = первая главная компонента
                    ```
                    
                    **Математически:** PC1 вычисляется как линейная комбинация стандартизированных признаков:
                    ```
                    PC1 = loading₁ × признак₁ + loading₂ × признак₂ + ... + loadingₙ × признакₙ
                    ```
                    
                    ---
                    
                    ### 🔄 Вычисление PC1 для конкретного WSI
                    
                    После обучения PCA модели, для каждого нового WSI (включая те, на которых обучалась модель) вычисляется PC1 следующим образом:
                    
                    **Шаг 1: Извлечение признаков WSI**
                    ```
                    X_wsi = [признак₁, признак₂, ..., признакₙ]
                    ```
                    Например, для WSI "image_001.tif":
                    ```
                    X_wsi = [Mild_relative_count=0.5, Dysplasia_relative_area=1.2, ..., Paneth_mean_relative_area=0.3]
                    ```
                    
                    **Шаг 2: Стандартизация признаков WSI**
                    Используются те же параметры стандартизации (μ и σ), что были вычислены при обучении:
                    ```
                    X_wsi_scaled[i] = (X_wsi[i] - μᵢ) / σᵢ
                    ```
                    Где:
                    - `μᵢ` - среднее значение i-го признака из обучающей выборки
                    - `σᵢ` - стандартное отклонение i-го признака из обучающей выборки
                    
                    **Важно:** Используются параметры из обучения, а не пересчитываются заново!
                    
                    **Шаг 3: Вычисление PC1**
                    ```
                    PC1(wsi) = loading₁ × X_wsi_scaled[1] + loading₂ × X_wsi_scaled[2] + ... + loadingₙ × X_wsi_scaled[n]
                    ```
                    
                    Или в матричной форме:
                    ```
                    PC1(wsi) = loadings^T × X_wsi_scaled
                    ```
                    Где `loadings` - вектор loadings первой главной компоненты (из `pca.components_[0]`)
                    
                    **Шаг 4: Нормализация PC1 (опционально)**
                    Для получения шкалы от 0 до 1:
                    ```
                    PC1_norm(wsi) = (PC1(wsi) - PC1_min) / (PC1_max - PC1_min)
                    ```
                    Где `PC1_min` и `PC1_max` - минимальное и максимальное значения PC1 из обучающей выборки
                    
                    ---
                    
                    **Пример вычисления PC1 для конкретного WSI:**
                    
                    Предположим, у нас есть WSI со следующими признаками:
                    ```
                    Mild_relative_count = 0.8
                    Dysplasia_relative_area = 1.5
                    Crypts_count = 100
                    ```
                    
                    После стандартизации (используя μ и σ из обучения):
                    ```
                    Mild_relative_count_scaled = (0.8 - 0.5) / 0.3 = 1.0
                    Dysplasia_relative_area_scaled = (1.5 - 1.0) / 0.5 = 1.0
                    Crypts_count_scaled = (100 - 120) / 20 = -1.0
                    ```
                    
                    Если loadings:
                    ```
                    Mild_relative_count: loading = +0.25
                    Dysplasia_relative_area: loading = +0.30
                    Crypts_count: loading = -0.10
                    ```
                    
                    Тогда PC1 вычисляется как:
                    ```
                    PC1(wsi) = (0.25 × 1.0) + (0.30 × 1.0) + (-0.10 × -1.0)
                             = 0.25 + 0.30 + 0.10
                             = 0.65
                    ```
                    
                    Если PC1_min = -2.0 и PC1_max = 3.0 из обучающей выборки:
                    ```
                    PC1_norm(wsi) = (0.65 - (-2.0)) / (3.0 - (-2.0))
                                 = 2.65 / 5.0
                                 = 0.53
                    ```
                    
                    **Интерпретация:** WSI имеет PC1_norm = 0.53, что означает средний уровень патологии (ближе к середине шкалы).
                    
                    ---
                    
                    **Где это происходит в коде:**
                    
                    В `scale/pca_scoring.py`, метод `transform()`:
                    ```python
                    # 1. Извлечение признаков
                    X = df[feature_columns].fillna(0).values
                    
                    # 2. Стандартизация (используя параметры из обучения)
                    X_scaled = self.scaler.transform(X)  # ← использует self.scaler.mean_ и self.scaler.scale_
                    
                    # 3. Вычисление PC1 (используя loadings из обучения)
                    X_pca = self.pca.transform(X_scaled)  # ← использует self.pca.components_
                    PC1 = X_pca[:, 0]  # Первая колонка = PC1 для каждого образца
                    
                    # 4. Нормализация
                    PC1_norm = (PC1 - self.pc1_min) / (self.pc1_max - self.pc1_min)
                    ```
                    
                    **Ключевой момент:** Все параметры (μ, σ, loadings, PC1_min, PC1_max) фиксируются при обучении (`fit()`) и используются для всех последующих WSI (`transform()`).
                    
                    ---
                    
                    ### Шаг 4: Интерпретация важности
                    
                    **Абсолютное значение loading** показывает важность признака:
                    - **Большое абсолютное значение** (например, |0.27|) → признак сильно влияет на PC1
                    - **Малое абсолютное значение** (например, |0.02|) → признак слабо влияет на PC1
                    
                    **Знак loading** показывает направление влияния:
                    - **Положительный loading** (+0.27) → увеличение признака увеличивает PC1 → выше патология
                    - **Отрицательный loading** (-0.15) → увеличение признака уменьшает PC1 → ниже патология (ближе к норме)
                    
                    ---
                    
                    ### Пример вычисления
                    
                    Предположим, у нас есть 3 признака с loadings:
                    - `Mild_relative_count`: loading = +0.25
                    - `Dysplasia_relative_area`: loading = +0.30
                    - `Crypts_count`: loading = -0.10
                    
                    Для WSI со стандартизированными значениями:
                    - `Mild_relative_count` = 1.5
                    - `Dysplasia_relative_area` = 2.0
                    - `Crypts_count` = -0.5
                    
                    PC1 вычисляется как:
                    ```
                    PC1 = (0.25 × 1.5) + (0.30 × 2.0) + (-0.10 × -0.5)
                        = 0.375 + 0.60 + 0.05
                        = 1.025
                    ```
                    
                    Видно, что `Dysplasia_relative_area` дает наибольший вклад (0.60), так как у него:
                    - Большой положительный loading (+0.30)
                    - Высокое значение признака (2.0)
                    
                    ---
                    
                    ### Почему это работает?
                    
                    PCA автоматически находит оптимальные веса (loadings), которые:
                    1. **Максимизируют дисперсию** - PC1 объясняет максимальную вариацию в данных
                    2. **Лучше всего разделяют образцы** - образцы с разной патологией максимально различаются по PC1
                    3. **Учитывают корреляции** - если признаки коррелируют, PCA это учитывает
                    
                    Поэтому loadings первой компоненты - это объективная мера важности признаков для разделения норма/патология.
                    """)
                
                feature_importance = analyzer.get_feature_importance()
                
                # Показываем информацию о признаках, используемых в PCA
                if analyzer.feature_columns is not None:
                    structural_features_in_pca = [f for f in analyzer.feature_columns if any(x in f.lower() for x in ['surface epithelium', 'muscularis mucosae'])]
                    if structural_features_in_pca:
                        st.info(f"📊 **Используется {len(analyzer.feature_columns)} признаков в PCA**, включая структурные: {', '.join(structural_features_in_pca)}")
                    else:
                        st.info(f"📊 **Используется {len(analyzer.feature_columns)} признаков в PCA**")

                # Таблица с важностью признаков (показываем все)
                # Создание DataFrame для таблицы
                importance_df = pd.DataFrame({
                    "Признак": feature_importance.index,
                    "Loading (важность)": feature_importance.values,
                    "Абсолютное значение": feature_importance.abs().values
                }).sort_values("Абсолютное значение", ascending=False)
                
                # Проверяем, есть ли структурные признаки в таблице
                structural_in_table = [f for f in importance_df["Признак"].values if any(x in f.lower() for x in ['surface epithelium', 'muscularis mucosae'])]
                if structural_in_table:
                    st.info(f"💡 **Структурные признаки в PCA:** {', '.join(structural_in_table)}. Их loadings могут быть малыми, что означает небольшой вклад в PC1, но они все равно учитываются при вычислении.")
                
                st.dataframe(importance_df, use_container_width=True, hide_index=True)

                # График важности признаков (показываем все)
                # Определяем размер графика в зависимости от количества признаков
                n_features = len(feature_importance)
                fig_height = max(6, n_features * 0.4)  # Минимум 6, плюс 0.4 на каждый признак
                fig, ax = plt.subplots(figsize=(10, fig_height))
                
                # Сортируем по абсолютному значению для графика
                features_sorted = feature_importance.sort_values(key=abs, ascending=True)

                colors = ['red' if x < 0 else 'blue' for x in features_sorted.values]
                ax.barh(
                    range(len(features_sorted)),
                    features_sorted.values,
                    align="center",
                    color=colors,
                    alpha=0.7
                )
                ax.set_yticks(range(len(features_sorted)))
                ax.set_yticklabels(features_sorted.index)
                ax.set_xlabel("Loading value")
                ax.set_title(f"Важность всех признаков в PC1 ({n_features} признаков)")
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)
                
                # Таблица и график GMM компонентов (если GMM обучен)
                if analyzer.gmm is not None:
                    st.subheader("🔬 GMM компоненты - Характеристика медицинских состояний")
                    
                    # Таблица 1: Параметры на сырой шкале PC1
                    st.markdown("**📋 Таблица 1: Параметры GMM компонентов (сырая шкала PC1)**")
                    try:
                        gmm_params_df = analyzer.get_gmm_components_table()
                        # Форматируем для отображения
                        gmm_params_df_display = gmm_params_df.copy()
                        gmm_params_df_display["Центр (μ) на PC1"] = gmm_params_df_display["Центр (μ) на PC1"].apply(lambda x: f"{x:.4f}")
                        gmm_params_df_display["Центр на шкале 0-1"] = gmm_params_df_display["Центр на шкале 0-1"].apply(lambda x: f"{x:.4f}")
                        gmm_params_df_display["Ширина (σ)"] = gmm_params_df_display["Ширина (σ)"].apply(lambda x: f"{x:.4f}")
                        gmm_params_df_display["Вес (w)"] = gmm_params_df_display["Вес (w)"].apply(lambda x: f"{x:.4f}")
                        gmm_params_df_display["Доля образцов (%)"] = gmm_params_df_display["Доля образцов (%)"].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(gmm_params_df_display, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Ошибка при создании таблицы GMM: {e}")
                    
                    # Таблица 2: Параметры на нормализованной шкале 0-1
                    st.markdown("**📋 Таблица 2: Параметры GMM компонентов (нормализованная шкала 0-1)**")
                    try:
                        gmm_params_norm_df = analyzer.get_gmm_components_table_normalized()
                        gmm_params_norm_display = gmm_params_norm_df.copy()
                        gmm_params_norm_display["Центр (μ) на шкале 0-1"] = gmm_params_norm_display["Центр (μ) на шкале 0-1"].apply(lambda x: f"{x:.4f}")
                        gmm_params_norm_display["Ширина (σ) на шкале 0-1"] = gmm_params_norm_display["Ширина (σ) на шкале 0-1"].apply(lambda x: f"{x:.4f}")
                        gmm_params_norm_display["Вес (w)"] = gmm_params_norm_display["Вес (w)"].apply(lambda x: f"{x:.4f}")
                        gmm_params_norm_display["Доля образцов (%)"] = gmm_params_norm_display["Доля образцов (%)"].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(gmm_params_norm_display, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Ошибка при создании нормализованной таблицы GMM: {e}")
                    
                    # Комментарии о смысле гауссианов
                    with st.expander("ℹ️ Что означают гауссианы и их параметры?"):
                        st.markdown("""
                        ## 📊 Смысл гауссианов в GMM
                        
                        **Каждый гауссиан = одно чистое медицинское состояние:**
                        
                        - **Гауссиан** - это математическая модель распределения образцов в определенном состоянии патологии
                        - GMM (Gaussian Mixture Model) находит несколько таких состояний в ваших данных
                        - Каждое состояние описывается гауссовым распределением с параметрами:
                        
                        ### Параметры гауссиана:
                        
                        1. **Центр (μ)** - типичное значение PC1 для этого состояния
                           - Показывает, где находится "пик" состояния на шкале патологии
                           - Образцы с PC1 близким к μ наиболее типичны для этого состояния
                        
                        2. **Ширина (σ)** - разброс образцов в этом состоянии
                           - Маленький σ → узкое состояние, образцы очень похожи
                           - Большой σ → широкое состояние, большой разброс характеристик
                        
                        3. **Вес (w)** - доля образцов, принадлежащих этому состоянию
                           - Показывает, какая часть ваших данных относится к этому состоянию
                           - Сумма всех весов = 1.0 (100%)
                        
                        ### Интерпретация для медицины:
                        
                        - **Normal (норма)**: Гауссиан с низким μ (близко к 0 на спектральной шкале)
                          - Образцы с минимальными патологическими признаками
                        
                        - **Mild (легкая патология)**: Гауссиан в диапазоне 0.2-0.5 на спектральной шкале
                          - Начальные признаки патологии
                        
                        - **Moderate (умеренная патология)**: Гауссиан в диапазоне 0.5-0.8
                          - Выраженные патологические изменения
                        
                        - **Severe (тяжелая патология)**: Гауссиан близко к 1.0 на спектральной шкале
                          - Максимальные патологические изменения
                        
                        ### Почему гауссианы, а не другие формы?
                        
                        - Гауссово распределение естественно возникает в биологических данных
                        - Центральная предельная теорема: сумма многих факторов → нормальное распределение
                        - GMM автоматически находит оптимальное число состояний через BIC критерий
                        - Параметрическая модель: легко интерпретировать и использовать для классификации
                        
                        ### Преобразование на нормализованную шкалу:
                        
                        - **Не искажает форму**: Линейное преобразование сохраняет форму гауссиана
                        - **Масштабирует параметры**: μ и σ преобразуются пропорционально
                        - **Универсальная шкала**: Позволяет сравнивать разные датасеты
                        """)
                    
                    # Графики: показываем 4 графика по умолчанию
                    st.subheader("📊 Графики GMM компонентов: сравнение шкал")
                    st.markdown("**4 графика показывают спектр и GMM компоненты на сырой и нормализованной шкалах**")
                    
                    try:
                        # Всегда показываем сравнение (4 графика)
                        fig_comparison = analyzer.visualize_spectrum_comparison(
                            df_pca, pc1_column="PC1", save_path=None, return_figure=True
                        )
                        st.pyplot(fig_comparison)
                        plt.close(fig_comparison)
                    except Exception as e:
                        st.error(f"Ошибка при построении графиков сравнения: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    # Рекомендации по улучшению аппроксимации
                    with st.expander("💡 Как улучшить аппроксимацию GMM?"):
                        st.markdown("""
                        ## 🔧 Варианты улучшения аппроксимации
                        
                        Если 2 гауссиана плохо аппроксимируют исходную плотность (высокий RMSE), попробуйте:
                        
                        ### 1. **Увеличить число компонентов GMM**
                        - Отключите "Автоматический выбор" и задайте число компонентов вручную
                        - Используйте графики качества выше, чтобы найти оптимальное число
                        - **Осторожно**: слишком много компонентов → переобучение
                        
                        ### 2. **Проверить полноту набора признаков** ⭐ ВАЖНО!
                        - **Убедитесь, что все патологические признаки имеют полный набор** (count, area, mean_relative_area)
                        - Пропуск признаков может исказить структуру данных и привести к неоптимальному числу компонентов GMM
                        - **Пример**: После добавления недостающих признаков (EoE_relative_area, EoE_mean_relative_area, Granulomas_mean_relative_area) 
                          GMM стал находить 3 компонента вместо 2, что улучшило аппроксимацию
                        - Перейдите в раздел "🎯 Выбор признаков" в боковой панели
                        - Исключайте признаки только если они действительно неинформативны или избыточны
                        - Исключите высоко коррелированные признаки (см. раздел "🔍 Анализ образцов")
                        
                        ### 3. **Использовать альтернативные методы**
                        - **KDE** (уже используется для сравнения) - непараметрический метод, точнее описывает форму
                        - **Другие смеси распределений**: Student's t, Skew-normal (если распределение асимметрично)
                        - **Непараметрические методы**: Histogram, Kernel Density с разными bandwidth
                        
                        ### 4. **Преобразование данных**
                        - Попробуйте логарифмическое преобразование PC1 (если распределение скошено)
                        - Box-Cox преобразование для нормализации
                        
                        ### 5. **Анализ причин плохой аппроксимации**
                        - Проверьте распределение PC1 на нормальность (Q-Q plot)
                        - Проверьте на выбросы (они могут искажать GMM)
                        - Проверьте на мультимодальность (может потребоваться больше компонентов)
                        
                        ### 📊 Интерпретация метрик качества:
                        
                        - **RMSE**: Среднеквадратичная ошибка между KDE и GMM. Меньше = лучше.
                          - Показывает, насколько точно GMM аппроксимирует реальное распределение (KDE)
                          - Обычно уменьшается с увеличением числа компонентов (но не всегда!)
                          - **Если RMSE лучше с 3 компонентами, чем с 2** → это означает, что 3 компонента лучше описывают структуру данных
                        
                        - **R²**: Коэффициент детерминации. Ближе к 1 = лучше.
                          - Показывает долю вариации, объясненную моделью
                          - R² = 1.0 означает идеальную аппроксимацию
                        
                        - **BIC (Bayesian Information Criterion)**: Балансирует точность и сложность модели. Меньше = лучше.
                          - **Формула**: `BIC = -2 × log_likelihood + k × log(n)`, где k = число параметров, n = число образцов
                          - Штрафует за сложность модели (больше компонентов = больше параметров)
                          - Может выбрать меньше компонентов, даже если RMSE лучше с большим числом
                          - **Если BIC выбрал 2, а RMSE лучше с 3** → это компромисс: BIC предпочитает простоту
                          - **Ссылка**: Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.
                        
                        - **Max Error**: Максимальная локальная ошибка. Показывает худший случай аппроксимации.
                        
                        ### 🎯 Что выбрать: RMSE или BIC?
                        
                        - **Для точной аппроксимации**: используйте число компонентов с минимальным RMSE
                        - **Для простой модели**: используйте число компонентов с минимальным BIC
                        - **Для практических целей (классификация, шкала)**: обычно лучше использовать число компонентов с лучшим RMSE, 
                          так как это означает более точное описание реальной структуры данных
                        
                        ### 📚 Методы аппроксимации:
                        
                        - **GMM (Gaussian Mixture Model)**: Параметрическая модель, аппроксимирующая данные как смесь гауссовых распределений
                          - Обучение через **EM-алгоритм** (Expectation-Maximization)
                          - **Ссылки**: 
                            - Scikit-learn: https://scikit-learn.org/stable/modules/mixture.html
                            - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 9
                        
                        - **BIC (Bayesian Information Criterion)**: Критерий выбора числа компонентов
                          - Балансирует точность (log-likelihood) и сложность (число параметров)
                          - **Ссылки**: 
                            - Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464
                            - Подробное объяснение: см. `GMM_BIC_EXPLANATION.md`
                        
                        ### ⚠️ Важно:
                        
                        - GMM - это **параметрическая модель** (гауссовы распределения)
                        - Если данные не гауссовы, GMM может плохо аппроксимировать даже с большим числом компонентов
                        - KDE - **непараметрический метод**, точнее описывает реальную форму распределения
                        - Для практических целей (классификация, шкала) GMM может быть достаточным даже при неидеальной аппроксимации
                        """)
                    
                    # Пояснение к графику и связи с классификацией
                    with st.expander("ℹ️ Как интерпретировать график GMM компонентов и связь с классификацией?"):
                        st.markdown("""
                        ## 🔗 Связь между GMM компонентами и классификацией образцов
                        
                        **Важно понимать разницу:**
                        
                        1. **GMM компоненты** = реальные кластеры в данных
                           - GMM автоматически находит оптимальное число компонентов (состояний) через BIC критерий
                           - Например, если GMM нашел 2 компонента (mild и severe), это означает, что в данных есть только 2 доминирующих пика распределения
                           - Каждый компонент = группа образцов с похожими характеристиками
                        
                        2. **Классификация образцов (PC1_mode)** = искусственное разделение на 4 категории
                           - По умолчанию все образцы делятся на 4 категории (normal, mild, moderate, severe) на основе их позиции на спектральной шкале 0-1
                           - Это делается независимо от того, сколько компонентов нашел GMM
                           - Пороги: 0.0-0.2 = normal, 0.2-0.5 = mild, 0.5-0.8 = moderate, 0.8-1.0 = severe
                        
                        **Два подхода к классификации:**
                        
                        **A. Фиксированные пороги (по умолчанию):**
                        - Все образцы классифицируются на основе их позиции на спектральной шкале
                        - Если GMM нашел 2 компонента, но образцы распределены по всему диапазону, они все равно будут разделены на 4 категории
                        - Подходит, когда нужно единообразное разделение независимо от структуры данных
                        
                        **B. Классификация по GMM компонентам (опция):**
                        - Образцы классифицируются по принадлежности к GMM компонентам
                        - Если GMM нашел 2 компонента, образцы будут разделены только на 2 категории (соответствующие этим компонентам)
                        - Каждый компонент получает метку (normal/mild/moderate/severe) на основе позиции его центра на спектральной шкале
                        - Подходит, когда нужно использовать реальную структуру данных для классификации
                        
                        ---
                        
                        ## 📊 Параметры GMM компонентов
                        
                        **⚠️ ВАЖНО: Гауссианы соответствуют НЕ нормализованной шкале (сырые значения PC1)**
                        
                        - GMM обучается на **сырых значениях PC1** (например, от -3 до +9)
                        - Все параметры гауссианов (μ, σ) также в **сырой шкале PC1**
                        - На графике:
                          - **Нижняя ось X**: PC1 (сырые значения) - на этой шкале строятся гауссианы
                          - **Верхняя ось X**: Спектральная шкала 0-1 (нормализованная) - только для интерпретации
                        
                        **Каждый компонент = одно чистое медицинское состояние:**
                        
                        - **Центр (μ)**: Позиция состояния на оси PC1 (сырые значения)
                        - **Ширина (σ)**: Разброс образцов в этом состоянии (чем больше σ, тем шире состояние)
                        - **Вес (w)**: Доля образцов, принадлежащих этому состоянию
                        - **Пик (маркер)**: Максимальная плотность компонента
                        
                        **Интерпретация:**
                        - Компоненты слева (низкий PC1) → нормальные состояния
                        - Компоненты справа (высокий PC1) → патологические состояния
                        - Широкий компонент (большой σ) → состояние с большим разбросом
                        - Узкий компонент (маленький σ) → четко определенное состояние
                        
                        **Практическое применение:**
                        - Используйте параметры компонентов для характеристики чистых медицинских состояний
                        - Центры компонентов показывают типичные значения PC1 для каждого состояния
                        - Веса показывают, какая доля образцов относится к каждому состоянию
                        - Для интерпретации можно использовать верхнюю ось (спектральная шкала 0-1)
                        """)
                
                # Интерпретация
                with st.expander("ℹ️ Как интерпретировать loadings?"):
                    st.markdown("""
                    **Положительные loadings (> 0):**
                    - Признак увеличивается вместе с PC1
                    - Высокие значения признака → высокий PC1 → высокий score патологии
                    
                    **Отрицательные loadings (< 0):**
                    - Признак уменьшается при увеличении PC1
                    - Высокие значения признака → низкий PC1 → низкий score (ближе к норме)
                    
                    **Абсолютное значение:**
                    - Показывает силу влияния признака на PC1
                    - Чем больше, тем важнее признак для разделения норма/патология
                    
                    ---
                    
                    ## 🎯 Как важности признаков влияют на положение WSI в шкале?
                    
                    **Формула вычисления PC1:**
                    ```
                    PC1 = Σ (признак_i × loading_i)
                    ```
                    
                    Где:
                    - `признак_i` - нормализованное значение i-го признака (например, Mild_relative_count)
                    - `loading_i` - важность (loading) i-го признака из таблицы выше
                    
                    **Как это работает:**
                    
                    1. **Признаки с большим положительным loading** (например, Dysplasia_mean_relative_area = +0.27):
                       - Если у WSI высокое значение этого признака → большой вклад в PC1
                       - WSI сдвигается вправо по шкале → выше score патологии
                       - **Пример:** WSI с высокой дисплазией → PC1 высокий → Spectrum близко к 1.0 → severe
                    
                    2. **Признаки с большим отрицательным loading**:
                       - Если у WSI высокое значение → отрицательный вклад в PC1
                       - WSI сдвигается влево по шкале → ниже score (ближе к норме)
                       - **Пример:** WSI с низкими патологическими признаками → PC1 низкий → Spectrum близко к 0.0 → normal
                    
                    3. **Признаки с маленьким loading** (близко к 0):
                       - Слабо влияют на положение WSI в шкале
                       - Можно игнорировать при интерпретации
                    
                    **Практический пример:**
                    
                    Два WSI с разными признаками:
                    - **WSI A:** Dysplasia_mean_relative_area = 0.5 (высокая), Mild_relative_count = 0.1 (низкая)
                    - **WSI B:** Dysplasia_mean_relative_area = 0.1 (низкая), Mild_relative_count = 0.5 (высокая)
                    
                    Если Dysplasia имеет loading = +0.27, а Mild = +0.25:
                    - **WSI A** получит больший вклад от Dysplasia → выше PC1 → выше в шкале
                    - **WSI B** получит больший вклад от Mild → тоже высокий PC1, но может быть немного ниже
                    
                    **Итог:** WSI с высокими значениями признаков, имеющих большие положительные loadings, будут находиться выше в шкале (ближе к severe). WSI с низкими значениями этих признаков - ниже (ближе к normal).
                    
                    ---
                    
                    **Примечание:**
                    - Loadings могут немного отличаться при разных наборах данных
                    - Значения из ноутбука (0.272) были на другом наборе образцов
                    - Текущие значения отражают важность признаков на ваших данных
                    """)

            else:
                st.info("Включите спектральный анализ в настройках")

        with tab4:
            st.header("🔍 Анализ конкретных образцов")
            
            if len(df_features) > 0:
                # Выбор образца для анализа
                # КРИТИЧНО: Используем session_state для стабильного значения selectbox
                # ВАЖНО: При изменении selectbox происходит rerun, и Streamlit может сбросить активную вкладку
                # Это известное ограничение Streamlit - вкладки не сохраняют свое состояние при rerun от виджетов
                # К сожалению, нет прямого способа программно сохранить активную вкладку
                sample_names = df_features["image"].tolist()
                selected_sample_key = "selected_sample_analysis"
                default_sample = safe_session_get(selected_sample_key, sample_names[0] if sample_names else None)
                if default_sample not in sample_names:
                    default_sample = sample_names[0] if sample_names else None
                
                selected_sample = st.selectbox(
                    "Выберите образец для анализа",
                    sample_names,
                    index=sample_names.index(default_sample) if default_sample and default_sample in sample_names else 0,
                    key=selected_sample_key,
                    help="Выберите образец, который нужно проанализировать. Например, 9_ibd_mod_6mod"
                )
                # Значение автоматически сохраняется Streamlit через key - НЕ устанавливаем вручную!
                
                if selected_sample:
                    sample_data = df_features[df_features["image"] == selected_sample].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"📊 Данные образца: {selected_sample}")
                        
                        # Показываем значения признаков
                        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
                        if "image" in numeric_cols:
                            numeric_cols.remove("image")
                        
                        sample_features = sample_data[numeric_cols].sort_values(ascending=False)
                        
                        st.markdown("**Топ-10 признаков с наибольшими значениями:**")
                        top_features_df = pd.DataFrame({
                            "Признак": sample_features.head(10).index,
                            "Значение": sample_features.head(10).values
                        })
                        st.dataframe(top_features_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.subheader("📈 Сравнение с другими образцами")
                        
                        # Вычисляем статистику по всем образцам
                        all_stats = df_features[numeric_cols].describe()
                        
                        # Показываем, где находится этот образец относительно других
                        comparison_data = []
                        for feat in numeric_cols:
                            sample_val = sample_data[feat]
                            mean_val = all_stats.loc['mean', feat]
                            std_val = all_stats.loc['std', feat]
                            
                            if std_val > 0:
                                z_score = (sample_val - mean_val) / std_val
                            else:
                                z_score = 0
                            
                            comparison_data.append({
                                "Признак": feat,
                                "Значение": sample_val,
                                "Среднее": mean_val,
                                "Z-score": z_score,
                                "Отклонение": "Выше нормы" if z_score > 1 else ("Ниже нормы" if z_score < -1 else "В норме")
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        comparison_df = comparison_df.sort_values("Z-score", key=abs, ascending=False)
                        
                        # Показываем топ-15 с наибольшими отклонениями
                        st.dataframe(comparison_df.head(15), use_container_width=True, hide_index=True)
                        
                        # Быстрое исключение признаков
                        st.markdown("**🚫 Быстрое исключение признаков:**")
                        st.markdown("Выберите признаки с аномально высокими/низкими значениями для исключения:")
                        
                        # Находим признаки с большими отклонениями
                        high_z_features = comparison_df[comparison_df["Z-score"].abs() > 2]["Признак"].tolist()
                        
                        if high_z_features:
                            st.info(f"⚠️ Обнаружены признаки с большими отклонениями (|Z-score| > 2): {', '.join(high_z_features[:5])}")
                            
                            # Инициализируем session_state для исключенных признаков
                            exclude_key = f"exclude_features_{selected_sample}"
                            if not safe_session_has(exclude_key):
                                safe_session_set(exclude_key, high_z_features[:3])
                            
                            # Фильтруем сохраненные значения, чтобы они были только из доступных опций
                            saved_excluded = safe_session_get(exclude_key, [])
                            valid_excluded_default = [f for f in saved_excluded if f in numeric_cols]
                            if not valid_excluded_default and high_z_features:
                                valid_excluded_default = high_z_features[:3]
                                safe_session_set(exclude_key, valid_excluded_default)
                            
                            st.info("💡 Выберите признаки, затем нажмите кнопку 'Применить исключение'.")
                            
                            with st.form(f"exclude_features_form_{selected_sample}", clear_on_submit=False):
                                # Мультиселект для быстрого исключения
                                features_to_exclude = st.multiselect(
                                    "Выберите признаки для исключения",
                                    numeric_cols,
                                    default=valid_excluded_default,
                                    key=f"exclude_{selected_sample}_form",
                                    help="Эти признаки будут исключены из PCA анализа."
                                )
                                
                                submitted = st.form_submit_button("✅ Применить исключение", use_container_width=True)
                                if submitted:
                                    safe_session_set(exclude_key, features_to_exclude)
                                    # Сохраняем в общий список исключенных признаков
                                    current_excluded = set(safe_session_get("excluded_features", []))
                                    current_excluded.update(features_to_exclude)
                                    safe_session_set("excluded_features", list(current_excluded))
                                    safe_session_set("selection_mode", "Исключить признаки (blacklist)")
                                    st.success(f"✅ Признаки сохранены! Перейдите в раздел '🎯 Выбор признаков' для применения.")
                            
                            # Показываем текущий статус
                            current_excluded_features = safe_session_get(exclude_key, [])
                            if current_excluded_features:
                                st.warning(
                                    f"⚠️ Выбрано {len(current_excluded_features)} признаков для исключения: {', '.join(current_excluded_features[:5])}{'...' if len(current_excluded_features) > 5 else ''}\n\n"
                                    f"**Чтобы применить исключение:**\n"
                                    f"1. Нажмите кнопку 'Применить исключение' выше\n"
                                    f"2. Перейдите в раздел '🎯 Выбор признаков' в боковой панели\n"
                                    f"3. Нажмите кнопку 'Обновить' там для применения изменений"
                                )
                                
                                # Сохраняем в session state для удобства
                                safe_session_set("suggested_exclusions", current_excluded_features)
                    
                    # Если есть результаты спектрального анализа
                    if safe_session_has("analyzer") and use_spectral_analysis:
                        st.subheader("🎯 Результаты спектрального анализа")
                        
                        # Безопасная проверка наличия df_spectrum в locals() или session_state
                        if "df_spectrum" in locals():
                            df_spectrum_val = locals()["df_spectrum"]
                        else:
                            df_spectrum_val = safe_session_get("df_spectrum")
                        if df_spectrum_val is not None:
                            if "df_spectrum" not in locals():
                                analyzer = safe_session_get("analyzer")
                                df_pca = analyzer.transform_pca(df_features)
                                df_spectrum = analyzer.transform_to_spectrum(df_pca, use_gmm_classification=False)
                            
                            sample_spectrum = df_spectrum[df_spectrum["image"] == selected_sample]
                            
                            if len(sample_spectrum) > 0:
                                spectrum_row = sample_spectrum.iloc[0]
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    pc1_val = spectrum_row["PC1"]
                                    st.metric("PC1", f"{pc1_val:.3f}")
                                
                                with col2:
                                    spectrum_val = spectrum_row["PC1_spectrum"]
                                    st.metric("PC1_spectrum", f"{spectrum_val:.3f}")
                                
                                with col3:
                                    if "PC1_mode" in spectrum_row:
                                        mode = spectrum_row["PC1_mode"]
                                        st.metric("Ближайшая мода", mode)
                                
                                with col4:
                                    # Интерпретация
                                    if spectrum_val < 0.3:
                                        interpretation = "🔵 Низкая патология (ближе к норме)"
                                        color = "blue"
                                    elif spectrum_val < 0.7:
                                        interpretation = "🟡 Средняя патология"
                                        color = "orange"
                                    else:
                                        interpretation = "🔴 Высокая патология"
                                        color = "red"
                                    
                                    st.markdown(f"**{interpretation}**")
                                
                                # Визуализация на шкале
                                st.subheader("📍 Расположение на спектральной шкале")
                                fig, ax = plt.subplots(figsize=(12, 2))
                                
                                # Все образцы
                                all_spectrum = df_spectrum["PC1_spectrum"].values
                                ax.scatter(all_spectrum, [0.5] * len(all_spectrum), 
                                          alpha=0.3, s=50, c='gray', label='Все образцы')
                                
                                # Выбранный образец
                                ax.scatter([spectrum_val], [0.5], 
                                          s=300, c=color, marker='*', 
                                          edgecolors='black', linewidth=2, 
                                          label=f'{selected_sample} (score={spectrum_val:.3f})',
                                          zorder=10)
                                
                                # Моды
                                if analyzer.modes:
                                    for mode in analyzer.modes:
                                        mode_spectrum = (mode["position"] - analyzer.pc1_p1) / (
                                            analyzer.pc1_p99 - analyzer.pc1_p1
                                        )
                                        mode_spectrum = np.clip(mode_spectrum, 0.0, 1.0)
                                        ax.axvline(mode_spectrum, color='red', linestyle='--', 
                                                  alpha=0.5, linewidth=1)
                                
                                ax.set_xlim(0, 1)
                                ax.set_ylim(0, 1)
                                ax.set_xlabel("Спектральная шкала (0-1)")
                                ax.set_ylabel("")
                                ax.set_yticks([])
                                ax.set_title(f"Расположение {selected_sample} на шкале патологии")
                                ax.legend(loc='upper right')
                                ax.grid(True, alpha=0.3, axis='x')
                                st.pyplot(fig)
                                
                                # Рекомендации
                                if spectrum_val < 0.3:
                                    st.warning(
                                        f"⚠️ Образец {selected_sample} имеет низкий score ({spectrum_val:.3f}), "
                                        f"что может не соответствовать диагнозу. "
                                        f"Рекомендуется:\n"
                                        f"1. Проверить исходные предсказания патологий\n"
                                        f"2. Исключить некоторые признаки, которые могут мешать\n"
                                        f"3. Проверить значения признаков этого образца"
                                    )
                    
                    # Сравнение с похожими образцами
                    st.subheader("🔬 Сравнение с другими образцами")
                    
                    # Находим похожие образцы по PC1_spectrum
                    # Безопасная проверка наличия df_spectrum в locals() или session_state
                    if "df_spectrum" in locals():
                        df_spectrum_val = locals()["df_spectrum"]
                    else:
                        df_spectrum_val = safe_session_get("df_spectrum")
                    if df_spectrum_val is not None:
                        if "df_spectrum" not in locals():
                            analyzer = safe_session_get("analyzer")
                            df_pca = analyzer.transform_pca(df_features)
                            df_spectrum = analyzer.transform_to_spectrum(df_pca)
                        
                        sample_spectrum_val = df_spectrum[df_spectrum["image"] == selected_sample]["PC1_spectrum"].iloc[0]
                        
                        # Находим ближайшие образцы
                        df_spectrum_sorted = df_spectrum.sort_values("PC1_spectrum")
                        sample_idx = df_spectrum_sorted[df_spectrum_sorted["image"] == selected_sample].index[0]
                        
                        # Берем 2 образца до и 2 после
                        start_idx = max(0, sample_idx - 2)
                        end_idx = min(len(df_spectrum_sorted), sample_idx + 3)
                        similar_samples = df_spectrum_sorted.iloc[start_idx:end_idx]
                        
                        st.dataframe(similar_samples[["image", "PC1", "PC1_spectrum"]], 
                                   use_container_width=True, hide_index=True)
                    
                    # Сравнение с образцами того же типа (ibd_mod, hp_ и т.д.)
                    st.subheader("🔍 Сравнение с образцами того же типа")
                    
                    # Пытаемся найти образцы с похожим паттерном в имени
                    sample_name_lower = selected_sample.lower()
                    similar_type_samples = []
                    
                    # Ищем образцы с похожими паттернами
                    if "ibd_mod" in sample_name_lower:
                        pattern = "ibd_mod"
                        pattern_name = "IBD moderate"
                    elif "hp_" in sample_name_lower:
                        pattern = "hp_"
                        pattern_name = "HP (Helicobacter pylori)"
                    else:
                        pattern = None
                        pattern_name = None
                    
                    if pattern:
                        similar_type = df_features[df_features["image"].str.contains(pattern, case=False, na=False)]
                        if len(similar_type) > 1:  # Больше чем сам образец
                            st.markdown(f"**Образцы типа '{pattern_name}' ({pattern}):**")
                            
                            # Сравнение признаков с группой
                            comparison_features = []
                            for feat in numeric_cols:
                                sample_val = sample_data[feat]
                                other_vals = similar_type[similar_type["image"] != selected_sample][feat].dropna()
                                
                                if len(other_vals) > 0:
                                    other_mean = other_vals.mean()
                                    other_std = other_vals.std()
                                    
                                    if other_std > 0:
                                        z_vs_group = (sample_val - other_mean) / other_std
                                    else:
                                        z_vs_group = 0
                                    
                                    # Процент от среднего в группе
                                    if other_mean != 0:
                                        pct_of_group = (sample_val / other_mean) * 100
                                    else:
                                        pct_of_group = 0
                                    
                                    comparison_features.append({
                                        "Признак": feat,
                                        f"{selected_sample}": f"{sample_val:.4f}",
                                        "Среднее в группе": f"{other_mean:.4f}",
                                        "Z-score vs группа": f"{z_vs_group:.2f}",
                                        "% от среднего": f"{pct_of_group:.1f}%",
                                        "Разница": f"{sample_val - other_mean:.4f}"
                                    })
                            
                            if comparison_features:
                                comp_df = pd.DataFrame(comparison_features)
                                # Сортируем по абсолютному Z-score
                                comp_df["Z_abs"] = comp_df["Z-score vs группа"].str.replace('%', '').astype(float).abs()
                                comp_df = comp_df.sort_values("Z_abs", ascending=False).drop(columns=["Z_abs"])
                                
                                st.markdown("**Признаки, где образец сильно отличается от группы:**")
                                st.dataframe(comp_df.head(20), use_container_width=True, hide_index=True)
                                
                                # Рекомендации по исключению
                                st.markdown("**💡 Рекомендации:**")
                                
                                # Находим признаки, где образец сильно отличается
                                comp_df_numeric = pd.DataFrame(comparison_features)
                                comp_df_numeric["Z_abs"] = pd.to_numeric(comp_df_numeric["Z-score vs группа"].str.replace('%', ''), errors='coerce').abs()
                                comp_df_numeric["Разница_num"] = pd.to_numeric(comp_df_numeric["Разница"], errors='coerce')
                                
                                low_features = comp_df_numeric[comp_df_numeric["Z_abs"] > 1.5]["Признак"].tolist()
                                high_features = comp_df_numeric[comp_df_numeric["Z_abs"] > 1.5]["Признак"].tolist()
                                
                                if low_features:
                                    st.warning(
                                        f"⚠️ У образца **сильно отличающиеся** значения по сравнению с группой:\n"
                                        f"{', '.join(low_features[:5])}\n\n"
                                        f"**Если эти признаки не связаны с тяжестью патологии, попробуйте их исключить.**"
                                    )
                                
                                # Показываем таблицу сравнения всех образцов группы
                                st.markdown("**📊 Сравнение всех образцов группы:**")
                                # Безопасная проверка наличия df_spectrum в locals() или session_state
                                if "df_spectrum" in locals():
                                    df_spectrum_val = locals()["df_spectrum"]
                                else:
                                    df_spectrum_val = safe_session_get("df_spectrum")
                                if df_spectrum_val is not None:
                                    if "df_spectrum" not in locals():
                                        analyzer = safe_session_get("analyzer")
                                        df_pca = analyzer.transform_pca(df_features)
                                        df_spectrum = analyzer.transform_to_spectrum(df_pca, use_gmm_classification=False)
                                    
                                    similar_type_with_spectrum = similar_type.merge(
                                        df_spectrum[["image", "PC1", "PC1_spectrum"]], 
                                        on="image", how="left"
                                    )
                                    display_cols = ["image", "PC1", "PC1_spectrum"]
                                    # Добавляем ключевые патологические признаки
                                    key_features = ["Dysplasia_relative_count", "Mild_relative_count", 
                                                 "Moderate_relative_count", "Neutrophils_relative_count",
                                                 "Plasma Cells_relative_count"]
                                    for kf in key_features:
                                        if kf in similar_type_with_spectrum.columns:
                                            display_cols.append(kf)
                                    
                                    st.dataframe(
                                        similar_type_with_spectrum[display_cols].sort_values("PC1_spectrum", ascending=False),
                                        use_container_width=True, 
                                        hide_index=True
                                    )
                                    
                                    # Анализ: почему 6mod имеет низкий score
                                    if selected_sample == "9_ibd_mod_6mod" or "6mod" in selected_sample:
                                        st.error(
                                            "**🔴 Проблема с образцом 6mod:**\n\n"
                                            "Образец имеет низкий score, хотя должен быть патологическим. "
                                            "Возможные причины:\n\n"
                                            "1. **Низкие значения патологических признаков** (Dysplasia, Mild, Moderate) "
                                            "по сравнению с другими ibd_mod образцами\n"
                                            "2. **Высокие значения других признаков** (Neutrophils, Plasma Cells), "
                                            "которые могут 'перетягивать' в другую сторону\n"
                                            "3. **Проблема в исходных предсказаниях** - возможно, модель не детектировала "
                                            "патологии для этого образца\n\n"
                                            "**Рекомендации:**\n"
                                            "- Проверьте исходный JSON файл с предсказаниями\n"
                                            "- Попробуйте исключить признаки с высокими значениями (Neutrophils, Plasma Cells)\n"
                                            "- Или используйте только патологические признаки (Dysplasia, Mild, Moderate)"
                                        )
            else:
                st.info("Загрузите данные для анализа образцов")

        with tab5:
            st.header("Статистика")

            if len(df_features) > 0:
                numeric_cols = df_features.select_dtypes(
                    include=[np.number]
                ).columns.tolist()

                if numeric_cols:
                    st.subheader("Описательная статистика")
                    st.dataframe(
                        df_features[numeric_cols].describe(),
                        use_container_width=True,
                    )

                    # Корреляционная матрица
                    if len(numeric_cols) > 1:
                        st.subheader("Корреляционная матрица")
                        
                        # Используем функцию из preprocessing
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                            tmp_path = Path(tmp_file.name)
                        
                        try:
                            preprocessing.visualize_correlations(
                                df_features,
                                feature_columns=numeric_cols,
                                save_path=tmp_path
                            )
                            if tmp_path.exists():
                                st.image(str(tmp_path))
                        finally:
                            if tmp_path.exists():
                                tmp_path.unlink()
                        
                        # Анализ высоко коррелированных признаков
                        with st.expander("🔍 Анализ высоко коррелированных признаков"):
                            threshold = st.slider("Порог корреляции", 0.7, 0.99, 0.95, 0.01)
                            highly_corr = preprocessing.find_highly_correlated_features(
                                df_features,
                                threshold=threshold,
                                feature_columns=numeric_cols
                            )
                            
                            if highly_corr:
                                st.warning(f"Найдено {len(highly_corr)} пар признаков с корреляцией >= {threshold}")
                                corr_df = pd.DataFrame(
                                    highly_corr,
                                    columns=["Признак 1", "Признак 2", "Корреляция"]
                                )
                                st.dataframe(corr_df, use_container_width=True)
                                
                                if st.button("Удалить избыточные признаки"):
                                    df_cleaned, removed = preprocessing.remove_redundant_features(
                                        df_features,
                                        threshold=threshold,
                                        feature_columns=numeric_cols
                                    )
                                    if removed:
                                        st.success(f"Удалено {len(removed)} признаков: {', '.join(removed)}")
                                        safe_session_set("df_results", df_cleaned)
                                        st.rerun()
                            else:
                                st.info("Нет высоко коррелированных признаков")

        # Вкладка инференса
        with tab_inference:
            st.header("🔮 Инференс для новых WSI")
            st.markdown("Примените обученную модель к новым WSI из выбранной директории.")
                
            # Проверяем, есть ли обученная модель
            analyzer = safe_session_get("analyzer", None)
            if analyzer is None:
                # Пробуем загрузить модель из эксперимента, если используется эксперимент
                if use_experiment_data and safe_session_has("experiment_dir"):
                    experiment_dir = Path(safe_session_get("experiment_dir"))
                    model_path = experiment_dir / "spectral_analyzer.pkl"
                    if model_path.exists():
                        try:
                            analyzer = spectral_analysis.SpectralAnalyzer()
                            analyzer.load(model_path)
                            safe_session_set("analyzer", analyzer)
                            st.success(f"✅ Загружена модель из эксперимента: {experiment_dir.name}")
                            st.info(f"ℹ️ Модель использует {len(analyzer.feature_columns) if analyzer.feature_columns else 0} признаков")
                        except Exception as e:
                            st.error(f"❌ Не удалось загрузить модель из эксперимента: {e}")
                            analyzer = None
                    else:
                        st.warning("⚠️ Модель не найдена в эксперименте.")
                        analyzer = None
                else:
                    st.warning("⚠️ Сначала выполните спектральный анализ для обучения модели.")
                    st.info("💡 Перейдите на вкладку '🔬 Спектральный анализ' и выполните анализ данных.")
                    analyzer = None
            else:
                analyzer = safe_session_get("analyzer")
                # Показываем информацию о модели
                if use_experiment_data and safe_session_has("experiment_dir"):
                    experiment_dir = Path(safe_session_get("experiment_dir"))
                    st.info(f"ℹ️ Используется модель из эксперимента: **{experiment_dir.name}**")
                if analyzer.feature_columns:
                    st.caption(f"Модель обучена на {len(analyzer.feature_columns)} признаках")
            
            if analyzer is None:
                st.stop()
            
            inference_dir_val = safe_session_get("inference_dir")
            if not inference_dir_val or not Path(inference_dir_val).exists():
                st.info("💡 Выберите директорию для инференса в боковой панели (секция '🔮 Инференс').")
            else:
                inference_dir = Path(inference_dir_val)
                json_files = list(inference_dir.glob("*.json"))
                
                if not json_files:
                    st.warning(f"⚠️ В директории {inference_dir} нет JSON файлов.")
                    st.info(f"💡 Поместите JSON файлы с предсказаниями в директорию `{inference_dir}`")
                else:
                    # ВАЖНО: Определяем тип признаков по модели, а не по настройкам!
                    # Модель была обучена на определенных признаках, нужно использовать те же
                    use_relative_features_for_inference = True  # По умолчанию
                    if analyzer.feature_columns:
                        # Проверяем, какие признаки использовались при обучении
                        # Если есть хотя бы один относительный признак - используем относительные
                        has_relative = any("_relative_" in feat for feat in analyzer.feature_columns)
                        has_absolute = any("_relative_" not in feat and feat not in ["image"] for feat in analyzer.feature_columns)
                        
                        if has_relative and not has_absolute:
                            use_relative_features_for_inference = True
                        elif has_absolute and not has_relative:
                            use_relative_features_for_inference = False
                        else:
                            # Если смешанные, используем настройки из session_state
                            settings = safe_session_get("settings", {})
                            use_relative_features_for_inference = settings.get("use_relative_features", True)
                            st.warning("⚠️ Модель использует смешанные признаки. Используются настройки из session_state.")
                    else:
                        # Если нет информации о признаках, используем настройки из session_state
                        settings = safe_session_get("settings", {})
                        use_relative_features_for_inference = settings.get("use_relative_features", True)
                    
                    st.info(f"ℹ️ Тип признаков для инференса: **{'Относительные' if use_relative_features_for_inference else 'Абсолютные'}** (определено по модели)")
                    
                    # Ключ кэша для инференса
                    inference_cache_key = f"inference_{inference_dir}_{hash(str(sorted([f.name for f in json_files])))}"
                    
                    if safe_session_has(inference_cache_key):
                        df_inference_spectrum = safe_session_get(inference_cache_key)
                        st.info(f"✅ Загружены результаты инференса для {len(df_inference_spectrum)} образцов (из кэша)")
                    else:
                        with st.spinner(f"Выполняется инференс для {len(json_files)} файлов..."):
                            try:
                                # Загружаем предсказания
                                inference_predictions = {}
                                for json_file in json_files:
                                    try:
                                        preds = domain.predictions_from_json(str(json_file))
                                        image_name = json_file.stem
                                        inference_predictions[image_name] = preds
                                    except Exception as e:
                                        st.warning(f"Ошибка при загрузке {json_file.name}: {e}")
                                
                                if not inference_predictions:
                                    st.error("❌ Не удалось загрузить ни одного файла для инференса")
                                    df_inference_spectrum = None
                                else:
                                    # Агрегируем данные
                                    inference_rows = []
                                    for image_name, preds in inference_predictions.items():
                                        pred_stats = aggregate.aggregate_predictions_from_dict(
                                            preds, image_name
                                        )
                                        inference_rows.append(pred_stats)
                                    
                                    df_inference = pd.DataFrame(inference_rows)
                        
                                    # Создаем признаки (используем ТОЧНО те же настройки, что и для обучения модели)
                                    if use_relative_features_for_inference:
                                        df_inference_features_full = aggregate.create_relative_features(df_inference)
                                    else:
                                        df_inference_features_full = df_inference.copy()
                                        # Удаляем относительные признаки, если они случайно попали
                                        relative_cols = [col for col in df_inference_features_full.columns if 'relative' in col.lower()]
                                        if relative_cols:
                                            df_inference_features_full = df_inference_features_full.drop(columns=relative_cols)
                                        # Удаляем White space
                                        white_space_cols = [col for col in df_inference_features_full.columns if 'white space' in col.lower()]
                                        if white_space_cols:
                                            df_inference_features_full = df_inference_features_full.drop(columns=white_space_cols)
                                    
                                    # Показываем информацию о созданных признаках
                                    st.caption(f"Создано признаков: {len(df_inference_features_full.columns) - 1} (тип: {'относительные' if use_relative_features_for_inference else 'абсолютные'})")
                        
                                    # ВАЖНО: Используем ТОЧНО те же признаки, что были при обучении модели
                                    # Это гарантирует идентичные результаты между обучением и инференсом
                                    if analyzer.feature_columns is not None:
                                        # Берем только те признаки, которые использовались при обучении
                                        required_features = analyzer.feature_columns.copy()
                                        
                                        # Проверяем наличие всех необходимых признаков
                                        missing_features = [f for f in required_features if f not in df_inference_features_full.columns]
                                        
                                        # Автоматически добавляем недостающие признаки с нулевыми значениями
                                        if missing_features:
                                            for feat in missing_features:
                                                df_inference_features_full[feat] = 0.0
                                            
                                            st.info(f"ℹ️ Автоматически добавлено {len(missing_features)} недостающих признаков с нулевыми значениями: {', '.join(missing_features[:3])}{'...' if len(missing_features) > 3 else ''}")
                                        
                                        # Используем ТОЛЬКО признаки из модели (в том же порядке)
                                        # Это критически важно для получения идентичных результатов!
                                        df_inference_features = df_inference_features_full[["image"] + required_features].copy()
                                        
                                        st.info(f"ℹ️ Используется {len(required_features)} признаков из обученной модели (те же, что при обучении)")
                        
                                        # Преобразуем через PCA
                                        df_inference_pca = analyzer.transform_pca(df_inference_features)
                                        
                                        # Преобразуем в спектральную шкалу
                                        df_inference_spectrum = analyzer.transform_to_spectrum(
                                            df_inference_pca, 
                                            use_gmm_classification=False
                                        )
                                        
                                        # Сохраняем в кэш
                                        safe_session_set(inference_cache_key, df_inference_spectrum)
                                        st.success(f"✅ Инференс выполнен для {len(df_inference_spectrum)} образцов")
                                    else:
                                        st.error("❌ Модель не содержит информацию о признаках")
                                        df_inference_spectrum = None
                            
                            except Exception as e:
                                st.error(f"❌ Ошибка при инференсе: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                                df_inference_spectrum = None
                    
                    # Показываем результаты инференса
                    df_inference_spectrum = safe_session_get(inference_cache_key)
                    if df_inference_spectrum is not None:
                        # Получаем обучающие данные для сравнения
                        df_spectrum_train = safe_session_get("df_spectrum")
                        if df_spectrum_train is None:
                            df_spectrum_train = None
                        
                        # График с точками инференса
                        st.markdown("**📊 Распределение WSI на спектральной шкале (с точками инференса)**")
                        fig_inference, ax_inference = plt.subplots(figsize=(14, 6))
                    
                        # Гистограмма для обучающих данных (если есть)
                        if df_spectrum_train is not None:
                            spectrum_values_train = df_spectrum_train["PC1_spectrum"].dropna().values
                            counts_train, bins_train, patches_train = ax_inference.hist(
                                spectrum_values_train,
                                bins=30,
                                alpha=0.4,
                                color='lightblue',
                                edgecolor='black',
                                linewidth=0.5,
                                label='Обучающие данные (гистограмма)'
                            )
                            
                            # Точки для обучающих данных
                            np.random.seed(42)
                            point_heights_train = []
                            for val in spectrum_values_train:
                                bin_idx = np.digitize(val, bins_train) - 1
                                bin_idx = np.clip(bin_idx, 0, len(counts_train) - 1)
                                height = counts_train[bin_idx] + np.random.uniform(0.1, 0.3)
                                point_heights_train.append(height)
                            
                            point_heights_train = np.array(point_heights_train)
                            colors_train = plt.cm.RdYlGn_r(spectrum_values_train)
                            ax_inference.scatter(
                                spectrum_values_train, point_heights_train,
                                alpha=0.6, s=100, c=colors_train,
                                edgecolors='black', linewidth=1, zorder=5,
                                label='Обучающие данные'
                            )
                        else:
                            # Если нет обучающих данных, создаем пустую гистограмму для масштаба
                            bins_train = np.linspace(0, 1, 31)
                            counts_train = np.zeros(30)
                        
                        # Точки для инференса
                        spectrum_values_inference = df_inference_spectrum["PC1_spectrum"].dropna().values
                        image_names_inference = df_inference_spectrum.loc[df_inference_spectrum["PC1_spectrum"].notna(), "image"].values
                        
                        if len(spectrum_values_inference) > 0:
                            point_heights_inference = []
                            for val in spectrum_values_inference:
                                bin_idx = np.digitize(val, bins_train) - 1
                                bin_idx = np.clip(bin_idx, 0, len(counts_train) - 1)
                                height = counts_train[bin_idx] + np.random.uniform(0.5, 0.8) if len(counts_train) > 0 else 1.0
                                point_heights_inference.append(height)
                            
                            point_heights_inference = np.array(point_heights_inference)
                            colors_inference = plt.cm.RdYlGn_r(spectrum_values_inference)
                            
                            # Рисуем точки инференса другим стилем
                            ax_inference.scatter(
                                spectrum_values_inference, point_heights_inference,
                                alpha=0.9, s=200, c=colors_inference,
                                edgecolors='red', linewidth=2.5, zorder=10,
                                marker='*', label='Инференс (новые WSI)'
                            )
                            
                            # Подписи для точек инференса
                            for i, (x, y, name) in enumerate(zip(spectrum_values_inference, point_heights_inference, image_names_inference)):
                                short_name = name[:20] + "..." if len(name) > 20 else name
                                ax_inference.annotate(
                                    short_name, (x, y), xytext=(5, 5),
                                    textcoords='offset points', fontsize=9, alpha=0.9,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='red', linewidth=1.5),
                                    fontweight='bold'
                                )
                                        
                        # Отметка мод (если есть обучающие данные)
                        if analyzer.modes and df_spectrum_train is not None:
                            for mode in analyzer.modes:
                                mode_spectrum = (mode["position"] - analyzer.pc1_p1) / (
                                    analyzer.pc1_p99 - analyzer.pc1_p1
                                )
                                mode_spectrum = np.clip(mode_spectrum, 0.0, 1.0)
                                ax_inference.axvline(
                                    mode_spectrum,
                                    color="r",
                                    linestyle="--",
                                    linewidth=2,
                                    alpha=0.7,
                                    label="Мода" if mode == analyzer.modes[0] else ""
                                )
                        
                        ax_inference.set_xlabel("Спектральная шкала (0-1)", fontsize=12)
                        ax_inference.set_ylabel("Частота (количество образцов в bin)", fontsize=12)
                        train_count = len(spectrum_values_train) if df_spectrum_train is not None else 0
                        ax_inference.set_title(
                            f"Распределение WSI на спектральной шкале\n"
                            f"Обучающие данные: {train_count} образцов | "
                            f"Инференс: {len(spectrum_values_inference) if len(spectrum_values_inference) > 0 else 0} образцов",
                            fontsize=13
                        )
                        ax_inference.set_xlim(0, 1)
                        ax_inference.set_ylim(bottom=0)
                        ax_inference.grid(True, alpha=0.3, axis="both")
                        ax_inference.legend(loc='upper right')
                        plt.tight_layout()
                        st.pyplot(fig_inference)
                        
                        # Таблица с результатами инференса
                        st.markdown("**📋 Результаты инференса для каждого WSI**")
                        st.markdown(
                            "**Эта таблица показывает каждый WSI из инференса отдельно** - здесь вы можете увидеть точное значение "
                            "спектральной шкалы для каждого нового образца."
                        )
                        inference_display_cols = ["image", "PC1", "PC1_spectrum"]
                        if "PC1_mode" in df_inference_spectrum.columns:
                            inference_display_cols.append("PC1_mode")
                        
                        st.dataframe(
                            df_inference_spectrum[inference_display_cols].sort_values(
                                by="PC1_spectrum", ascending=False
                            ),
                            use_container_width=True,
                        )
                            
                        # Скачивание результатов инференса
                        csv_inference = df_inference_spectrum[inference_display_cols].to_csv(index=False)
                        st.download_button(
                            label="📥 Скачать результаты инференса (CSV)",
                            data=csv_inference,
                            file_name=f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )

    else:
        # Нет данных для отображения
        if use_experiment_data:
            st.warning("⚠️ Данные из эксперимента не загружены. Выберите эксперимент в боковой панели.")
        else:
            st.info("👈 Загрузите JSON файлы с предсказаниями в боковой панели")


if __name__ == "__main__":
    render_dashboard()

