"""
FastAPI сервер для анализа патологий Whole Slide Images.

Предоставляет REST API с функциональностью аналогичной Streamlit dashboard:
- Загрузка данных из разных источников
- Агрегация predictions и создание признаков
- PCA анализ и scoring
- Спектральный анализ
- Feature selection и evaluation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import io

# Импорты из scale модулей
from scale import aggregate, pca_scoring, spectral_analysis, domain
from scale.dashboard_common import load_predictions_from_gdrive
from scale.gcs_integration import load_json_from_gcs_bucket
from scale import dashboard_experiment_selector
from model_development.feature_selection_automated import evaluate_feature_set, identify_sample_type

# Настройка логирования с детальным форматированием
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# Включаем логирование для google библиотек на уровне WARNING, чтобы видеть только важные сообщения
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('googleapiclient').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)

# Создаем FastAPI приложение
app = FastAPI(
    title="Pathology Analysis API",
    description="API для анализа патологий Whole Slide Images",
    version="1.0.0"
)

# CORS middleware для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic модели для запросов/ответов
class LoadDataRequest(BaseModel):
    source: str = Field(..., description="Источник данных: 'directory', 'gdrive', 'gcs', 'experiment'")
    path: Optional[str] = Field(None, description="Путь к директории или URL/bucket")
    bucket_name: Optional[str] = Field(None, description="Имя GCS bucket (для source='gcs')")
    prefix: Optional[str] = Field("", description="Prefix для GCS (для source='gcs')")
    experiment_name: Optional[str] = Field(None, description="Имя эксперимента (для source='experiment')")
    experiments_dir: Optional[str] = Field("experiments", description="Директория с экспериментами")


class AggregateRequest(BaseModel):
    cache_key: str = Field(..., description="Cache key из /api/v1/load-data")


class FeatureEvaluationRequest(BaseModel):
    df_features_cache_key: str = Field(..., description="Cache key для df_features из /api/v1/aggregate")
    feature_columns: List[str] = Field(..., description="Список признаков для оценки")
    mod_samples: Optional[List[str]] = Field(None, description="Список mod образцов (опционально, определяется автоматически)")
    normal_samples: Optional[List[str]] = Field(None, description="Список normal образцов (опционально, определяется автоматически)")


class PCAScoreRequest(BaseModel):
    df_features_cache_key: str = Field(..., description="Cache key для df_features из /api/v1/aggregate")
    feature_columns: List[str] = Field(..., description="Список признаков для PCA")
    use_relative_features: bool = Field(True, description="Использовать относительные признаки")


class SpectralAnalysisRequest(BaseModel):
    df_features_cache_key: str = Field(..., description="Cache key для df_features из /api/v1/aggregate")
    feature_columns: List[str] = Field(..., description="Список признаков для спектрального анализа")
    use_relative_features: bool = Field(True, description="Использовать относительные признаки")
    percentile_low: float = Field(1.0, description="Нижний процентиль")
    percentile_high: float = Field(99.0, description="Верхний процентиль")
    use_gmm_classification: bool = Field(False, description="Использовать GMM классификацию для спектра")


class LoadExperimentRequest(BaseModel):
    experiment_name: str = Field(..., description="Имя эксперимента или путь к директории")
    experiments_dir: Optional[str] = Field("experiments", description="Базовая директория с экспериментами")


class CreateSpectrumRequest(BaseModel):
    df_features_cache_key: str = Field(..., description="Cache key для df_features из /api/v1/aggregate")
    feature_columns: List[str] = Field(..., description="Список признаков для PCA")
    percentile_low: float = Field(1.0, description="Нижний процентиль")
    percentile_high: float = Field(99.0, description="Верхний процентиль")
    use_relative_features: bool = Field(True, description="Использовать относительные признаки")
    use_gmm_classification: bool = Field(False, description="Использовать GMM классификацию для спектра")


class SaveExperimentRequest(BaseModel):
    experiment_name: str = Field(..., description="Имя эксперимента")
    df_features_cache_key: str = Field(..., description="Cache key для df_features из /api/v1/aggregate")
    feature_columns: List[str] = Field(..., description="Список выбранных признаков")
    metrics: Optional[Dict[str, float]] = Field(None, description="Метрики эксперимента (score, separation, etc.)")
    use_relative_features: bool = Field(True, description="Использовать относительные признаки")
    method: str = Field("api_manual", description="Метод создания эксперимента")
    experiments_dir: str = Field("experiments", description="Директория для сохранения экспериментов")


# Глобальное хранилище данных (в продакшене использовать Redis или БД)
_data_cache: Dict[str, Any] = {}


# Подключаем статические файлы для веб-интерфейса
try:
    from pathlib import Path
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


@app.get("/")
async def root():
    """Корневой endpoint с информацией об API или веб-интерфейсом."""
    try:
        from pathlib import Path
        static_file = Path(__file__).parent / "static" / "index.html"
        if static_file.exists():
            return FileResponse(str(static_file))
    except Exception:
        pass
    
    return {
        "name": "Pathology Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "load_data": "/api/v1/load-data",
            "aggregate": "/api/v1/aggregate",
            "evaluate_features": "/api/v1/evaluate-features",
            "pca_score": "/api/v1/pca-score",
            "spectral_analysis": "/api/v1/spectral-analysis",
            "create_spectrum": "/api/v1/create-spectrum",
            "list_experiments": "/api/v1/experiments",
            "load_experiment": "/api/v1/load-experiment",
            "save_experiment": "/api/v1/save-experiment",
            "load_progress": "/api/v1/load-progress",
            "download_csv": "/api/v1/download-csv",
            "health": "/api/v1/health",
        },
        "web_interface": "/static/index.html"
    }


@app.post("/api/v1/load-data")
async def load_data(request: LoadDataRequest):
    """
    Загружает данные из указанного источника.
    
    Sources:
    - 'directory': загрузка из локальной директории
    - 'gdrive': загрузка из Google Drive
    - 'gcs': загрузка из Google Cloud Storage
    """
    try:
        predictions = {}
        
        if request.source == "directory":
            if not request.path:
                raise HTTPException(status_code=400, detail="Path required for directory source")
            
            predictions_dir = Path(request.path)
            if not predictions_dir.exists():
                raise HTTPException(status_code=404, detail=f"Directory not found: {request.path}")
            
            json_files = list(predictions_dir.glob("*.json"))
            if not json_files:
                raise HTTPException(status_code=404, detail=f"No JSON files found in {request.path}")
            
            # Используем тот же метод, что и dashboard: domain.predictions_from_json()
            # Он возвращает dict[str, list[Prediction]] (объекты Prediction)
            predictions_converted = {}
            total_files = len(json_files)
            
            # Создаем callback для прогресса (используем фиксированный ключ для последнего активного прогресса)
            progress_key = "load_progress_latest"
            progress_data = {"current": 0, "total": total_files, "message": f"Найдено {total_files} файлов", "status": "loading", "progress": 0.0}
            _data_cache[progress_key] = progress_data.copy()  # Устанавливаем начальный прогресс сразу
            
            def log_progress(current, total, message=""):
                # Сохраняем прогресс в кэш для получения через другой endpoint
                progress_data["current"] = current
                progress_data["total"] = total
                progress_data["message"] = message
                progress_data["progress"] = current / total if total > 0 else 0
                _data_cache[progress_key] = progress_data.copy()
            
            log_progress(0, total_files, f"Найдено {total_files} файлов")
            
            for idx, json_file in enumerate(json_files):
                try:
                    preds = domain.predictions_from_json(str(json_file))
                    image_name = json_file.stem
                    predictions_converted[image_name] = preds
                    log_progress(idx + 1, total_files, f"Загружен {json_file.name} ({idx + 1}/{total_files})")
                except Exception as e:
                    logger.error(f"Error loading {json_file.name}: {e}")
                    log_progress(idx + 1, total_files, f"Ошибка при загрузке {json_file.name}")
                    continue
            
            # Очищаем прогресс после завершения
            if progress_key in _data_cache:
                del _data_cache[progress_key]
            
            # Очищаем прогресс после завершения
            if progress_key in _data_cache:
                del _data_cache[progress_key]
        
        elif request.source == "gdrive":
            if not request.path:
                raise HTTPException(status_code=400, detail="Google Drive URL required")
            
            import time
            start_time = time.time()
            progress_key = "load_progress_latest"
            
            try:
                logger.info(f"[GDRIVE] Начало загрузки из Google Drive. URL: {request.path}")
                
                # Создаем callback для прогресса (используем фиксированный ключ для последнего активного прогресса)
                progress_data = {"current": 0, "total": 0, "message": "Инициализация Google Drive...", "status": "loading", "progress": 0.0}
                _data_cache[progress_key] = progress_data.copy()  # Устанавливаем начальный прогресс сразу
                
                def log_progress(message):
                    # Логируем все сообщения для диагностики
                    logger.info(f"[GDRIVE] Progress: {message}")
                    # Парсим прогресс из сообщений типа "[1/36]" или "📥 Начинаю загрузку 36 JSON файлов..."
                    import re
                    progress_match = re.search(r'\[(\d+)/(\d+)\]', message)
                    if progress_match:
                        current = int(progress_match.group(1))
                        total = int(progress_match.group(2))
                        progress_data["current"] = current
                        progress_data["total"] = total
                        progress_data["message"] = message
                        progress_data["progress"] = current / total if total > 0 else 0
                        _data_cache[progress_key] = progress_data.copy()
                    elif "Начинаю загрузку" in message or "Начало загрузки" in message:
                        total_match = re.search(r'(\d+)\s+JSON файлов', message)
                        if total_match:
                            total = int(total_match.group(1))
                            progress_data["total"] = total
                            progress_data["message"] = message
                            progress_data["progress"] = 0.0
                            _data_cache[progress_key] = progress_data.copy()
                    else:
                        # Обновляем сообщение, но сохраняем текущий прогресс
                        progress_data["message"] = message
                        _data_cache[progress_key] = progress_data.copy()
                
                # Этап 1: Извлечение folder_id
                logger.info(f"[GDRIVE] Этап 1: Извлечение folder_id из URL...")
                init_time = time.time()
                try:
                    from scale.dashboard_common import extract_folder_id_from_url
                    folder_id = extract_folder_id_from_url(request.path)
                    elapsed = time.time() - init_time
                    logger.info(f"[GDRIVE] Этап 1 завершен за {elapsed:.2f}с. Folder ID: {folder_id}")
                    if elapsed > 5.0:
                        logger.warning(f"[GDRIVE] Этап 1 занял {elapsed:.2f}с - возможно медленная работа")
                except Exception as e:
                    elapsed = time.time() - init_time
                    logger.error(f"[GDRIVE] Этап 1 завершился с ошибкой после {elapsed:.2f}с: {e}", exc_info=True)
                    raise
                
                if not folder_id:
                    error_msg = "❌ Не удалось извлечь ID папки из URL"
                    logger.error(f"[GDRIVE] {error_msg}")
                    raise HTTPException(status_code=400, detail=error_msg)
                
                # Этап 2: Получение credentials
                logger.info(f"[GDRIVE] Этап 2: Получение credentials...")
                logger.info(f"[GDRIVE] Проверка переменных окружения...")
                import os
                has_env_creds = bool(os.getenv('GOOGLE_DRIVE_CREDENTIALS_JSON_B64') or os.getenv('GOOGLE_DRIVE_CREDENTIALS_JSON'))
                logger.info(f"[GDRIVE] Env credentials найдены: {has_env_creds}")
                
                # Проверяем наличие token файла
                token_paths = ['.gdrive_token.json', Path.home() / '.gdrive_token.json']
                token_exists = any(Path(p).exists() for p in token_paths)
                logger.info(f"[GDRIVE] Token файл существует: {token_exists}")
                
                init_time = time.time()
                try:
                    from scale.dashboard_common import get_credentials
                    logger.info(f"[GDRIVE] Вызов get_credentials...")
                    credentials = get_credentials(credentials_path=None)
                    elapsed = time.time() - init_time
                    logger.info(f"[GDRIVE] Этап 2 завершен за {elapsed:.2f}с. Credentials получены: {credentials is not None}")
                    if elapsed > 10.0:
                        logger.warning(f"[GDRIVE] ⚠️ Этап 2 занял {elapsed:.2f}с - возможно зависание при получении credentials")
                    if not credentials:
                        logger.error(f"[GDRIVE] ❌ Credentials не получены после {elapsed:.2f}с - проверьте наличие token файла или авторизацию")
                        logger.error(f"[GDRIVE] Token файлы проверены: {token_paths}")
                        logger.error(f"[GDRIVE] Env credentials: {has_env_creds}")
                except Exception as e:
                    elapsed = time.time() - init_time
                    logger.error(f"[GDRIVE] ❌ Этап 2 завершился с ошибкой после {elapsed:.2f}с: {e}", exc_info=True)
                    logger.error(f"[GDRIVE] Traceback:", exc_info=True)
                    raise
                
                if not credentials:
                    error_msg = "⚠️ Требуется авторизация Google Drive"
                    logger.error(f"[GDRIVE] {error_msg}")
                    raise HTTPException(status_code=401, detail=error_msg)
                
                # Этап 3: Загрузка данных
                logger.info(f"[GDRIVE] Этап 3: Загрузка данных из папки...")
                init_time = time.time()
                try:
                    from scale.dashboard_common import load_predictions_from_gdrive
                    predictions_raw = load_predictions_from_gdrive(request.path, log_callback=log_progress)
                    elapsed = time.time() - init_time
                    logger.info(f"[GDRIVE] Этап 3 завершен за {elapsed:.2f}с. Загружено файлов: {len(predictions_raw)}")
                    if elapsed > 60.0:
                        logger.warning(f"[GDRIVE] Этап 3 занял {elapsed:.2f}с - возможно медленная загрузка файлов")
                except Exception as e:
                    elapsed = time.time() - init_time
                    logger.error(f"[GDRIVE] Этап 3 завершился с ошибкой после {elapsed:.2f}с: {e}", exc_info=True)
                    raise
                
                # Этап 4: Конвертация данных
                logger.info(f"[GDRIVE] Этап 4: Конвертация данных в Prediction объекты...")
                init_time = time.time()
                predictions_converted = {}
                for name, data in predictions_raw.items():
                    predictions_converted[name] = domain.predictions_from_dict(data)
                logger.info(f"[GDRIVE] Этап 4 завершен за {time.time() - init_time:.2f}с. Конвертировано: {len(predictions_converted)}")
                
                total_time = time.time() - start_time
                logger.info(f"[GDRIVE] Полная загрузка завершена за {total_time:.2f}с. Всего файлов: {len(predictions_converted)}")
                
                # Очищаем прогресс после завершения
                if progress_key in _data_cache:
                    del _data_cache[progress_key]
            except HTTPException:
                raise
            except Exception as e:
                total_time = time.time() - start_time
                logger.error(f"[GDRIVE] Ошибка после {total_time:.2f}с: {str(e)}", exc_info=True)
                # Очищаем прогресс при ошибке
                if progress_key in _data_cache:
                    del _data_cache[progress_key]
                raise HTTPException(status_code=500, detail=f"Error loading from Google Drive: {str(e)}")
        
        elif request.source == "gcs":
            if not request.bucket_name:
                raise HTTPException(status_code=400, detail="Bucket name required for GCS source")
            
            import time
            start_time = time.time()
            progress_key = "load_progress_latest"
            
            try:
                logger.info(f"[GCS] Начало загрузки из GCS. Bucket: {request.bucket_name}, Prefix: {request.prefix or ''}")
                
                # Создаем callback для прогресса (используем фиксированный ключ для последнего активного прогресса)
                progress_data = {"current": 0, "total": 0, "message": f"Инициализация GCS bucket: {request.bucket_name}...", "status": "loading", "progress": 0.0}
                _data_cache[progress_key] = progress_data.copy()  # Устанавливаем начальный прогресс сразу
                
                def log_progress(message):
                    # Логируем все сообщения для диагностики
                    logger.info(f"[GCS] Progress: {message}")
                    # Парсим прогресс из сообщений типа "[1/36]" или "📥 Начинаю загрузку 36 JSON файлов..."
                    import re
                    progress_match = re.search(r'\[(\d+)/(\d+)\]', message)
                    if progress_match:
                        current = int(progress_match.group(1))
                        total = int(progress_match.group(2))
                        progress_data["current"] = current
                        progress_data["total"] = total
                        progress_data["message"] = message
                        progress_data["progress"] = current / total if total > 0 else 0
                        _data_cache[progress_key] = progress_data.copy()
                    elif "Начинаю загрузку" in message or "Начало загрузки" in message or "Найдено" in message:
                        total_match = re.search(r'(\d+)\s+JSON файлов|(\d+)\s+файлов', message)
                        if total_match:
                            total = int(total_match.group(1) or total_match.group(2))
                            progress_data["total"] = total
                            progress_data["message"] = message
                            progress_data["progress"] = 0.0
                            _data_cache[progress_key] = progress_data.copy()
                    else:
                        # Обновляем сообщение, но сохраняем текущий прогресс
                        progress_data["message"] = message
                        _data_cache[progress_key] = progress_data.copy()
                
                # Этап 1: Создание GCS клиента
                logger.info(f"[GCS] Этап 1: Создание GCS клиента...")
                init_time = time.time()
                try:
                    from scale.gcs_integration import _get_gcs_client
                    gcs_client = _get_gcs_client(log_callback=lambda m: logger.info(f"[GCS] Client init: {m}"))
                    elapsed = time.time() - init_time
                    logger.info(f"[GCS] Этап 1 завершен за {elapsed:.2f}с. Client создан: {gcs_client is not None}")
                    if elapsed > 10.0:
                        logger.warning(f"[GCS] Этап 1 занял {elapsed:.2f}с - возможно зависание при создании клиента")
                    if not gcs_client:
                        logger.error(f"[GCS] GCS клиент не создан после {elapsed:.2f}с - проверьте credentials")
                except Exception as e:
                    elapsed = time.time() - init_time
                    logger.error(f"[GCS] Этап 1 завершился с ошибкой после {elapsed:.2f}с: {e}", exc_info=True)
                    raise
                
                if not gcs_client:
                    error_msg = "❌ Не удалось создать GCS клиент"
                    logger.error(f"[GCS] {error_msg}")
                    raise HTTPException(status_code=500, detail=error_msg)
                
                # Этап 2: Получение списка файлов
                logger.info(f"[GCS] Этап 2: Получение списка файлов из bucket...")
                init_time = time.time()
                try:
                    from scale.gcs_integration import list_files_from_gcs_bucket
                    files = list_files_from_gcs_bucket(
                        request.bucket_name,
                        prefix=request.prefix or "",
                        file_type='json',
                        log_callback=lambda m: logger.info(f"[GCS] List files: {m}")
                    )
                    elapsed = time.time() - init_time
                    logger.info(f"[GCS] Этап 2 завершен за {elapsed:.2f}с. Найдено файлов: {len(files) if files else 0}")
                    if elapsed > 30.0:
                        logger.warning(f"[GCS] Этап 2 занял {elapsed:.2f}с - возможно зависание при получении списка файлов")
                except Exception as e:
                    elapsed = time.time() - init_time
                    logger.error(f"[GCS] Этап 2 завершился с ошибкой после {elapsed:.2f}с: {e}", exc_info=True)
                    raise
                
                if not files:
                    error_msg = "⚠️ JSON файлы не найдены в указанном bucket/prefix"
                    logger.warning(f"[GCS] {error_msg}")
                    raise HTTPException(status_code=404, detail=error_msg)
                
                # Этап 3: Загрузка данных
                logger.info(f"[GCS] Этап 3: Загрузка данных из bucket...")
                init_time = time.time()
                try:
                    from scale.gcs_integration import load_json_from_gcs_bucket
                    predictions_raw = load_json_from_gcs_bucket(
                        request.bucket_name,
                        prefix=request.prefix or "",
                        log_callback=log_progress
                    )
                    elapsed = time.time() - init_time
                    logger.info(f"[GCS] Этап 3 завершен за {elapsed:.2f}с. Загружено файлов: {len(predictions_raw)}")
                    if elapsed > 60.0:
                        logger.warning(f"[GCS] Этап 3 занял {elapsed:.2f}с - возможно медленная загрузка файлов")
                except Exception as e:
                    elapsed = time.time() - init_time
                    logger.error(f"[GCS] Этап 3 завершился с ошибкой после {elapsed:.2f}с: {e}", exc_info=True)
                    raise
                
                # Этап 4: Конвертация данных
                logger.info(f"[GCS] Этап 4: Конвертация данных в Prediction объекты...")
                init_time = time.time()
                predictions_converted = {}
                for name, data in predictions_raw.items():
                    predictions_converted[name] = domain.predictions_from_dict(data)
                logger.info(f"[GCS] Этап 4 завершен за {time.time() - init_time:.2f}с. Конвертировано: {len(predictions_converted)}")
                
                total_time = time.time() - start_time
                logger.info(f"[GCS] Полная загрузка завершена за {total_time:.2f}с. Всего файлов: {len(predictions_converted)}")
                
                # Очищаем прогресс после завершения
                if progress_key in _data_cache:
                    del _data_cache[progress_key]
            except HTTPException:
                raise
            except Exception as e:
                total_time = time.time() - start_time
                logger.error(f"[GCS] Ошибка после {total_time:.2f}с: {str(e)}", exc_info=True)
                # Очищаем прогресс при ошибке
                if progress_key in _data_cache:
                    del _data_cache[progress_key]
                raise HTTPException(status_code=500, detail=f"Error loading from GCS: {str(e)}")
        
        elif request.source == "experiment":
            if not request.experiment_name:
                raise HTTPException(status_code=400, detail="Experiment name required for experiment source")
            
            try:
                # Загружаем данные из эксперимента (как в dashboard)
                experiments_dir = Path(request.experiments_dir)
                experiment_dir = experiments_dir / request.experiment_name
                
                if not experiment_dir.exists():
                    raise HTTPException(status_code=404, detail=f"Experiment not found: {request.experiment_name}")
                
                # Ищем CSV файлы с данными (как в dashboard)
                aggregated_files = sorted(experiment_dir.glob("aggregated_data_*.csv"))
                relative_files = sorted(experiment_dir.glob("relative_features_*.csv"))
                all_features_files = sorted(experiment_dir.glob("all_features_*.csv"))
                
                if not (aggregated_files or relative_files or all_features_files):
                    raise HTTPException(status_code=404, detail=f"No data files found in experiment: {request.experiment_name}")
                
                # Загружаем данные из эксперимента
                # Используем all_features или relative_features (как в dashboard)
                if all_features_files:
                    df_from_experiment = pd.read_csv(all_features_files[-1])
                elif relative_files:
                    df_from_experiment = pd.read_csv(relative_files[-1])
                elif aggregated_files:
                    df_from_experiment = pd.read_csv(aggregated_files[-1])
                else:
                    raise HTTPException(status_code=404, detail="No data files found")
                
                # Сохраняем данные в кэш для дальнейшего использования
                # НЕ конвертируем в predictions, так как это уже агрегированные данные
                experiment_cache_key = f"experiment_{request.experiment_name}_df_features"
                experiment_df_cache_key = f"experiment_{request.experiment_name}_df"
                
                _data_cache[experiment_cache_key] = df_from_experiment.to_dict(orient="records")
                
                if aggregated_files:
                    df_aggregated = pd.read_csv(aggregated_files[-1])
                    _data_cache[experiment_df_cache_key] = df_aggregated.to_dict(orient="records")
                
                # Загружаем конфигурацию эксперимента (best_features_*.json)
                experiment_config = None
                best_features_files = sorted(experiment_dir.glob("best_features_*.json"))
                if best_features_files:
                    try:
                        with open(best_features_files[-1], 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        experiment_config = {
                            'selected_features': config.get('selected_features', []),
                            'method': config.get('method', 'unknown'),
                            'metrics': config.get('metrics', {}),
                            'timestamp': config.get('timestamp', ''),
                        }
                    except Exception as e:
                        logger.warning(f"Could not load experiment config: {e}")
                
                # Загружаем PCA данные (results.csv), если есть
                pca_data = None
                pca_cache_key = None
                results_files = sorted(experiment_dir.glob("results.csv"))
                if results_files:
                    try:
                        df_results = pd.read_csv(results_files[-1])
                        # Проверяем, что это действительно PCA данные (должны быть колонки PC1, PC1_norm или image)
                        if 'PC1' in df_results.columns or 'image' in df_results.columns:
                            pca_data = df_results.to_dict(orient="records")
                            # Сохраняем PCA данные в кэш
                            pca_cache_key = f"{experiment_cache_key}_pca"
                            _data_cache[pca_cache_key] = pca_data
                            logger.info(f"[EXPERIMENT] Загружены PCA данные: {len(pca_data)} записей из {results_files[-1]}")
                        else:
                            logger.warning(f"[EXPERIMENT] results.csv найден, но не содержит PCA данных (колонки: {list(df_results.columns)})")
                    except Exception as e:
                        logger.warning(f"Could not load PCA data: {e}", exc_info=True)
                else:
                    logger.info(f"[EXPERIMENT] results.csv не найден в эксперименте {request.experiment_name}")
                
                # Проверяем наличие analyzer (spectral_analyzer.pkl)
                has_analyzer = (experiment_dir / "spectral_analyzer.pkl").exists()
                
                # Возвращаем информацию о загруженных данных
                response_data = {
                    "status": "success",
                    "source": request.source,
                    "experiment_name": request.experiment_name,
                    "files_count": len(df_from_experiment),
                    "cache_key": experiment_cache_key,  # Для совместимости
                    "df_features_cache_key": experiment_cache_key,  # Основной ключ для features
                    "df_cache_key": experiment_df_cache_key if aggregated_files else None,
                    "has_aggregated": len(aggregated_files) > 0,
                    "has_features": len(relative_files) > 0 or len(all_features_files) > 0,
                    "sample_names": df_from_experiment['image'].tolist()[:10] if 'image' in df_from_experiment.columns else []
                }
                
                # Добавляем конфигурацию эксперимента
                if experiment_config:
                    response_data['experiment_config'] = experiment_config
                    response_data['selected_features'] = experiment_config.get('selected_features', [])
                    response_data['metrics'] = experiment_config.get('metrics', {})
                    response_data['method'] = experiment_config.get('method', 'unknown')
                
                # Добавляем информацию о PCA данных
                if pca_data:
                    response_data['has_pca'] = True
                    response_data['pca_cache_key'] = f"{experiment_cache_key}_pca"
                    response_data['pca_samples_count'] = len(pca_data)
                else:
                    response_data['has_pca'] = False
                
                response_data['has_analyzer'] = has_analyzer
                
                return response_data
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading from experiment: {str(e)}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source: {request.source}")
        
        # Для источников, отличных от experiment, обрабатываем predictions
        if request.source != "experiment":
            if not predictions_converted:
                raise HTTPException(status_code=404, detail="No predictions loaded")
            
            # Сохраняем в кэш
            cache_key = f"predictions_{request.source}_{request.path or request.bucket_name}"
            _data_cache[cache_key] = predictions_converted
            
            return {
                "status": "success",
                "source": request.source,
                "files_count": len(predictions_converted),
                "cache_key": cache_key,
                "sample_names": list(predictions_converted.keys())[:10]  # Первые 10 для примера
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in load_data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/aggregate")
async def aggregate_predictions(request: AggregateRequest):
    """
    Агрегирует загруженные predictions и создает DataFrame с признаками.
    
    Требует cache_key из /api/v1/load-data.
    Если данные из эксперимента, использует уже агрегированные данные.
    """
    cache_key = request.cache_key
    try:
        # Проверяем, это данные из эксперимента (уже агрегированные)
        if cache_key.startswith("experiment_") and cache_key.endswith("_df_features"):
            if cache_key not in _data_cache:
                raise HTTPException(status_code=404, detail=f"Experiment data not found: {cache_key}")
            
            # Данные из эксперимента уже агрегированы, используем их напрямую
            df_features_data = _data_cache[cache_key]
            df_features = pd.DataFrame(df_features_data)
            
            # Получаем aggregated data, если есть
            df_cache_key = cache_key.replace("_df_features", "_df")
            df = None
            if df_cache_key in _data_cache:
                df_data = _data_cache[df_cache_key]
                df = pd.DataFrame(df_data)
            else:
                # Если нет aggregated, создаем из features (обратная операция не точная, но для совместимости)
                df = df_features.copy()
            
            # Создаем all_features
            df_all_features = aggregate.select_all_feature_columns(df_features)
            
            # Сохраняем в кэш с правильными ключами
            df_all_features_cache_key = f"{cache_key.replace('_df_features', '')}_df_all"
            _data_cache[df_all_features_cache_key] = df_all_features.to_dict(orient="records")
            
            return {
                "status": "success",
                "aggregated_rows": len(df_features),
                "features_count": len(df_features.columns) - 1,  # -1 для колонки 'image'
                "df_cache_key": df_cache_key if df_cache_key in _data_cache else None,
                "df_features_cache_key": cache_key,
                "df_all_features_cache_key": df_all_features_cache_key,
                "feature_columns": [col for col in df_features.columns if col != 'image'],
                "from_experiment": True
            }
        
        # Обычная агрегация из predictions
        if cache_key not in _data_cache:
            raise HTTPException(status_code=404, detail=f"Data not found. Load data first using cache_key: {cache_key}")
        
        predictions = _data_cache[cache_key]
        
        # Агрегация данных
        rows = []
        for image_name, preds in predictions.items():
            pred_stats = aggregate.aggregate_predictions_from_dict(preds, image_name)
            rows.append(pred_stats)
        
        df = pd.DataFrame(rows)
        
        # Создаем относительные признаки
        df_features = aggregate.create_relative_features(df)
        
        # Создаем all_features (все доступные признаки)
        df_all_features = aggregate.select_all_feature_columns(df_features)
        
        # Сохраняем в кэш
        df_cache_key = f"{cache_key}_df"
        df_features_cache_key = f"{cache_key}_df_features"
        df_all_features_cache_key = f"{cache_key}_df_all"
        _data_cache[df_cache_key] = df.to_dict(orient="records")
        _data_cache[df_features_cache_key] = df_features.to_dict(orient="records")
        _data_cache[df_all_features_cache_key] = df_all_features.to_dict(orient="records")
        
        return {
            "status": "success",
            "aggregated_rows": len(df),
            "features_count": len(df_features.columns) - 1,  # -1 для колонки 'image'
            "df_cache_key": df_cache_key,
            "df_features_cache_key": df_features_cache_key,
            "df_all_features_cache_key": df_all_features_cache_key,
            "feature_columns": [col for col in df_features.columns if col != 'image']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in aggregate_predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/evaluate-features")
async def evaluate_features(request: FeatureEvaluationRequest):
    """
    Оценивает качество набора признаков для разделения mod и normal образцов.
    
    Возвращает метрики: score, separation, mean_pc1_norm_mod, explained_variance
    """
    try:
        df_features_cache_key = request.df_features_cache_key
        if df_features_cache_key not in _data_cache:
            raise HTTPException(status_code=404, detail=f"Features not found. Run aggregate first.")
        
        df_features_data = _data_cache[df_features_cache_key]
        df_features = pd.DataFrame(df_features_data)
        
        # Определяем mod и normal образцы автоматически, если не указаны
        if not request.mod_samples or not request.normal_samples:
            mod_samples = []
            normal_samples = []
            
            if "image" in df_features.columns:
                for img_name in df_features["image"].unique():
                    sample_type = identify_sample_type(str(img_name))
                    if sample_type == 'mod':
                        mod_samples.append(img_name)
                    elif sample_type == 'normal':
                        normal_samples.append(img_name)
        else:
            mod_samples = request.mod_samples
            normal_samples = request.normal_samples
        
        if not mod_samples or not normal_samples:
            raise HTTPException(
                status_code=400,
                detail="No mod or normal samples found. Specify mod_samples and normal_samples explicitly."
            )
        
        # Оцениваем признаки
        metrics = evaluate_feature_set(
            df_features,
            request.feature_columns,
            mod_samples,
            normal_samples
        )
        
        return {
            "status": "success",
            "metrics": metrics,
            "mod_samples_count": len(mod_samples),
            "normal_samples_count": len(normal_samples),
            "features_count": len(request.feature_columns)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluate_features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/pca-score")
async def pca_score(request: PCAScoreRequest):
    """
    Вычисляет PCA score для образцов на основе выбранных признаков.
    
    Возвращает PC1 значения для каждого образца.
    """
    try:
        df_features_cache_key = request.df_features_cache_key
        if df_features_cache_key not in _data_cache:
            raise HTTPException(status_code=404, detail=f"Features not found. Run aggregate first.")
        
        df_features_data = _data_cache[df_features_cache_key]
        df_features = pd.DataFrame(df_features_data)
        
        # Выбираем нужные признаки
        if request.use_relative_features:
            df_for_pca = df_features
        else:
            # Используем абсолютные признаки (нужно загрузить df, а не df_features)
            df_cache_key = df_features_cache_key.replace("_df_features", "_df")
            if df_cache_key not in _data_cache:
                raise HTTPException(status_code=404, detail="Absolute features not found. Use use_relative_features=true")
            df_data = _data_cache[df_cache_key]
            df_for_pca = pd.DataFrame(df_data)
        
        # Проверяем наличие признаков
        missing_features = [f for f in request.feature_columns if f not in df_for_pca.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Features not found: {missing_features}"
            )
        
        # Вычисляем PCA score используя те же методы, что и dashboard
        # Используем fit_transform для обучения и преобразования данных
        pca_scorer = pca_scoring.PCAScorer()
        df_result = pca_scorer.fit_transform(
            df_for_pca,
            feature_columns=request.feature_columns
        )
        
        # Возвращаем результаты
        # Проверяем наличие колонки 'image' (она может отсутствовать)
        if 'image' in df_result.columns:
            results = df_result[['image', 'PC1', 'PC1_norm']].to_dict(orient="records")
        else:
            # Если нет колонки 'image', используем индекс
            df_result['image'] = df_result.index.astype(str)
            results = df_result[['image', 'PC1', 'PC1_norm']].to_dict(orient="records")
        
        return {
            "status": "success",
            "results": results,
            "samples_count": len(results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pca_score: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/create-spectrum")
async def create_spectrum(request: CreateSpectrumRequest):
    """
    Создает спектр на основе PCA данных (алиас для spectral-analysis).
    Используется для совместимости с фронтендом.
    """
    # Преобразуем CreateSpectrumRequest в SpectralAnalysisRequest
    spectral_request = SpectralAnalysisRequest(
        df_features_cache_key=request.df_features_cache_key,
        feature_columns=request.feature_columns,
        percentile_low=request.percentile_low,
        percentile_high=request.percentile_high,
        use_relative_features=request.use_relative_features,
        use_gmm_classification=request.use_gmm_classification
    )
    # Вызываем spectral_analysis_endpoint
    return await spectral_analysis_endpoint(spectral_request)


@app.post("/api/v1/spectral-analysis")
async def spectral_analysis_endpoint(request: SpectralAnalysisRequest):
    """
    Выполняет спектральный анализ данных.
    
    Возвращает моды, KDE, GMM и другие метрики спектрального анализа.
    """
    try:
        df_features_cache_key = request.df_features_cache_key
        if df_features_cache_key not in _data_cache:
            raise HTTPException(status_code=404, detail=f"Features not found. Run aggregate first.")
        
        df_features_data = _data_cache[df_features_cache_key]
        df_features = pd.DataFrame(df_features_data)
        
        # Выбираем нужные признаки
        if request.use_relative_features:
            df_for_analysis = df_features
        else:
            df_cache_key = df_features_cache_key.replace("_df_features", "_df")
            if df_cache_key not in _data_cache:
                raise HTTPException(status_code=404, detail="Absolute features not found")
            df_data = _data_cache[df_cache_key]
            df_for_analysis = pd.DataFrame(df_data)
        
        # Проверяем наличие признаков
        missing_features = [f for f in request.feature_columns if f not in df_for_analysis.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Features not found: {missing_features}"
            )
        
        # Выполняем спектральный анализ (как в dashboard)
        analyzer = spectral_analysis.SpectralAnalyzer()
        
        # 1. Обучаем PCA
        analyzer.fit_pca(df_for_analysis, request.feature_columns)
        
        # 2. Преобразуем данные через PCA
        df_pca = analyzer.transform_pca(df_for_analysis)
        
        # 3. Анализируем спектр
        analyzer.fit_spectrum(
            df_pca,
            percentile_low=request.percentile_low,
            percentile_high=request.percentile_high
        )
        
        # 4. Обучаем GMM, если требуется
        if request.use_gmm_classification:
            analyzer.fit_gmm(df_pca)
        
        # 5. Преобразуем в спектральную шкалу (как в dashboard)
        df_spectrum = analyzer.transform_to_spectrum(
            df_pca,
            use_gmm_classification=request.use_gmm_classification
        )
        
        # Сохраняем analyzer и spectrum в кэш для сохранения эксперимента
        spectrum_cache_key = f"{df_features_cache_key}_spectrum"
        analyzer_cache_key = f"{df_features_cache_key}_analyzer"
        _data_cache[spectrum_cache_key] = df_spectrum.to_dict(orient="records")
        _data_cache[analyzer_cache_key] = analyzer  # Сохраняем analyzer для сохранения эксперимента
        
        # Получаем результаты
        modes = analyzer.get_modes()
        kde_data = analyzer.get_kde_data()
        gmm_data = analyzer.get_gmm_data()
        
        return {
            "status": "success",
            "modes": modes,
            "kde": kde_data,
            "gmm": gmm_data,
            "spectrum_data": df_spectrum.to_dict(orient="records"),
            "percentiles": analyzer.pc1_percentiles,
            "samples_count": len(df_pca),
            "spectrum_cache_key": spectrum_cache_key,
            "analyzer_cache_key": analyzer_cache_key
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in spectral_analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/experiments")
async def list_experiments():
    """
    Возвращает список доступных экспериментов.
    """
    try:
        from scale import dashboard_experiment_selector
        
        experiments = dashboard_experiment_selector.list_available_experiments(
            use_tracker=True,
            top_n=None,
            check_data=True
        )
        
        return {
            "status": "success",
            "experiments": experiments,
            "count": len(experiments)
        }
    
    except Exception as e:
        logger.error(f"Error in list_experiments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/load-experiment")
async def load_experiment(request: LoadExperimentRequest):
    """
    Загружает данные из эксперимента (аналог load_data с source='experiment').
    Используется для совместимости с существующим фронтендом.
    """
    try:
        # Используем ту же логику, что и в load_data с source='experiment'
        experiments_dir = Path(request.experiments_dir)
        experiment_dir = experiments_dir / request.experiment_name
        
        if not experiment_dir.exists():
            raise HTTPException(status_code=404, detail=f"Experiment not found: {request.experiment_name}")
        
        # Ищем CSV файлы с данными
        aggregated_files = sorted(experiment_dir.glob("aggregated_data_*.csv"))
        relative_files = sorted(experiment_dir.glob("relative_features_*.csv"))
        all_features_files = sorted(experiment_dir.glob("all_features_*.csv"))
        
        if not (aggregated_files or relative_files or all_features_files):
            raise HTTPException(status_code=404, detail=f"No data files found in experiment: {request.experiment_name}")
        
        # Загружаем данные из эксперимента
        if all_features_files:
            df_from_experiment = pd.read_csv(all_features_files[-1])
        elif relative_files:
            df_from_experiment = pd.read_csv(relative_files[-1])
        elif aggregated_files:
            df_from_experiment = pd.read_csv(aggregated_files[-1])
        else:
            raise HTTPException(status_code=404, detail="No data files found")
        
        # Сохраняем данные в кэш
        experiment_cache_key = f"experiment_{request.experiment_name}_df_features"
        experiment_df_cache_key = f"experiment_{request.experiment_name}_df"
        
        _data_cache[experiment_cache_key] = df_from_experiment.to_dict(orient="records")
        
        if aggregated_files:
            df_aggregated = pd.read_csv(aggregated_files[-1])
            _data_cache[experiment_df_cache_key] = df_aggregated.to_dict(orient="records")
        
        # Загружаем конфигурацию эксперимента
        experiment_config = None
        best_features_files = sorted(experiment_dir.glob("best_features_*.json"))
        if best_features_files:
            try:
                with open(best_features_files[-1], 'r', encoding='utf-8') as f:
                    config = json.load(f)
                experiment_config = {
                    'selected_features': config.get('selected_features', []),
                    'method': config.get('method', 'unknown'),
                    'metrics': config.get('metrics', {}),
                    'timestamp': config.get('timestamp', ''),
                }
            except Exception as e:
                logger.warning(f"Could not load experiment config: {e}")
        
        # Загружаем PCA данные (results.csv), если есть
        pca_data = None
        pca_cache_key = None
        results_files = sorted(experiment_dir.glob("results.csv"))
        if results_files:
            try:
                df_results = pd.read_csv(results_files[-1])
                # Проверяем, что это действительно PCA данные (должны быть колонки PC1, PC1_norm или image)
                if 'PC1' in df_results.columns or 'image' in df_results.columns:
                    pca_data = df_results.to_dict(orient="records")
                    # Сохраняем PCA данные в кэш
                    pca_cache_key = f"{experiment_cache_key}_pca"
                    _data_cache[pca_cache_key] = pca_data
                    logger.info(f"[EXPERIMENT] Загружены PCA данные: {len(pca_data)} записей из {results_files[-1]}")
                else:
                    logger.warning(f"[EXPERIMENT] results.csv найден, но не содержит PCA данных (колонки: {list(df_results.columns)})")
            except Exception as e:
                logger.warning(f"Could not load PCA data: {e}", exc_info=True)
        else:
            logger.info(f"[EXPERIMENT] results.csv не найден в эксперименте {request.experiment_name}")
        
        # Формируем ответ в формате, ожидаемом фронтендом
        response_data = {
            "status": "success",
            "experiment_name": request.experiment_name,
            "n_features": len(experiment_config.get('selected_features', [])) if experiment_config else 0,
            "method": experiment_config.get('method', 'unknown') if experiment_config else 'unknown',
            "metrics": experiment_config.get('metrics', {}) if experiment_config else {},
            "features": experiment_config.get('selected_features', []) if experiment_config else [],
            "pca_data": pca_data[:10] if pca_data else [],  # Первые 10 для отображения
            "has_pca": pca_data is not None and len(pca_data) > 0,
            "pca_cache_key": pca_cache_key,
            "pca_samples_count": len(pca_data) if pca_data else 0,
            "cache_key": experiment_cache_key,
            "df_features_cache_key": experiment_cache_key,
        }
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in load_experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading experiment: {str(e)}")


@app.get("/api/v1/load-progress")
async def get_load_progress():
    """
    Возвращает прогресс последней активной загрузки данных.
    """
    progress_key = "load_progress_latest"
    if progress_key in _data_cache:
        return {
            "status": "loading",
            **(_data_cache[progress_key])
        }
    return {
        "status": "completed",
        "current": 0,
        "total": 0,
        "progress": 1.0,
        "message": "Загрузка завершена"
    }


@app.get("/api/v1/download-csv")
async def download_csv(cache_key: str, filename: Optional[str] = None):
    """
    Скачивает CSV файл из кэша.
    
    Args:
        cache_key: Cache key для данных (например, df_features_cache_key)
        filename: Имя файла для скачивания (опционально)
    """
    try:
        if cache_key not in _data_cache:
            raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")
        
        df_data = _data_cache[cache_key]
        df = pd.DataFrame(df_data)
        
        # Создаем CSV в памяти
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        csv_bytes = csv_content.encode('utf-8')
        
        # Определяем имя файла
        if not filename:
            filename = f"{cache_key}.csv"
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download_csv: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/save-experiment")
async def save_experiment(request: SaveExperimentRequest):
    """
    Сохраняет эксперимент в формате dashboard со всеми компонентами.
    
    Сохраняет:
    - aggregated_data_{timestamp}.csv
    - relative_features_{timestamp}.csv (или all_features)
    - results.csv (если есть спектр)
    - spectral_analyzer.pkl (если есть спектр)
    - best_features_{timestamp}.json
    - metadata.json
    """
    try:
        from scale.dashboard import create_experiment_dir, save_experiment as dashboard_save_experiment
        from scale import aggregate
        
        # Проверяем наличие данных
        if request.df_features_cache_key not in _data_cache:
            raise HTTPException(status_code=404, detail=f"Features not found: {request.df_features_cache_key}")
        
        df_features_data = _data_cache[request.df_features_cache_key]
        df_features = pd.DataFrame(df_features_data)
        
        # Получаем aggregated data
        df_cache_key = request.df_features_cache_key.replace("_df_features", "_df")
        if df_cache_key not in _data_cache:
            raise HTTPException(status_code=404, detail="Aggregated data not found. Run aggregate first.")
        
        df_aggregated_data = _data_cache[df_cache_key]
        df_aggregated = pd.DataFrame(df_aggregated_data)
        
        # Создаем директорию эксперимента
        experiments_dir = Path(request.experiments_dir)
        exp_dir = create_experiment_dir(experiments_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем CSV файлы (как в export_complete_results)
        aggregated_path = exp_dir / f"aggregated_data_{timestamp}.csv"
        df_aggregated.to_csv(aggregated_path, index=False)
        
        features_path = exp_dir / f"relative_features_{timestamp}.csv" if request.use_relative_features else exp_dir / f"all_features_{timestamp}.csv"
        df_features.to_csv(features_path, index=False)
        
        # Получаем all_features (если есть)
        df_all_features = None
        all_features_cache_key = request.df_features_cache_key.replace("_df_features", "_df_all")
        if all_features_cache_key in _data_cache:
            df_all_features_data = _data_cache[all_features_cache_key]
            df_all_features = pd.DataFrame(df_all_features_data)
            all_features_path = exp_dir / f"all_features_{timestamp}.csv"
            df_all_features.to_csv(all_features_path, index=False)
        
        # Проверяем наличие спектра для сохранения
        analyzer = None
        df_results = None
        spectrum_cache_key = f"{request.df_features_cache_key}_spectrum"
        analyzer_cache_key = f"{request.df_features_cache_key}_analyzer"
        
        if spectrum_cache_key in _data_cache:
            df_spectrum_data = _data_cache[spectrum_cache_key]
            df_results = pd.DataFrame(df_spectrum_data)
            
            # Получаем analyzer из кэша
            if analyzer_cache_key in _data_cache:
                analyzer = _data_cache[analyzer_cache_key]
        
        # Подготавливаем метаданные
        metadata = {
            "method": request.method,
            "use_relative_features": request.use_relative_features,
            "n_features": len(request.feature_columns),
            "selected_features": request.feature_columns,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(df_features)
        }
        
        if request.metrics:
            metadata["metrics"] = request.metrics
        
        # Сохраняем эксперимент через dashboard функцию
        if df_results is not None and analyzer is not None:
            dashboard_save_experiment(
                exp_dir,
                df_results,
                analyzer=analyzer,
                metadata=metadata,
                selected_features=request.feature_columns,
                metrics=request.metrics,
                use_relative_features=request.use_relative_features
            )
        else:
            # Если нет спектра, сохраняем только метаданные и конфигурацию
            metadata_path = exp_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Сохраняем best_features JSON
            if request.feature_columns and request.metrics:
                best_features_path = exp_dir / f"best_features_{timestamp}.json"
                config = {
                    'method': request.method,
                    'selected_features': request.feature_columns,
                    'metrics': {
                        'score': float(request.metrics.get('score', 0)),
                        'separation': float(request.metrics.get('separation', 0)),
                        'mean_pc1_norm_mod': float(request.metrics.get('mean_pc1_norm_mod', 0)),
                        'explained_variance': float(request.metrics.get('explained_variance', 0)),
                    },
                    'timestamp': timestamp,
                    'use_relative_features': request.use_relative_features,
                }
                with open(best_features_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Регистрируем в трекере
        try:
            from model_development.experiment_tracker import ExperimentTracker, register_experiment_from_directory
            tracker = ExperimentTracker(experiments_dir)
            exp_id = register_experiment_from_directory(
                experiment_dir=exp_dir,
                tracker=tracker,
                train_set=metadata.get("train_set", "results/predictions"),
                aggregation_version=metadata.get("aggregation_version", "current"),
            )
        except Exception as e:
            logger.warning(f"Could not register experiment in tracker: {e}")
        
        return {
            "status": "success",
            "experiment_name": request.experiment_name,
            "experiment_dir": str(exp_dir),
            "experiment_path": str(exp_dir),
            "files_saved": {
                "aggregated_data": str(aggregated_path),
                "features": str(features_path),
                "all_features": str(all_features_path) if df_all_features is not None else None,
                "results": str(exp_dir / "results.csv") if df_results is not None else None,
                "spectral_analyzer": str(exp_dir / "spectral_analyzer.pkl") if analyzer is not None else None,
                "best_features": str(exp_dir / f"best_features_{timestamp}.json") if request.feature_columns else None,
                "metadata": str(exp_dir / "metadata.json")
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in save_experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    # Увеличиваем таймауты для длительных операций загрузки
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=300,  # 5 минут для keep-alive
        timeout_graceful_shutdown=300  # 5 минут для graceful shutdown
    )

