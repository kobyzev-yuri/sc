"""
Интеграция с Google Cloud Storage для загрузки JSON файлов.

Поддерживает:
- Загрузку файлов из GCS bucket
- Автоматическую авторизацию через GCP credentials
- Логирование процесса загрузки
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from google.cloud import storage
    from google.oauth2 import service_account
    import os
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    service_account = None


def is_gcs_available() -> bool:
    """Проверяет, доступна ли интеграция с Google Cloud Storage."""
    return GCS_AVAILABLE


def _get_gcs_client(log_callback: Optional[callable] = None):
    """
    Создает и возвращает GCS клиент с автоматическим поиском service account key.
    
    Args:
        log_callback: Функция для логирования (message)
        
    Returns:
        storage.Client или None при ошибке
    """
    if not GCS_AVAILABLE:
        return None
    
    # Пытаемся найти service account key
    service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    # Если переменная не установлена, ищем в стандартных местах
    if not service_account_path or not os.path.exists(service_account_path):
        # Получаем абсолютный путь к текущей рабочей директории (для Docker это /app)
        current_dir = os.getcwd()
        possible_paths = [
            os.path.join(current_dir, '.config', 'gcs', 'service-account-key.json'),  # Для Docker образа (/app/.config/gcs/)
            os.path.join('.config', 'gcs', 'service-account-key.json'),  # Относительный путь
            os.path.join(os.path.expanduser('~'), '.config', 'gcs', 'service-account-key.json'),
            '/mnt/ai/cnn/.config/gcs/service-account-key.json',
            os.path.join(os.path.expanduser('~'), 'service-account-key.json'),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                service_account_path = abs_path
                break
    
    # Используем service account key если найден
    if service_account_path and os.path.exists(service_account_path):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            return storage.Client(credentials=credentials, project=credentials.project_id)
        except Exception as e:
            if log_callback:
                log_callback(f"⚠️ Не удалось использовать service account из {service_account_path}: {e}")
            # Продолжаем с дефолтными credentials
            try:
                return storage.Client()
            except Exception:
                return None
    else:
        # Используем Application Default Credentials
        try:
            return storage.Client()
        except Exception as e:
            if log_callback:
                log_callback(f"⚠️ Не удалось создать GCS клиент: {e}")
            return None


def list_files_from_gcs_bucket(
    bucket_name: str,
    prefix: str = "",
    file_type: Optional[str] = None,
    log_callback: Optional[callable] = None
) -> List[Dict]:
    """
    Получает список файлов из GCS bucket.
    
    Args:
        bucket_name: Имя GCS bucket
        prefix: Префикс пути для поиска файлов (например, 'data/predictions/')
        file_type: Фильтр по типу файла (например, 'json')
        log_callback: Функция для логирования (message)
        
    Returns:
        Список словарей с информацией о файлах:
        [{'name': 'file.json', 'size': 1234, 'updated': '2024-01-01', ...}]
    """
    if not GCS_AVAILABLE:
        log_msg = "❌ Google Cloud Storage API недоступен"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        return []
    
    try:
        log_msg = f"📂 Подключение к GCS bucket: {bucket_name}"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        
        # Получаем GCS клиент с автоматическим поиском credentials
        client = _get_gcs_client(log_callback)
        if not client:
            raise Exception("Не удалось создать GCS клиент. Проверьте авторизацию.")
        
        bucket = client.bucket(bucket_name)
        
        # Проверяем существование bucket
        if not bucket.exists():
            error_msg = f"❌ Bucket '{bucket_name}' не существует"
            if log_callback:
                log_callback(error_msg)
            else:
                print(error_msg)
            return []
        
        log_msg = f"🔍 Поиск файлов с префиксом: '{prefix}'"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        
        # Получаем список файлов
        blobs = bucket.list_blobs(prefix=prefix)
        
        files = []
        for blob in blobs:
            # Фильтр по типу файла
            if file_type and not blob.name.endswith(f'.{file_type}'):
                continue
            
            # Пропускаем директории (заканчиваются на /)
            if blob.name.endswith('/'):
                continue
            
            file_info = {
                'name': blob.name,
                'size': blob.size,
                'updated': blob.updated.isoformat() if blob.updated else None,
                'content_type': blob.content_type,
            }
            files.append(file_info)
        
        log_msg = f"✅ Найдено файлов: {len(files)}"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        
        # Логируем имена файлов
        if files and log_callback:
            for idx, file_info in enumerate(files[:10], 1):  # Показываем первые 10
                size_mb = file_info['size'] / (1024 * 1024) if file_info['size'] else 0
                log_callback(f"  {idx}. {file_info['name']} ({size_mb:.2f} MB)")
            if len(files) > 10:
                log_callback(f"  ... и еще {len(files) - 10} файлов")
        
        return files
    
    except Exception as e:
        error_msg = str(e)
        if "credentials" in error_msg.lower() or "authentication" in error_msg.lower() or "adc" in error_msg.lower():
            full_error = f"❌ Ошибка авторизации GCS: {error_msg}"
            if log_callback:
                log_callback(full_error)
                log_callback("")
                log_callback("🔧 **Как исправить:**")
                log_callback("")
                log_callback("**Вариант 1 (для локальной разработки):**")
                log_callback("  Выполните в терминале:")
                log_callback("  gcloud auth application-default login")
                log_callback("")
                log_callback("**Вариант 2 (для Cloud Run / Service Account):**")
                log_callback("  Установите переменную окружения:")
                log_callback("  export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/service-account-key.json\"")
                log_callback("")
                log_callback("**Вариант 3 (если используете gcloud):**")
                log_callback("  gcloud auth login")
                log_callback("  gcloud config set project YOUR_PROJECT_ID")
            else:
                print(full_error)
                print("\n🔧 Как исправить:")
                print("Выполните: gcloud auth application-default login")
        else:
            error_msg = f"❌ Ошибка при получении списка файлов из GCS: {e}"
            if log_callback:
                log_callback(error_msg)
            else:
                print(error_msg)
        return []


def download_file_from_gcs(
    bucket_name: str,
    blob_name: str,
    log_callback: Optional[callable] = None
) -> Optional[str]:
    """
    Скачивает файл из GCS bucket.
    
    Args:
        bucket_name: Имя GCS bucket
        blob_name: Имя файла (путь) в bucket
        log_callback: Функция для логирования (message)
        
    Returns:
        Содержимое файла как строка или None при ошибке
    """
    if not GCS_AVAILABLE:
        return None
    
    try:
        file_size_mb = 0
        try:
            client = _get_gcs_client(log_callback)
            if not client:
                if log_callback:
                    log_callback("❌ Не удалось создать GCS клиент")
                return None
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Получаем размер файла для логирования
            blob.reload()
            file_size_mb = blob.size / (1024 * 1024) if blob.size else 0
            
            log_msg = f"⬇️  Скачивание: {blob_name}"
            if file_size_mb > 0:
                log_msg += f" ({file_size_mb:.2f} MB)"
            if log_callback:
                log_callback(log_msg)
            else:
                print(log_msg)
            
            # Скачиваем файл с таймаутом
            import time
            FILE_DOWNLOAD_TIMEOUT = 60  # 60 секунд на файл для Cloud Run
            
            start_time = time.time()
            
            # Используем download_as_bytes() для лучшего контроля таймаута
            try:
                # Проверяем размер файла перед загрузкой
                blob.reload()
                file_size_mb = blob.size / (1024 * 1024) if blob.size else 0
                
                # Для больших файлов (>10MB) увеличиваем таймаут
                if file_size_mb > 10:
                    FILE_DOWNLOAD_TIMEOUT = min(120, int(file_size_mb * 2))  # До 2 секунд на MB, максимум 120 секунд
                
                # Загружаем файл
                content_bytes = blob.download_as_bytes()
                
                elapsed = time.time() - start_time
                if elapsed > FILE_DOWNLOAD_TIMEOUT:
                    raise TimeoutError(f"Загрузка файла {blob_name} превысила таймаут {FILE_DOWNLOAD_TIMEOUT} секунд")
                
                content = content_bytes.decode('utf-8')
                log_msg = f"✅ Загружен: {blob_name} ({len(content)} символов, {elapsed:.1f}s)"
                if log_callback:
                    log_callback(log_msg)
                else:
                    print(log_msg)
                
                return content
            except TimeoutError:
                raise
            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed > FILE_DOWNLOAD_TIMEOUT:
                    raise TimeoutError(f"Загрузка файла {blob_name} превысила таймаут {FILE_DOWNLOAD_TIMEOUT} секунд: {e}")
                raise
        
        except TimeoutError as e:
            error_msg = f"⏱️ Таймаут при скачивании файла {blob_name}: {e}"
            if log_callback:
                log_callback(error_msg)
            else:
                print(error_msg)
            return None
        except Exception as e:
            error_msg = f"❌ Ошибка при скачивании файла {blob_name}: {e}"
            if log_callback:
                log_callback(error_msg)
            else:
                print(error_msg)
            return None
    
    except Exception as e:
        error_msg = f"❌ Неожиданная ошибка при скачивании из GCS: {e}"
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        return None


def load_json_from_gcs_bucket(
    bucket_name: str,
    prefix: str = "",
    log_callback: Optional[callable] = None
) -> Dict[str, dict]:
    """
    Загружает все JSON файлы из GCS bucket.
    
    Args:
        bucket_name: Имя GCS bucket
        prefix: Префикс пути для поиска файлов (например, 'data/predictions/')
        log_callback: Функция для логирования (message)
        
    Returns:
        Словарь {имя_файла: данные_json}
    """
    log_msg = f"🚀 Начало загрузки файлов из GCS bucket: {bucket_name}"
    if log_callback:
        log_callback(log_msg)
    else:
        print(log_msg)
    
    # Получаем список файлов
    files = list_files_from_gcs_bucket(
        bucket_name,
        prefix=prefix,
        file_type='json',
        log_callback=log_callback
    )
    
    if not files:
        log_msg = "⚠️ JSON файлы не найдены в указанном bucket/prefix"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        return {}
    
    # Загружаем данные из каждого файла
    # КРИТИЧНО: Восстанавливаем частично загруженные данные из session state (если есть)
    # Это позволяет продолжить загрузку после rerun
    predictions = {}
    
    # КРИТИЧНО: Проверяем, есть ли частично загруженные данные в session state
    # Это позволяет продолжить загрузку после rerun
    try:
        import streamlit as st
        partial_predictions_key = f"gcs_partial_predictions_{bucket_name}_{prefix}"
        if hasattr(st, 'session_state') and partial_predictions_key in st.session_state:
            predictions = st.session_state[partial_predictions_key].copy()
            loaded_count = len(predictions)
            log_msg = f"📥 Продолжаю загрузку: уже загружено {loaded_count} из {len(files)} файлов..."
            if log_callback:
                log_callback(log_msg)
            else:
                print(log_msg)
    except Exception:
        pass
    
    total_files = len(files)
    if len(predictions) == 0:
        log_msg = f"📥 Начинаю загрузку {total_files} JSON файлов..."
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
    
    # Определяем, какие файлы уже загружены
    loaded_file_names = set(predictions.keys())
    
    for idx, file_info in enumerate(files, 1):
        blob_name = file_info['name']
        file_name = Path(blob_name).stem  # Без расширения
        
        # Пропускаем уже загруженные файлы
        if file_name in loaded_file_names:
            log_msg = f"⏭️  [{idx}/{total_files}] Пропущен (уже загружен): {blob_name}"
            if log_callback:
                log_callback(log_msg)
            continue
        
        log_msg = f"📄 [{idx}/{total_files}] Обработка файла: {blob_name}"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        
        # Скачиваем файл
        content = download_file_from_gcs(
            bucket_name,
            blob_name,
            log_callback=log_callback
        )
        
        if content:
            try:
                data = json.loads(content)
                predictions[file_name] = data
                log_msg = f"✅ [{idx}/{total_files}] Успешно загружен и распарсен: {blob_name}"
                if log_callback:
                    log_callback(log_msg)
                else:
                    print(log_msg)
                
                # КРИТИЧНО: Сохраняем частично загруженные данные в session state после каждого файла
                # Это позволяет сохранить прогресс при rerun
                try:
                    import streamlit as st
                    if hasattr(st, 'session_state'):
                        partial_predictions_key = f"gcs_partial_predictions_{bucket_name}_{prefix}"
                        st.session_state[partial_predictions_key] = predictions.copy()
                except Exception:
                    pass  # Если streamlit недоступен, просто продолжаем
            except json.JSONDecodeError as e:
                error_msg = f"❌ [{idx}/{total_files}] Ошибка при парсинге JSON из {blob_name}: {e}"
                if log_callback:
                    log_callback(error_msg)
                else:
                    print(error_msg)
        else:
            error_msg = f"❌ [{idx}/{total_files}] Не удалось скачать файл: {blob_name}"
            if log_callback:
                log_callback(error_msg)
            else:
                print(error_msg)
    
    log_msg = f"🎉 Загрузка завершена! Успешно загружено {len(predictions)} из {total_files} файлов"
    if log_callback:
        log_callback(log_msg)
    else:
        print(log_msg)
    
    # КРИТИЧНО: Очищаем ключ частичной загрузки после успешной загрузки всех файлов
    try:
        import streamlit as st
        if hasattr(st, 'session_state'):
            partial_predictions_key = f"gcs_partial_predictions_{bucket_name}_{prefix}"
            if len(predictions) == total_files and partial_predictions_key in st.session_state:
                del st.session_state[partial_predictions_key]
    except Exception:
        pass
    
    return predictions

