"""
Интеграция с Google Drive для загрузки и сохранения JSON файлов.

Поддерживает:
- Загрузку файлов из расшаренных папок Google Drive
- Сохранение файлов в Google Drive
- Автоматическую авторизацию через OAuth 2.0
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING
import io

if TYPE_CHECKING:
    from google_auth_oauthlib.flow import Flow

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    from googleapiclient.errors import HttpError
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    # Flow будет None если импорт не удался, но аннотация типа использует строку "Flow"


# OAuth 2.0 Scopes для Google Drive
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',  # Чтение файлов
    'https://www.googleapis.com/auth/drive.file',      # Запись файлов (только созданные приложением)
]

# Redirect URI для OAuth (должен совпадать с настройками в Google Cloud Console)
REDIRECT_URI = 'http://localhost:8080'  # Для Cloud Run нужно будет изменить


def extract_folder_id_from_url(url: str) -> Optional[str]:
    """
    Извлекает Folder ID из URL Google Drive.
    
    Поддерживает форматы:
    - https://drive.google.com/drive/folders/FOLDER_ID
    - https://drive.google.com/open?id=FOLDER_ID
    - FOLDER_ID (если передан напрямую)
    
    Args:
        url: URL или ID папки Google Drive
        
    Returns:
        Folder ID или None, если не удалось извлечь
    """
    if not url:
        return None
    
    # Если это уже ID (нет слэшей и точек)
    if '/' not in url and '.' not in url:
        return url
    
    # Паттерны для извлечения ID
    patterns = [
        r'/folders/([a-zA-Z0-9_-]+)',
        r'[?&]id=([a-zA-Z0-9_-]+)',
        r'([a-zA-Z0-9_-]{25,})',  # Fallback: любой длинный ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def get_credentials(credentials_path: Optional[str] = None, token_path: Optional[str] = None):
    """
    Получает или создает OAuth credentials для Google Drive API.
    
    Args:
        credentials_path: Путь к файлу credentials.json (OAuth client config)
        token_path: Путь для сохранения токена (опционально)
        
    Returns:
        Credentials объект или None, если авторизация не удалась
    """
    if not GDRIVE_AVAILABLE:
        return None
    
    creds = None
    token_file = Path(token_path) if token_path else Path('.gdrive_token.json')
    
    # Загружаем сохраненный токен, если есть
    if token_file.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        except Exception:
            pass
    
    # Если токена нет или он истек - запрашиваем новый
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Обновляем истекший токен
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        
        if not creds:
            # Нужна авторизация через OAuth flow
            # Это должно быть сделано через веб-интерфейс
            return None
    
    # Сохраняем токен для следующего использования
    if token_file and creds:
        token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    
    return creds


def create_oauth_flow(credentials_path: str, redirect_uri: str = None) -> Optional["Flow"]:
    """
    Создает OAuth flow для авторизации.
    
    Args:
        credentials_path: Путь к файлу credentials.json
        redirect_uri: Redirect URI (должен совпадать с настройками в Google Cloud Console)
        
    Returns:
        Flow объект или None
    """
    if not GDRIVE_AVAILABLE:
        return None
    
    if not Path(credentials_path).exists():
        return None
    
    redirect_uri = redirect_uri or REDIRECT_URI
    
    try:
        flow = Flow.from_client_secrets_file(
            credentials_path,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )
        return flow
    except Exception:
        return None


def list_files_from_drive_folder(
    folder_id: str,
    credentials: Optional[Credentials] = None,
    file_type: Optional[str] = None,
    credentials_path: Optional[str] = None,
    log_callback: Optional[callable] = None
) -> List[Dict]:
    """
    Получает список файлов из папки Google Drive.
    
    Args:
        folder_id: ID папки Google Drive
        credentials: Credentials объект (опционально)
        file_type: Фильтр по типу файла (например, 'json')
        credentials_path: Путь к credentials.json (если credentials не предоставлен)
        log_callback: Функция для логирования (message)
        
    Returns:
        Список словарей с информацией о файлах:
        [{'id': 'file_id', 'name': 'file.json', 'mimeType': 'application/json', ...}]
    """
    if not GDRIVE_AVAILABLE:
        log_msg = "❌ Google Drive API недоступен"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        return []
    
    if not credentials:
        credentials = get_credentials(credentials_path)
    
    if not credentials:
        log_msg = "❌ Не удалось получить credentials для Google Drive"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        return []
    
    try:
        log_msg = f"📂 Получение списка файлов из папки Google Drive (ID: {folder_id[:20]}...)"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        
        service = build('drive', 'v3', credentials=credentials)
        
        # Запрос для поиска файлов в папке
        query = f"'{folder_id}' in parents and trashed=false"
        
        if file_type:
            # Фильтр по расширению файла
            query += f" and name contains '.{file_type}'"
            log_msg = f"🔍 Поиск файлов типа: .{file_type}"
            if log_callback:
                log_callback(log_msg)
            else:
                print(log_msg)
        
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, size, modifiedTime)",
            pageSize=1000
        ).execute()
        
        files = results.get('files', [])
        log_msg = f"✅ Найдено файлов: {len(files)}"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        
        # Логируем имена файлов
        if files and log_callback:
            for idx, file_info in enumerate(files[:10], 1):  # Показываем первые 10
                size_mb = int(file_info.get('size', 0)) / (1024 * 1024) if file_info.get('size') else 0
                log_callback(f"  {idx}. {file_info['name']} ({size_mb:.2f} MB)")
            if len(files) > 10:
                log_callback(f"  ... и еще {len(files) - 10} файлов")
        
        return files
    
    except HttpError as error:
        error_msg = f"❌ Ошибка при получении списка файлов: {error}"
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        return []
    except Exception as e:
        error_msg = f"❌ Неожиданная ошибка: {e}"
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        return []


def download_file_from_drive(
    file_id: str,
    credentials: Optional[Credentials] = None,
    credentials_path: Optional[str] = None,
    file_name: Optional[str] = None,
    log_callback: Optional[callable] = None
) -> Optional[str]:
    """
    Скачивает файл из Google Drive.
    
    Args:
        file_id: ID файла в Google Drive
        credentials: Credentials объект (опционально)
        credentials_path: Путь к credentials.json (если credentials не предоставлен)
        file_name: Имя файла для логирования (опционально)
        log_callback: Функция для логирования (message)
        
    Returns:
        Содержимое файла как строка или None при ошибке
    """
    if not GDRIVE_AVAILABLE:
        return None
    
    if not credentials:
        credentials = get_credentials(credentials_path)
    
    if not credentials:
        return None
    
    display_name = file_name or file_id[:20]
    
    try:
        service = build('drive', 'v3', credentials=credentials)
        
        # Получаем метаданные файла
        file_metadata = service.files().get(fileId=file_id).execute()
        actual_name = file_metadata.get('name', display_name)
        file_size = int(file_metadata.get('size', 0)) if file_metadata.get('size') else 0
        size_mb = file_size / (1024 * 1024) if file_size > 0 else 0
        
        log_msg = f"⬇️  Скачивание: {actual_name}"
        if size_mb > 0:
            log_msg += f" ({size_mb:.2f} MB)"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        
        # Скачиваем файл
        request = service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)
        
        done = False
        chunk_count = 0
        while not done:
            status, done = downloader.next_chunk()
            chunk_count += 1
            if status and log_callback and chunk_count % 10 == 0:  # Логируем каждые 10 chunks
                progress = int(status.progress() * 100)
                log_callback(f"  📥 Прогресс: {progress}%")
        
        # Возвращаем содержимое как строку
        file_content.seek(0)
        content = file_content.read().decode('utf-8')
        
        log_msg = f"✅ Загружен: {actual_name} ({len(content)} символов)"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        
        return content
    
    except HttpError as error:
        error_msg = f"❌ Ошибка при скачивании файла {display_name}: {error}"
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        return None
    except Exception as e:
        error_msg = f"❌ Неожиданная ошибка при скачивании {display_name}: {e}"
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        return None


def upload_file_to_drive(
    file_path: Path,
    folder_id: str,
    credentials: Optional[Credentials] = None,
    credentials_path: Optional[str] = None,
    file_name: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Optional[str]:
    """
    Загружает файл в Google Drive с поддержкой прогресса и таймаутов.
    
    Args:
        file_path: Путь к локальному файлу
        folder_id: ID папки в Google Drive для загрузки
        credentials: Credentials объект (опционально)
        credentials_path: Путь к credentials.json (если credentials не предоставлен)
        file_name: Имя файла в Drive (если отличается от локального)
        progress_callback: Функция для отчета о прогрессе (progress, total)
        
    Returns:
        ID загруженного файла или None при ошибке
    """
    if not GDRIVE_AVAILABLE:
        return None
    
    if not credentials:
        credentials = get_credentials(credentials_path)
    
    if not credentials:
        return None
    
    if not file_path.exists():
        return None
    
    file_name = file_name or file_path.name
    file_size = file_path.stat().st_size
    
    try:
        import time
        
        service = build('drive', 'v3', credentials=credentials)
        
        # Метаданные файла
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        
        # Определяем mimetype
        mimetype = 'application/json'
        if file_path.suffix.lower() in ['.json']:
            mimetype = 'application/json'
        elif file_path.suffix.lower() in ['.txt', '.csv']:
            mimetype = 'text/plain'
        else:
            mimetype = 'application/octet-stream'
        
        # Для больших файлов используем resumable upload
        # Порог для resumable: 5MB (рекомендация Google)
        use_resumable = file_size > 5 * 1024 * 1024
        
        if use_resumable:
            # Resumable upload для больших файлов
            media = MediaFileUpload(
                str(file_path),
                mimetype=mimetype,
                resumable=True,
                chunksize=1024*1024  # 1MB chunks
            )
            
            # Инициируем загрузку
            request = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            )
            
            # Выполняем загрузку с прогрессом
            response = None
            start_time = time.time()
            timeout = 300  # 5 минут таймаут
            max_retries = 10
            retry_count = 0
            
            while response is None:
                try:
                    # Проверка таймаута перед каждой попыткой
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Загрузка превысила таймаут {timeout} секунд")
                    
                    status, response = request.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        if progress_callback:
                            progress_callback(progress, 100)
                    # Сбрасываем счетчик при успешном прогрессе
                    retry_count = 0
                except HttpError as e:
                    # Для HTTP ошибок не повторяем бесконечно
                    if e.resp.status in [400, 401, 403, 404]:
                        raise  # Критические ошибки
                    # Для других HTTP ошибок повторяем с ограничением
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception(f"Превышено максимальное количество попыток ({max_retries})")
                    time.sleep(min(retry_count * 2, 10))  # Экспоненциальная задержка
                except Exception as e:
                    if isinstance(e, (TimeoutError, KeyboardInterrupt)):
                        raise
                    # Повторяем попытку при других ошибках сети
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception(f"Превышено максимальное количество попыток ({max_retries}): {e}")
                    time.sleep(min(retry_count * 2, 10))  # Экспоненциальная задержка
            
            return response.get('id') if response else None
        else:
            # Простая загрузка для маленьких файлов
            media = MediaFileUpload(
                str(file_path),
                mimetype=mimetype,
                resumable=False
            )
            
            if progress_callback:
                progress_callback(50, 100)
            
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            if progress_callback:
                progress_callback(100, 100)
            
            return file.get('id')
    
    except TimeoutError as e:
        print(f"Таймаут при загрузке файла: {e}")
        return None
    except HttpError as error:
        print(f"Ошибка при загрузке файла: {error}")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка при загрузке: {e}")
        return None


def load_json_from_drive_folder(
    folder_url: str,
    credentials: Optional[Credentials] = None,
    credentials_path: Optional[str] = None,
    log_callback: Optional[callable] = None
) -> Dict[str, dict]:
    """
    Загружает все JSON файлы из папки Google Drive.
    
    Args:
        folder_url: URL или ID папки Google Drive
        credentials: Credentials объект (опционально)
        credentials_path: Путь к credentials.json (если credentials не предоставлен)
        log_callback: Функция для логирования (message)
        
    Returns:
        Словарь {имя_файла: данные_json}
    """
    log_msg = f"🚀 Начало загрузки файлов из Google Drive"
    if log_callback:
        log_callback(log_msg)
    else:
        print(log_msg)
    
    folder_id = extract_folder_id_from_url(folder_url)
    if not folder_id:
        error_msg = "❌ Не удалось извлечь ID папки из URL"
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        return {}
    
    # Получаем список файлов
    files = list_files_from_drive_folder(
        folder_id,
        credentials=credentials,
        file_type='json',
        credentials_path=credentials_path,
        log_callback=log_callback
    )
    
    if not files:
        log_msg = "⚠️ JSON файлы не найдены в указанной папке"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        return {}
    
    # Загружаем данные из каждого файла
    predictions = {}
    
    if not credentials:
        credentials = get_credentials(credentials_path)
    
    total_files = len(files)
    log_msg = f"📥 Начинаю загрузку {total_files} JSON файлов..."
    if log_callback:
        log_callback(log_msg)
    else:
        print(log_msg)
    
    for idx, file_info in enumerate(files, 1):
        file_id = file_info['id']
        file_name = file_info['name']
        file_name_stem = Path(file_name).stem  # Без расширения
        
        log_msg = f"📄 [{idx}/{total_files}] Обработка файла: {file_name}"
        if log_callback:
            log_callback(log_msg)
        else:
            print(log_msg)
        
        # Скачиваем файл
        content = download_file_from_drive(
            file_id,
            credentials=credentials,
            credentials_path=credentials_path,
            file_name=file_name,
            log_callback=log_callback
        )
        
        if content:
            try:
                data = json.loads(content)
                predictions[file_name_stem] = data
                log_msg = f"✅ [{idx}/{total_files}] Успешно загружен и распарсен: {file_name}"
                if log_callback:
                    log_callback(log_msg)
                else:
                    print(log_msg)
            except json.JSONDecodeError as e:
                error_msg = f"❌ [{idx}/{total_files}] Ошибка при парсинге JSON из {file_name}: {e}"
                if log_callback:
                    log_callback(error_msg)
                else:
                    print(error_msg)
        else:
            error_msg = f"❌ [{idx}/{total_files}] Не удалось скачать файл: {file_name}"
            if log_callback:
                log_callback(error_msg)
            else:
                print(error_msg)
    
    log_msg = f"🎉 Загрузка завершена! Успешно загружено {len(predictions)} из {total_files} файлов"
    if log_callback:
        log_callback(log_msg)
    else:
        print(log_msg)
    
    return predictions


def is_gdrive_available() -> bool:
    """Проверяет, доступна ли интеграция с Google Drive."""
    return GDRIVE_AVAILABLE

