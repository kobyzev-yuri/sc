# Pathology Analysis API

FastAPI сервер с функциональностью аналогичной Streamlit dashboard.

## Возможности

- ✅ Загрузка данных из разных источников (директория, Google Drive, GCS)
- ✅ Агрегация predictions и создание признаков
- ✅ Оценка качества признаков (feature evaluation)
- ✅ PCA scoring для образцов
- ✅ Спектральный анализ данных
- ✅ Список доступных экспериментов

## Установка

```bash
cd api
pip install -r requirements.txt
```

## Локальный запуск

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Или:

```bash
python -m api.main
```

## Развертывание в Cloud Run

```bash
./api/deploy_api.sh
```

Или вручную:

```bash
# Сборка образа
gcloud builds submit --tag gcr.io/scalepathology/pathology-api

# Развертывание
gcloud run deploy pathology-api \
    --image gcr.io/scalepathology/pathology-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --port 8080
```

## API Endpoints

### `GET /`
Информация об API и доступных endpoints.

### `POST /api/v1/load-data`
Загружает данные из указанного источника.

**Request body:**
```json
{
    "source": "directory",  // или "gdrive", "gcs"
    "path": "results/predictions",  // путь к директории или URL
    "bucket_name": "scalebucket",  // для GCS
    "prefix": ""  // для GCS
}
```

**Response:**
```json
{
    "status": "success",
    "source": "directory",
    "files_count": 36,
    "cache_key": "predictions_directory_results/predictions",
    "sample_names": ["image1", "image2", ...]
}
```

### `POST /api/v1/aggregate`
Агрегирует загруженные predictions.

**Form data:**
- `cache_key`: ключ из ответа load-data

**Response:**
```json
{
    "status": "success",
    "aggregated_rows": 36,
    "features_count": 150,
    "df_cache_key": "...",
    "df_features_cache_key": "...",
    "feature_columns": ["feature1", "feature2", ...]
}
```

### `POST /api/v1/evaluate-features`
Оценивает качество набора признаков.

**Request body:**
```json
{
    "feature_columns": ["feature1", "feature2", ...],
    "mod_samples": ["image1", "image2"],  // опционально
    "normal_samples": ["image3", "image4"]  // опционально
}
```

**Form data:**
- `df_features_cache_key`: ключ из ответа aggregate

**Response:**
```json
{
    "status": "success",
    "metrics": {
        "score": 3.2,
        "separation": 2.5,
        "mean_pc1_norm_mod": 0.85,
        "explained_variance": 0.65
    },
    "mod_samples_count": 10,
    "normal_samples_count": 26,
    "features_count": 15
}
```

### `POST /api/v1/pca-score`
Вычисляет PCA score для образцов.

**Request body:**
```json
{
    "feature_columns": ["feature1", "feature2", ...],
    "use_relative_features": true
}
```

**Form data:**
- `df_features_cache_key`: ключ из ответа aggregate

**Response:**
```json
{
    "status": "success",
    "results": [
        {"image": "image1", "PC1": 1.5, "PC1_norm": 0.75},
        ...
    ],
    "samples_count": 36
}
```

### `POST /api/v1/spectral-analysis`
Выполняет спектральный анализ.

**Request body:**
```json
{
    "feature_columns": ["feature1", "feature2", ...],
    "use_relative_features": true
}
```

**Form data:**
- `df_features_cache_key`: ключ из ответа aggregate

### `GET /api/v1/experiments`
Возвращает список доступных экспериментов.

### `GET /api/v1/health`
Health check endpoint.

## Пример использования

```python
import requests

BASE_URL = "https://pathology-api-xxx.run.app"

# 1. Загрузить данные
response = requests.post(f"{BASE_URL}/api/v1/load-data", json={
    "source": "directory",
    "path": "results/predictions"
})
data = response.json()
cache_key = data["cache_key"]

# 2. Агрегировать данные
response = requests.post(f"{BASE_URL}/api/v1/aggregate", data={
    "cache_key": cache_key
})
aggregated = response.json()
df_features_key = aggregated["df_features_cache_key"]
features = aggregated["feature_columns"]

# 3. Оценить признаки
response = requests.post(
    f"{BASE_URL}/api/v1/evaluate-features",
    json={
        "feature_columns": features[:10]  # Первые 10 признаков
    },
    data={"df_features_cache_key": df_features_key}
)
metrics = response.json()
print(f"Score: {metrics['metrics']['score']}")
```

## Отличия от Streamlit Dashboard

- ✅ **REST API** вместо веб-интерфейса
- ✅ **JSON ответы** вместо визуализаций
- ✅ **Легче интегрировать** в другие системы
- ✅ **Быстрее** для автоматизации
- ❌ Нет визуализаций (графики, графики спектра)
- ❌ Нет интерактивного выбора признаков

## Примечания

- Данные хранятся в памяти (`_data_cache`) - в продакшене использовать Redis или БД
- Для работы с Google Drive/GCS нужны credentials (как в dashboard)
- API не сохраняет результаты экспериментов автоматически (можно добавить endpoint)






