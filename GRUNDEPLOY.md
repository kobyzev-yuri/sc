## Деплой нового кода в Google Cloud Run

### 1. Предварительные условия

- Код в `main` обновлён и запушен на GitHub.
- Локально установлен `gcloud` и выполнен логин:

```bash
gcloud auth login
gcloud config set project scalepathology   # или другой PROJECT_ID
```

### 2. Быстрый деплой через скрипт

В корне проекта:

```bash
cd api
./deploy_api.sh
```

Скрипт делает:

- Ставит параметры (можно переопределить переменными окружения):
  - `PROJECT_ID` (по умолчанию `scalepathology`)
  - `REGION` (по умолчанию `us-central1`)
  - `SERVICE_NAME` (по умолчанию `pathology-api`)
- Собирает Docker‑образ:

```bash
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME} --project=${PROJECT_ID}
```

- Деплоит в Cloud Run:

```bash
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 0 \
  --port 8080 \
  --timeout 300 \
  --project=${PROJECT_ID}
```

- В конце показывает URL сервиса:

```bash
gcloud run services describe ${SERVICE_NAME} \
  --region=${REGION} \
  --format="value(status.url)" \
  --project=${PROJECT_ID}
```

### 3. Ручной деплой (без скрипта)

Из директории `api/` или корня проекта:

```bash
PROJECT_ID=scalepathology
REGION=us-central1
SERVICE_NAME=pathology-api
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

gcloud config set project ${PROJECT_ID}

# Сборка образа
gcloud builds submit --tag ${IMAGE} --project=${PROJECT_ID}

# Деплой в Cloud Run
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 0 \
  --port 8080 \
  --timeout 300 \
  --project=${PROJECT_ID}
```

### 4. Что важно не забыть

- **Конфиг шкалы**: `scale/cfg/feature_selection_config_relative.json` уже синхронизирован с лучшим экспериментом.
- **Данные по умолчанию**: `results/predictions` и `results/inference` содержат merged JSON (Endocrine включён).
- **Доступы к внешним сервисам (опционально)**:
  - Для работы с Google Drive: переменные `GOOGLE_DRIVE_CREDENTIALS_JSON_B64` или `GOOGLE_DRIVE_CREDENTIALS_JSON`.
  - Для GCS: у сервисного аккаунта Cloud Run должны быть права на нужные бакеты (Storage Object Viewer / Admin по ситуации).



