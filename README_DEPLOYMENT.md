# Развертывание Dashboard в Google Cloud

Это руководство описывает процесс развертывания минимального Dashboard приложения в Google Cloud Platform.

## 📋 Требования

- Google Cloud аккаунт с активным проектом
- Установленный [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- Установленный [Docker](https://docs.docker.com/get-docker/)
- Billing аккаунт подключен к проекту (для Cloud Run)

## 🚀 Быстрый старт

### 1. Первоначальная настройка

```bash
# Запустите скрипт настройки
./scripts/setup_gcp.sh
```

Скрипт выполнит:
- Проверку и включение необходимых API
- Настройку Docker для Google Container Registry
- Создание файла конфигурации `.gcp_config.env`

### 2. Развертывание

```bash
# Загрузите конфигурацию (опционально)
source .gcp_config.env

# Разверните приложение
./scripts/deploy_gcp.sh
```

После успешного развертывания вы получите URL вашего приложения.

## 🐳 Локальная разработка с Docker

### Сборка образа

```bash
./scripts/build_docker.sh
```

Или вручную:
```bash
docker build -f Dockerfile.dashboard -t dashboard:latest .
```

### Запуск с Docker Compose

```bash
docker-compose up
```

Приложение будет доступно по адресу: http://localhost:8080

### Запуск напрямую

```bash
./scripts/run_docker.sh
```

Или вручную:
```bash
docker run -p 8080:8080 dashboard:latest
```

## 📁 Структура файлов

```
.
├── dashboard_minimal.py          # Минимальное приложение Dashboard
├── requirements_dashboard_minimal.txt  # Зависимости для dashboard
├── Dockerfile.dashboard          # Dockerfile для сборки образа
├── docker-compose.yml            # Docker Compose конфигурация
├── cloudbuild.yaml               # Конфигурация Cloud Build
├── .dockerignore                 # Исключения для Docker
├── .gcloudignore                 # Исключения для gcloud
└── scripts/
    ├── setup_gcp.sh              # Настройка GCP проекта
    ├── deploy_gcp.sh             # Развертывание в Cloud Run
    ├── build_docker.sh           # Локальная сборка Docker
    └── run_docker.sh             # Запуск Docker контейнера
```

## ⚙️ Конфигурация

### Переменные окружения

Вы можете настроить следующие переменные:

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
export SERVICE_NAME="dashboard"
export PORT=8080
```

Или создайте файл `.gcp_config.env`:

```bash
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
SERVICE_NAME=dashboard
IMAGE_NAME=gcr.io/your-project-id/dashboard:latest
```

### Настройка Cloud Run

Параметры развертывания можно изменить в `scripts/deploy_gcp.sh`:

- `--memory`: Память (по умолчанию: 2Gi)
- `--cpu`: CPU (по умолчанию: 2)
- `--max-instances`: Максимальное количество инстансов (по умолчанию: 10)
- `--timeout`: Таймаут запроса (по умолчанию: 300 секунд)

## 🔄 Обновление приложения

Для обновления развернутого приложения:

```bash
./scripts/deploy_gcp.sh
```

Или используйте Cloud Build (если настроен CI/CD):

```bash
gcloud builds submit --config cloudbuild.yaml
```

## 📊 Мониторинг и логи

### Просмотр логов

```bash
gcloud run services logs read dashboard --region us-central1 --limit 50
```

### Мониторинг в консоли

Откройте [Cloud Run Console](https://console.cloud.google.com/run) для просмотра метрик и логов.

## 🛠️ Устранение неполадок

### Ошибка авторизации

```bash
gcloud auth login
gcloud auth configure-docker
```

### Ошибка billing

Убедитесь, что billing аккаунт подключен к проекту:
```bash
gcloud beta billing projects describe PROJECT_ID
```

### Ошибка при сборке Docker образа

Проверьте, что все зависимости указаны в `requirements_dashboard_minimal.txt` и файл `dashboard_minimal.py` существует.

### Проверка статуса сервиса

```bash
gcloud run services describe dashboard --region us-central1
```

## 💰 Стоимость

Cloud Run использует модель оплаты за использование:
- Первые 2 миллиона запросов в месяц - бесплатно
- Оплата за время выполнения и память
- Подробнее: https://cloud.google.com/run/pricing

## 🔒 Безопасность

- По умолчанию сервис развертывается с `--allow-unauthenticated`
- Для ограничения доступа используйте IAM роли:
  ```bash
  gcloud run services add-iam-policy-binding dashboard \
    --region us-central1 \
    --member "user:email@example.com" \
    --role "roles/run.invoker"
  ```

## 📚 Дополнительные ресурсы

- [Cloud Run документация](https://cloud.google.com/run/docs)
- [Docker документация](https://docs.docker.com/)
- [Streamlit документация](https://docs.streamlit.io/)

