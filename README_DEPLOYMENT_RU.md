# Развертывание Dashboard в Google Cloud

Минимальное Dashboard приложение для развертывания в Google Cloud Platform с поддержкой Docker.

## 📋 Требования

- Аккаунт Google Cloud с проектом
- Установленный [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- Установленный [Docker](https://docs.docker.com/get-docker/)
- Подключенный billing аккаунт (для Cloud Run)

## 🚀 Быстрый старт

### Вариант 1: Локальное развертывание (если есть SSH/локальный доступ)

#### 1. Первоначальная настройка Google Cloud

```bash
# Запустите скрипт настройки
./scripts/setup_gcp.sh
```

Скрипт автоматически:
- Проверит и включит необходимые API
- Настроит Docker для Google Container Registry
- Создаст файл конфигурации `.gcp_config.env`

#### 2. Развертывание в Cloud Run

```bash
# Загрузите конфигурацию (опционально)
source .gcp_config.env

# Разверните приложение
./scripts/deploy_gcp.sh
```

После успешного развертывания вы получите URL вашего приложения.

### Вариант 2: Развертывание без SSH (через Cloud Shell) ⭐

Если у вас нет SSH доступа к удаленному серверу, используйте Cloud Shell:

#### 1. Создайте архив проекта

```bash
./scripts/package_for_deployment.sh
```

#### 2. Загрузите архив в Cloud Storage

```bash
./scripts/upload_to_gcs.sh
```

#### 3. Откройте Cloud Shell и разверните

- Откройте [Cloud Shell](https://console.cloud.google.com/cloudshell)
- Скачайте архив: `gsutil cp gs://YOUR-PROJECT-dashboard-deployment/dashboard_deployment_*.tar.gz .`
- Распакуйте: `tar -xzf dashboard_deployment_*.tar.gz && cd dashboard_deployment`
- Разверните: `./scripts/setup_gcp.sh && ./scripts/deploy_gcp.sh`

**Подробная инструкция:** см. [docs/DEPLOYMENT_WITHOUT_SSH.md](docs/DEPLOYMENT_WITHOUT_SSH.md)

### Вариант 3: Через терминал (загрузка в Cloud Storage)

Если вы хотите загрузить архив через терминал и затем использовать в Cloud Shell:

```bash
# 1. Создайте архив
./scripts/package_for_deployment.sh

# 2. Загрузите в Cloud Storage
./scripts/upload_to_gcs.sh

# 3. В Cloud Shell выполните:
#    gsutil cp gs://YOUR-PROJECT-dashboard-deployment/dashboard_deployment_*.tar.gz .
#    tar -xzf dashboard_deployment_*.tar.gz
#    cd dashboard_deployment
#    ./scripts/setup_gcp.sh && ./scripts/deploy_gcp.sh
```

**Или используйте автоматизированный скрипт:**
```bash
./scripts/deploy_from_terminal.sh
```

**Подробная инструкция:** см. [docs/DEPLOYMENT_VIA_TERMINAL.md](docs/DEPLOYMENT_VIA_TERMINAL.md)

## 🐳 Локальная разработка с Docker

### Вариант 1: Docker Compose (рекомендуется)

```bash
docker-compose up
```

Приложение будет доступно по адресу: http://localhost:8080

### Вариант 2: Скрипт запуска

```bash
./scripts/run_docker.sh
```

### Вариант 3: Ручная сборка и запуск

```bash
# Сборка образа
./scripts/build_docker.sh

# Запуск контейнера
docker run -p 8080:8080 dashboard:latest
```

## 📁 Структура проекта

```
.
├── dashboard_minimal.py              # Минимальное Dashboard приложение
├── requirements_dashboard_minimal.txt  # Зависимости
├── Dockerfile.dashboard              # Dockerfile для сборки
├── docker-compose.yml                # Docker Compose конфигурация
├── cloudbuild.yaml                   # Конфигурация Cloud Build
├── .dockerignore                     # Исключения для Docker
├── .gcloudignore                     # Исключения для gcloud
├── .streamlit/
│   └── config.toml                   # Конфигурация Streamlit
└── scripts/
    ├── setup_gcp.sh                  # Настройка GCP проекта
    ├── deploy_gcp.sh                 # Развертывание в Cloud Run
    ├── build_docker.sh               # Локальная сборка Docker
    └── run_docker.sh                 # Запуск Docker контейнера
```

## ⚙️ Конфигурация

### Переменные окружения

Вы можете настроить следующие переменные:

```bash
export GCP_PROJECT_ID="ваш-project-id"
export GCP_REGION="us-central1"
export SERVICE_NAME="dashboard"
export PORT=8080
```

Или создайте файл `.gcp_config.env`:

```bash
GCP_PROJECT_ID=ваш-project-id
GCP_REGION=us-central1
SERVICE_NAME=dashboard
IMAGE_NAME=gcr.io/ваш-project-id/dashboard:latest
```

### Настройка ресурсов Cloud Run

Параметры можно изменить в `scripts/deploy_gcp.sh`:

- `--memory`: Память (по умолчанию: 2Gi)
- `--cpu`: CPU (по умолчанию: 2)
- `--max-instances`: Максимум инстансов (по умолчанию: 10)
- `--timeout`: Таймаут запроса (по умолчанию: 300 секунд)

## 🔄 Обновление приложения

Для обновления развернутого приложения:

```bash
./scripts/deploy_gcp.sh
```

Или используйте Cloud Build:

```bash
gcloud builds submit --config cloudbuild.yaml
```

## 📊 Мониторинг

### Просмотр логов

```bash
gcloud run services logs read dashboard --region us-central1 --limit 50
```

### Веб-консоль

Откройте [Cloud Run Console](https://console.cloud.google.com/run) для просмотра метрик.

## 🛠️ Устранение неполадок

### Ошибка авторизации

```bash
gcloud auth login
gcloud auth configure-docker
```

### Ошибка billing

Убедитесь, что billing аккаунт подключен:
```bash
gcloud beta billing projects describe PROJECT_ID
```

### Проверка статуса

```bash
gcloud run services describe dashboard --region us-central1
```

## 💰 Стоимость

Cloud Run использует оплату за использование:
- Первые 2 миллиона запросов в месяц - бесплатно
- Оплата за время выполнения и память
- Подробнее: https://cloud.google.com/run/pricing

## 🔒 Безопасность

По умолчанию сервис доступен всем (`--allow-unauthenticated`).

Для ограничения доступа используйте IAM:

```bash
gcloud run services add-iam-policy-binding dashboard \
  --region us-central1 \
  --member "user:email@example.com" \
  --role "roles/run.invoker"
```

## 📝 Особенности минимального Dashboard

Минимальный Dashboard включает:

- ✅ Загрузку JSON файлов через веб-интерфейс
- ✅ Загрузку из директории
- ✅ Базовую агрегацию данных
- ✅ Визуализацию распределений
- ✅ Корреляционный анализ
- ✅ Экспорт данных в CSV

Для расширенного функционала используйте полную версию `scale/dashboard.py`.

## 📚 Дополнительная информация

- [Cloud Run документация](https://cloud.google.com/run/docs)
- [Docker документация](https://docs.docker.com/)
- [Streamlit документация](https://docs.streamlit.io/)

