# Развертывание Dashboard в Google Cloud Platform

Это руководство поможет вам развернуть Streamlit дашборд в Google Cloud Platform.

## Варианты развертывания

### Вариант 1: Google Cloud Run (Рекомендуется) ⭐

**Преимущества:**
- ✅ Автоматическое масштабирование до нуля
- ✅ Платите только за использование
- ✅ HTTPS автоматически
- ✅ Простое развертывание
- ✅ Бесплатный tier: 2 миллиона запросов в месяц

**Недостатки:**
- ⚠️ Холодный старт при первом запросе (если нет активных инстансов)

---

## Подготовка

### 1. Установка Google Cloud SDK

```bash
# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Или через пакетный менеджер
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install google-cloud-sdk

# Проверка установки
gcloud --version
```

### 2. Инициализация Google Cloud

```bash
# Войдите в аккаунт Google Cloud
gcloud auth login

# Установите проект (замените YOUR_PROJECT_ID на ID вашего проекта)
gcloud config set project YOUR_PROJECT_ID

# Включите необходимые API
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

---

## Развертывание в Cloud Run

### Способ 1: Через gcloud CLI (Рекомендуется)

#### Шаг 1: Сборка и развертывание

```bash
# Перейдите в директорию проекта
cd /mnt/ai/cnn/sc

# Соберите и разверните в одну команду
gcloud run deploy dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10
```

**Параметры:**
- `--source .` - автоматически соберет Dockerfile и развернет
- `--region us-central1` - выберите ближайший регион
- `--allow-unauthenticated` - публичный доступ (или уберите для приватного)
- `--memory 2Gi` - память (можно уменьшить до 1Gi для экономии)
- `--cpu 2` - количество CPU
- `--timeout 300` - таймаут запроса (5 минут)
- `--max-instances 10` - максимум инстансов

#### Шаг 2: Получение URL

После развертывания вы получите URL вида:
```
https://dashboard-xxxxx-uc.a.run.app
```

Сохраните этот URL - это адрес вашего дашборда!

### Способ 2: Через Docker (Ручная сборка)

#### Шаг 1: Сборка Docker образа

```bash
# Соберите образ локально
docker build -t gcr.io/YOUR_PROJECT_ID/dashboard:latest .

# Или используйте Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/dashboard:latest
```

#### Шаг 2: Развертывание образа

```bash
gcloud run deploy dashboard \
  --image gcr.io/YOUR_PROJECT_ID/dashboard:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2
```

### Способ 3: Автоматическое развертывание через Cloud Build

#### Шаг 1: Настройка Cloud Build

```bash
# Создайте триггер (опционально, для автоматического развертывания при push)
gcloud builds triggers create github \
  --repo-name=YOUR_REPO_NAME \
  --repo-owner=YOUR_GITHUB_USERNAME \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```

#### Шаг 2: Ручной запуск сборки

```bash
# Запустите сборку вручную
gcloud builds submit --config cloudbuild.yaml
```

---

## Развертывание в App Engine (Альтернативный вариант)

### Подготовка

1. **Создайте entrypoint скрипт** (`main.py`):

```python
import subprocess
import sys

if __name__ == '__main__':
    # Запускаем Streamlit через gunicorn или напрямую
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        'scale/dashboard.py',
        '--server.port=8080',
        '--server.address=0.0.0.0'
    ])
```

2. **Обновите app.yaml** (уже создан)

3. **Развертывание:**

```bash
gcloud app deploy app.yaml
```

**Примечание:** App Engine Flexible требует больше настроек и обычно дороже, чем Cloud Run.

---

## Настройка переменных окружения

Если нужно передать переменные окружения:

```bash
gcloud run deploy dashboard \
  --source . \
  --set-env-vars "ENV_VAR1=value1,ENV_VAR2=value2" \
  --region us-central1
```

Или через файл:

```bash
# Создайте .env.yaml
cat > .env.yaml << EOF
ENV_VAR1: value1
ENV_VAR2: value2
EOF

# Разверните с переменными
gcloud run deploy dashboard \
  --source . \
  --env-vars-file .env.yaml \
  --region us-central1
```

---

## Работа с данными

### Вариант 1: Cloud Storage (Рекомендуется)

Для хранения больших файлов (модели, результаты):

```bash
# Создайте bucket
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://YOUR_BUCKET_NAME

# Загрузите данные
gsutil -m cp -r models/ gs://YOUR_BUCKET_NAME/
gsutil -m cp -r results/ gs://YOUR_BUCKET_NAME/
```

В коде дашборда добавьте загрузку из Cloud Storage:

```python
from google.cloud import storage

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
```

### Вариант 2: Локальное хранилище (временное)

Cloud Run имеет эфемерную файловую систему. Данные сохраняются только во время работы инстанса.

---

## Мониторинг и логи

### Просмотр логов

```bash
# Логи в реальном времени
gcloud run services logs read dashboard --region us-central1 --follow

# Последние 100 строк
gcloud run services logs read dashboard --region us-central1 --limit 100
```

### Мониторинг в консоли

1. Откройте [Cloud Console](https://console.cloud.google.com)
2. Перейдите в **Cloud Run** → **dashboard**
3. Вкладка **Логи** - просмотр логов
4. Вкладка **Метрики** - мониторинг производительности

---

## Обновление приложения

### Обновление кода

```bash
# После изменений в коде
gcloud run deploy dashboard \
  --source . \
  --region us-central1
```

### Обновление только образа

```bash
# Если изменили только Dockerfile
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/dashboard:latest
gcloud run deploy dashboard \
  --image gcr.io/YOUR_PROJECT_ID/dashboard:latest \
  --region us-central1
```

---

## Управление доступом

### Публичный доступ (текущая настройка)

```bash
# Уже настроено через --allow-unauthenticated
```

### Приватный доступ

```bash
# Удалите публичный доступ
gcloud run services update dashboard \
  --no-allow-unauthenticated \
  --region us-central1

# Добавьте доступ для конкретного пользователя
gcloud run services add-iam-policy-binding dashboard \
  --member="user:email@example.com" \
  --role="roles/run.invoker" \
  --region us-central1
```

---

## Оптимизация производительности

### Увеличение ресурсов

```bash
gcloud run services update dashboard \
  --memory 4Gi \
  --cpu 4 \
  --region us-central1
```

### Настройка масштабирования

```bash
# Минимум инстансов (для уменьшения холодного старта)
gcloud run services update dashboard \
  --min-instances 1 \
  --region us-central1

# Максимум инстансов
gcloud run services update dashboard \
  --max-instances 20 \
  --region us-central1
```

### Уменьшение холодного старта

1. Увеличьте `--min-instances` до 1
2. Уменьшите размер образа (оптимизируйте Dockerfile)
3. Используйте более быстрый регион

---

## Стоимость

### Cloud Run (примерная стоимость)

- **Бесплатный tier:** 2 миллиона запросов/месяц, 360,000 ГБ-секунд
- **Платные тарифы:**
  - CPU: $0.00002400 за vCPU-секунду
  - Память: $0.00000250 за ГБ-секунду
  - Запросы: $0.40 за миллион запросов

**Пример:** Приложение с 1 vCPU, 2GB RAM, работающее 10% времени:
- ~$15-30/месяц (в зависимости от использования)

### Оптимизация стоимости

1. Используйте `--min-instances 0` (масштабирование до нуля)
2. Уменьшите `--memory` до минимума (1Gi)
3. Уменьшите `--cpu` до 1
4. Используйте бесплатный tier максимально

---

## Troubleshooting

### Ошибка: "Container failed to start"

```bash
# Проверьте логи
gcloud run services logs read dashboard --region us-central1 --limit 50
```

**Возможные причины:**
- Неправильный порт (должен быть 8080)
- Ошибки в коде
- Недостаточно памяти

### Ошибка: "Permission denied"

```bash
# Проверьте права доступа
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### Медленная загрузка

1. Проверьте размер образа: `docker images`
2. Оптимизируйте Dockerfile (многослойный кэш)
3. Используйте более быстрый регион

### Проблемы с зависимостями

```bash
# Проверьте requirements.txt
pip install -r requirements.txt

# Проверьте совместимость версий
pip check
```

---

## Быстрый старт (TL;DR)

```bash
# 1. Установите gcloud CLI
# 2. Войдите и настройте проект
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Включите API
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# 4. Разверните
cd /mnt/ai/cnn/sc
gcloud run deploy dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi

# 5. Откройте URL из вывода команды
```

---

## Дополнительные ресурсы

- [Документация Cloud Run](https://cloud.google.com/run/docs)
- [Цены Cloud Run](https://cloud.google.com/run/pricing)
- [Best Practices](https://cloud.google.com/run/docs/tips)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

---

## Поддержка

Если возникли проблемы:
1. Проверьте логи: `gcloud run services logs read dashboard --region us-central1`
2. Проверьте документацию Google Cloud
3. Проверьте issues в репозитории проекта

