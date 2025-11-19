# Развертывание без SSH подключения

Это руководство описывает способы развертывания Dashboard в Google Cloud без настройки SSH подключения.

## 🎯 Доступные способы

### 1. Google Cloud Shell (Рекомендуется) ⭐

Cloud Shell - это встроенный терминал в браузере с предустановленными инструментами.

#### Шаги:

1. **Создайте архив проекта:**
   ```bash
   ./scripts/package_for_deployment.sh
   ```
   Это создаст файл `dashboard_deployment_YYYYMMDD_HHMMSS.tar.gz`

2. **Откройте Cloud Shell:**
   - Перейдите в [Google Cloud Console](https://console.cloud.google.com)
   - Нажмите на иконку Cloud Shell в правом верхнем углу (терминал)
   - Или откройте напрямую: https://console.cloud.google.com/cloudshell

3. **Загрузите архив в Cloud Shell:**
   
   **Вариант A: Через веб-интерфейс Cloud Shell**
   - В Cloud Shell нажмите на меню (три точки) → "Upload file"
   - Выберите созданный архив
   - Дождитесь завершения загрузки

   **Вариант B: Через Cloud Storage (рекомендуется)**
   ```bash
   # На вашем компьютере
   ./scripts/upload_to_gcs.sh
   
   # В Cloud Shell
   gsutil cp gs://YOUR-PROJECT-ID-dashboard-deployment/dashboard_deployment_*.tar.gz .
   ```

4. **Распакуйте архив:**
   ```bash
   tar -xzf dashboard_deployment_*.tar.gz
   cd dashboard_deployment
   ```

5. **Разверните приложение:**
   ```bash
   chmod +x scripts/*.sh
   ./scripts/setup_gcp.sh
   ./scripts/deploy_gcp.sh
   ```

---

### 2. Google Cloud Storage + Cloud Build

Автоматическое развертывание через Cloud Build.

#### Шаги:

1. **Создайте и загрузите архив:**
   ```bash
   ./scripts/package_for_deployment.sh
   ./scripts/upload_to_gcs.sh
   ```

2. **Откройте Cloud Shell и скачайте архив:**
   ```bash
   gsutil cp gs://YOUR-PROJECT-ID-dashboard-deployment/dashboard_deployment_*.tar.gz .
   tar -xzf dashboard_deployment_*.tar.gz
   cd dashboard_deployment
   ```

3. **Запустите Cloud Build:**
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

---

### 3. Прямая загрузка через веб-интерфейс

Если у вас нет доступа к командной строке.

#### Шаги:

1. **Создайте архив:**
   ```bash
   ./scripts/package_for_deployment.sh
   ```

2. **Загрузите в Cloud Storage через веб-интерфейс:**
   - Откройте [Cloud Storage Console](https://console.cloud.google.com/storage)
   - Создайте bucket (если нужно)
   - Нажмите "Upload files"
   - Выберите созданный архив

3. **Используйте Cloud Shell:**
   - Откройте Cloud Shell
   - Скачайте архив из bucket
   - Следуйте инструкциям из способа 1

---

### 4. GitHub / GitLab + Cloud Build

Если ваш проект уже в Git репозитории.

#### Шаги:

1. **Закоммитьте файлы развертывания:**
   ```bash
   git add dashboard_minimal.py Dockerfile.dashboard cloudbuild.yaml
   git add requirements_dashboard_minimal.txt scripts/
   git commit -m "Add deployment files"
   git push
   ```

2. **Настройте Cloud Build для подключения к репозиторию:**
   - Откройте [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers)
   - Создайте новый trigger
   - Подключите ваш GitHub/GitLab репозиторий
   - Укажите `cloudbuild.yaml` как конфигурацию

3. **Запустите сборку:**
   - Cloud Build автоматически соберет и развернет приложение
   - Или запустите вручную: `gcloud builds submit --config cloudbuild.yaml`

---

## 📋 Пошаговая инструкция (Cloud Shell)

### Шаг 1: Подготовка архива на вашем компьютере

```bash
cd /mnt/ai/cnn/sc
./scripts/package_for_deployment.sh
```

Результат: файл `dashboard_deployment_YYYYMMDD_HHMMSS.tar.gz`

### Шаг 2: Загрузка в Cloud Storage

**Вариант A: Автоматически (рекомендуется)**
```bash
./scripts/upload_to_gcs.sh
```

**Вариант B: Вручную**
```bash
# Установите проект
gcloud config set project YOUR-PROJECT-ID

# Создайте bucket (если нужно)
gsutil mb -p YOUR-PROJECT-ID -l us-central1 gs://YOUR-PROJECT-ID-dashboard-deployment

# Загрузите архив
gsutil cp dashboard_deployment_*.tar.gz gs://YOUR-PROJECT-ID-dashboard-deployment/
```

### Шаг 3: Работа в Cloud Shell

1. **Откройте Cloud Shell:**
   - https://console.cloud.google.com/cloudshell

2. **Скачайте архив:**
   ```bash
   gsutil cp gs://YOUR-PROJECT-ID-dashboard-deployment/dashboard_deployment_*.tar.gz .
   ```

3. **Распакуйте:**
   ```bash
   tar -xzf dashboard_deployment_*.tar.gz
   cd dashboard_deployment
   ```

4. **Настройте проект:**
   ```bash
   chmod +x scripts/*.sh
   ./scripts/setup_gcp.sh
   ```

5. **Разверните:**
   ```bash
   ./scripts/deploy_gcp.sh
   ```

---

## 🔧 Альтернатива: Использование Cloud Build напрямую

Если у вас уже есть архив в Cloud Storage:

```bash
# В Cloud Shell
gsutil cp gs://YOUR-BUCKET/dashboard_deployment_*.tar.gz .
tar -xzf dashboard_deployment_*.tar.gz
cd dashboard_deployment

# Запустите Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

---

## 📝 Что включено в архив

Архив содержит:
- ✅ `dashboard_minimal.py` - приложение
- ✅ `requirements_dashboard_minimal.txt` - зависимости
- ✅ `Dockerfile.dashboard` - Docker образ
- ✅ `cloudbuild.yaml` - конфигурация Cloud Build
- ✅ `scripts/` - все скрипты развертывания
- ✅ `.streamlit/config.toml` - конфигурация Streamlit
- ✅ Документация

---

## ❓ Часто задаваемые вопросы

### Как узнать мой Project ID?

```bash
gcloud config get-value project
```

Или в [Cloud Console](https://console.cloud.google.com) - Project ID отображается в верхней панели.

### Нужен ли billing аккаунт?

Да, для Cloud Run необходим подключенный billing аккаунт. Первые 2 миллиона запросов в месяц бесплатны.

### Можно ли использовать другой регион?

Да, измените `GCP_REGION` в скриптах или укажите при запуске:
```bash
GCP_REGION=europe-west1 ./scripts/deploy_gcp.sh
```

### Как обновить развернутое приложение?

Просто повторите процесс развертывания - Cloud Run автоматически обновит сервис.

---

## 🆘 Устранение неполадок

### Ошибка "Permission denied"

Убедитесь, что у вас есть права:
- Cloud Run Admin
- Cloud Build Editor
- Storage Admin

```bash
gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
  --member="user:YOUR-EMAIL" \
  --role="roles/run.admin"
```

### Ошибка при загрузке в Cloud Storage

Проверьте, что bucket существует и у вас есть права:
```bash
gsutil ls gs://YOUR-BUCKET-NAME
```

### Cloud Shell не открывается

Попробуйте:
- Очистить кеш браузера
- Использовать другой браузер
- Проверить, что JavaScript включен

---

## 📚 Полезные ссылки

- [Cloud Shell документация](https://cloud.google.com/shell/docs)
- [Cloud Storage документация](https://cloud.google.com/storage/docs)
- [Cloud Build документация](https://cloud.google.com/build/docs)
- [Cloud Run документация](https://cloud.google.com/run/docs)

