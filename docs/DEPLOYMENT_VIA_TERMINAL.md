# Развертывание через терминал

Это руководство описывает способы развертывания Dashboard через терминал без использования веб-интерфейса.

## 🎯 Два основных способа

### Способ 1: Архив → Cloud Storage → Cloud Shell ⭐ (Рекомендуется)

**Преимущества:**
- Не требует SSH подключения
- Работает из любого терминала
- Простой процесс

#### Шаг 1: На вашем компьютере (локальный терминал)

```bash
# 1. Создайте архив
./scripts/package_for_deployment.sh

# 2. Загрузите в Cloud Storage
./scripts/upload_to_gcs.sh
```

Или вручную:

```bash
# Установите проект (если еще не установлен)
gcloud config set project YOUR-PROJECT-ID

# Создайте bucket (если нужно)
gsutil mb -p YOUR-PROJECT-ID -l us-central1 gs://YOUR-PROJECT-ID-dashboard-deployment

# Загрузите архив
gsutil cp dashboard_deployment_*.tar.gz gs://YOUR-PROJECT-ID-dashboard-deployment/
```

#### Шаг 2: В Cloud Shell (браузерный терминал)

1. **Откройте Cloud Shell:**
   ```bash
   # Перейдите по ссылке:
   https://console.cloud.google.com/cloudshell
   ```

2. **Скачайте архив:**
   ```bash
   gsutil cp gs://YOUR-PROJECT-ID-dashboard-deployment/dashboard_deployment_*.tar.gz .
   ```

3. **Распакуйте и разверните:**
   ```bash
   tar -xzf dashboard_deployment_*.tar.gz
   cd dashboard_deployment
   chmod +x scripts/*.sh
   ./scripts/setup_gcp.sh
   ./scripts/deploy_gcp.sh
   ```

---

### Способ 2: Git → Cloud Shell

**Преимущества:**
- Версионирование кода
- Легко обновлять
- Работает с любым Git репозиторием

#### Шаг 1: Запушьте код в Git

```bash
# Если еще не в Git репозитории
git init
git add dashboard_minimal.py Dockerfile.dashboard cloudbuild.yaml
git add requirements_dashboard_minimal.txt scripts/
git add .streamlit/ .dockerignore .gcloudignore
git commit -m "Add deployment files"

# Запушьте в GitHub/GitLab/Bitbucket
git remote add origin YOUR-GIT-URL
git push -u origin main
```

#### Шаг 2: В Cloud Shell

1. **Откройте Cloud Shell:**
   ```bash
   https://console.cloud.google.com/cloudshell
   ```

2. **Клонируйте репозиторий:**
   ```bash
   git clone YOUR-GIT-URL
   cd YOUR-REPO-NAME
   ```

3. **Разверните:**
   ```bash
   chmod +x scripts/*.sh
   ./scripts/setup_gcp.sh
   ./scripts/deploy_gcp.sh
   ```

---

## 🚀 Автоматизированный скрипт

Используйте универсальный скрипт:

```bash
./scripts/deploy_from_terminal.sh
```

Скрипт предложит выбрать метод и выполнит все шаги автоматически.

---

## 📋 Пошаговая инструкция (детальная)

### Вариант A: Через Cloud Storage

#### 1. Подготовка на вашем компьютере

```bash
cd /mnt/ai/cnn/sc

# Создайте архив
./scripts/package_for_deployment.sh

# Результат: dashboard_deployment_YYYYMMDD_HHMMSS.tar.gz
```

#### 2. Загрузка в Cloud Storage

**Автоматически:**
```bash
./scripts/upload_to_gcs.sh
```

**Вручную:**
```bash
# Установите проект
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID

# Создайте bucket (если нужно)
gsutil mb -p $GCP_PROJECT_ID -l us-central1 \
  gs://${GCP_PROJECT_ID}-dashboard-deployment

# Загрузите архив
ARCHIVE=$(ls -t dashboard_deployment_*.tar.gz | head -1)
gsutil cp "$ARCHIVE" gs://${GCP_PROJECT_ID}-dashboard-deployment/
```

#### 3. Работа в Cloud Shell

```bash
# Откройте Cloud Shell в браузере
# https://console.cloud.google.com/cloudshell

# Скачайте архив
PROJECT_ID="your-project-id"
gsutil cp gs://${PROJECT_ID}-dashboard-deployment/dashboard_deployment_*.tar.gz .

# Распакуйте
tar -xzf dashboard_deployment_*.tar.gz
cd dashboard_deployment

# Разверните
chmod +x scripts/*.sh
./scripts/setup_gcp.sh
./scripts/deploy_gcp.sh
```

---

### Вариант B: Через Git

#### 1. Подготовка Git репозитория

```bash
cd /mnt/ai/cnn/sc

# Инициализируйте Git (если еще не сделано)
git init

# Добавьте файлы развертывания
git add dashboard_minimal.py
git add Dockerfile.dashboard
git add cloudbuild.yaml
git add requirements_dashboard_minimal.txt
git add scripts/
git add .streamlit/
git add .dockerignore .gcloudignore

# Закоммитьте
git commit -m "Add deployment configuration"

# Запушьте в удаленный репозиторий
git remote add origin YOUR-GIT-URL
git push -u origin main
```

#### 2. В Cloud Shell

```bash
# Клонируйте репозиторий
git clone YOUR-GIT-URL
cd YOUR-REPO-NAME

# Разверните
chmod +x scripts/*.sh
./scripts/setup_gcp.sh
./scripts/deploy_gcp.sh
```

---

## 🔧 Полезные команды

### Проверка статуса загрузки

```bash
# Список файлов в bucket
gsutil ls gs://YOUR-PROJECT-ID-dashboard-deployment/

# Информация о файле
gsutil ls -l gs://YOUR-PROJECT-ID-dashboard-deployment/dashboard_deployment_*.tar.gz
```

### Прямая загрузка через gcloud

```bash
# Альтернативный способ загрузки
gcloud storage cp dashboard_deployment_*.tar.gz \
  gs://YOUR-PROJECT-ID-dashboard-deployment/
```

### Скачивание из Cloud Shell

```bash
# Скачать архив
gsutil cp gs://BUCKET-NAME/FILE-NAME.tar.gz .

# Или через gcloud
gcloud storage cp gs://BUCKET-NAME/FILE-NAME.tar.gz .
```

---

## 📝 Пример полного процесса

```bash
# === НА ВАШЕМ КОМПЬЮТЕРЕ ===

# 1. Создайте архив
./scripts/package_for_deployment.sh
# Результат: dashboard_deployment_20241119_143000.tar.gz

# 2. Загрузите в Cloud Storage
export GCP_PROJECT_ID="my-project-123"
gsutil mb -p $GCP_PROJECT_ID -l us-central1 \
  gs://${GCP_PROJECT_ID}-dashboard-deployment 2>/dev/null || true

gsutil cp dashboard_deployment_*.tar.gz \
  gs://${GCP_PROJECT_ID}-dashboard-deployment/

echo "Архив загружен! URL: gs://${GCP_PROJECT_ID}-dashboard-deployment/dashboard_deployment_*.tar.gz"

# === В CLOUD SHELL ===

# 1. Скачайте архив
gsutil cp gs://my-project-123-dashboard-deployment/dashboard_deployment_*.tar.gz .

# 2. Распакуйте
tar -xzf dashboard_deployment_*.tar.gz
cd dashboard_deployment

# 3. Разверните
chmod +x scripts/*.sh
./scripts/setup_gcp.sh
./scripts/deploy_gcp.sh
```

---

## ❓ Часто задаваемые вопросы

### Можно ли загрузить напрямую из терминала без Cloud Shell?

Да, если у вас установлены все инструменты (Docker, gcloud) и есть доступ к интернету:

```bash
./scripts/setup_gcp.sh
./scripts/deploy_gcp.sh
```

### Как узнать URL архива в Cloud Storage?

```bash
gsutil ls gs://YOUR-PROJECT-ID-dashboard-deployment/
```

### Можно ли использовать другой bucket?

Да, просто укажите имя bucket:

```bash
gsutil cp dashboard_deployment_*.tar.gz gs://YOUR-BUCKET-NAME/
```

### Как обновить развернутое приложение?

Просто повторите процесс - загрузите новый архив и разверните заново. Cloud Run автоматически обновит сервис.

---

## 🆘 Устранение неполадок

### Ошибка "Access Denied"

Убедитесь, что у вас есть права:
```bash
gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
  --member="user:YOUR-EMAIL" \
  --role="roles/storage.admin"
```

### Ошибка "Bucket not found"

Создайте bucket:
```bash
gsutil mb -p YOUR-PROJECT-ID -l us-central1 gs://YOUR-BUCKET-NAME
```

### Ошибка при скачивании в Cloud Shell

Проверьте, что bucket существует и файл загружен:
```bash
gsutil ls gs://YOUR-BUCKET-NAME/
```

---

## 📚 Полезные ссылки

- [gsutil документация](https://cloud.google.com/storage/docs/gsutil)
- [Cloud Shell документация](https://cloud.google.com/shell/docs)
- [Cloud Storage документация](https://cloud.google.com/storage/docs)

