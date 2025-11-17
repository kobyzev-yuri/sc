# ⚡ Быстрый старт: Развертывание в Google Cloud

## За 3 шага

### 1️⃣ Установите Google Cloud SDK

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### 2️⃣ Настройте проект

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

**Где найти PROJECT_ID?**
- Откройте [Google Cloud Console](https://console.cloud.google.com)
- Вверху страницы будет показан ID проекта
- Или создайте новый проект: `gcloud projects create YOUR_PROJECT_ID`

### 3️⃣ Разверните

```bash
cd /mnt/ai/cnn/sc
./deploy_gcp.sh
```

Готово! 🎉 Скрипт покажет URL вашего дашборда.

---

## Что дальше?

- **Откройте URL** из вывода скрипта в браузере
- **Просмотр логов:** `gcloud run services logs read dashboard --region us-central1 --follow`
- **Обновление:** просто запустите `./deploy_gcp.sh` снова

---

## Проблемы?

**Ошибка "project not set":**
```bash
gcloud config set project YOUR_PROJECT_ID
```

**Ошибка "permission denied":**
```bash
gcloud auth login
```

**Ошибка "API not enabled":**
Скрипт автоматически включит API, но если не работает:
```bash
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
```

---

Подробнее: [DEPLOYMENT_GCP.md](DEPLOYMENT_GCP.md)

