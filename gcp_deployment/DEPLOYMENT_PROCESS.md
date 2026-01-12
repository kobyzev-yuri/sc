# Процесс развертывания: создание архива и перенос на сервер

## Два способа развертывания

### Способ 1: Создание архива и ручной перенос

**Скрипт:** `create_full_deployment_package.sh`

**Что делает:**
1. Создает директорию `/mnt/ai/cnn/sc/deployment_full/` (или указанную)
2. Копирует туда все необходимые файлы:
   - `scale/` - весь код dashboard
   - `model_development/` - модули для feature selection
   - `results/inference/*.json` - файлы для инференса
   - `experiments/` - эксперименты (JSON, CSV, PKL)
   - `models/` - модели (.pkl)
   - `requirements.txt`
   - `Dockerfile`, `deploy_gcp.sh`, `cloudbuild.yaml`
   - `.dockerignore` (создается автоматически)
3. Создает архив: `deployment_full.tar.gz`
4. **ВЫ должны вручную скопировать архив на сервер:**
   ```bash
   scp /mnt/ai/cnn/sc/deployment_full.tar.gz ai8049520@136.116.116.95:~
   ```
5. **На сервере распаковываете и деплоите:**
   ```bash
   ssh ai8049520@136.116.116.95
   tar -xzf deployment_full.tar.gz
   cd deployment_full
   ./deploy_gcp.sh
   ```

**Где создается архив:**
- Локально: `/mnt/ai/cnn/sc/deployment_full.tar.gz`
- Строка 286 в `create_full_deployment_package.sh`: `tar -czf "$(basename "$PACKAGE_DIR").tar.gz"`

**Использование:**
```bash
cd /mnt/ai/cnn/sc/gcp_deployment
./create_full_deployment_package.sh
# Затем вручную:
scp /mnt/ai/cnn/sc/deployment_full.tar.gz ai8049520@136.116.116.95:~
```

---

### Способ 2: Прямая синхронизация (без архива)

**Скрипт:** `full_sync_and_deploy.sh`

**Что делает:**
1. Подключается к серверу по SSH
2. Синхронизирует файлы напрямую через `rsync` или `scp`:
   - `scale/*.py` - все Python модули
   - `results/inference/*.json` - файлы инференса
   - `results/predictions/*.json` - файлы предсказаний
   - `model_development/` - модули feature selection
   - `Dockerfile`, `deploy_gcp.sh` - скрипты развертывания
   - `.config/gdrive/credentials.json` - credentials
   - `.config/gcs/service-account-key.json` - GCS credentials
3. **Автоматически копирует на сервер** в `~/scalepathology/`
4. Опционально запускает деплой автоматически

**Где происходит синхронизация:**
- Строки 85-104: синхронизация scale модулей
- Строки 118-140: синхронизация `results/inference/`
- Строки 146-163: синхронизация `results/predictions/`
- Строки 247-287: синхронизация credentials
- Строки 309-332: синхронизация Dockerfile и deploy_gcp.sh

**Использование:**
```bash
cd /mnt/ai/cnn/sc/gcp_deployment

# Только синхронизация
./full_sync_and_deploy.sh ai8049520@136.116.116.95

# Синхронизация + автоматический деплой
./full_sync_and_deploy.sh ai8049520@136.116.116.95 --deploy
```

---

## Текущая проблема и решение

**Проблема:** Файлы из `results/inference/` попадали в Docker образ при сборке, из-за чего старые данные оставались в контейнере.

**Решение:** 
1. ✅ Создан `.dockerignore` в `gcp_deployment/` - исключает `results/predictions/`, `results/inference/`, `results/visualization/`
2. ✅ Обновлен `Dockerfile` - директории создаются пустыми
3. ✅ Обновлен `prepare_for_deployment.sh` - добавляет исключения в `.dockerignore`

**Важно:** Теперь при использовании любого из способов:
- Архив (`create_full_deployment_package.sh`) создаст `.dockerignore` с правильными исключениями
- Синхронизация (`full_sync_and_deploy.sh`) скопирует `.dockerignore` на сервер

---

## Рекомендуемый процесс

### Вариант A: Быстрая синхронизация (рекомендуется)
```bash
cd /mnt/ai/cnn/sc/gcp_deployment
./full_sync_and_deploy.sh ai8049520@136.116.116.95 --deploy
```

### Вариант B: С архивом (если нужен полный контроль)
```bash
cd /mnt/ai/cnn/sc/gcp_deployment
./create_full_deployment_package.sh
scp /mnt/ai/cnn/sc/deployment_full.tar.gz ai8049520@136.116.116.95:~
ssh ai8049520@136.116.116.95 "cd ~ && tar -xzf deployment_full.tar.gz && cd deployment_full && ./deploy_gcp.sh"
```

---

## Где что находится

### Локально (ноутбук):
- Исходный код: `/mnt/ai/cnn/sc/`
- Скрипты развертывания: `/mnt/ai/cnn/sc/gcp_deployment/`
- Архив (если создан): `/mnt/ai/cnn/sc/deployment_full.tar.gz`

### На сервере:
- Развернутый код: `~/scalepathology/`
- Архив (если скопирован): `~/deployment_full.tar.gz` или `~/deployment_full/`

### В Cloud Run:
- Docker образ собирается из `~/scalepathology/` на сервере
- Используется `Dockerfile` из `gcp_deployment/Dockerfile`
- `.dockerignore` исключает `results/predictions/` и `results/inference/` из образа



