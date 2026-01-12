#!/bin/bash
# Полная синхронизация кода с ноутбука на сервер и подготовка к деплою
# Использование: ./full_sync_and_deploy.sh [SERVER_USER@SERVER_IP] [--deploy]

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🔄 Полная синхронизация кода и подготовка к деплою${NC}"
echo "=================================================================="
echo ""

# Параметры
SERVER="${1:-ai8049520@35.225.95.250}"
AUTO_DEPLOY="${2:-}"

SOURCE_DIR="${SOURCE_DIR:-/mnt/ai/cnn/sc}"
SERVER_SCALE_DIR="${SERVER_SCALE_DIR:-~/scalepathology/scale}"
SERVER_DEPLOY_DIR="${SERVER_DEPLOY_DIR:-~/scalepathology}"

echo -e "${BLUE}📋 Параметры:${NC}"
echo "   Ноутбук (источник): $SOURCE_DIR"
echo "   Сервер: $SERVER"
echo "   Директория на сервере: $SERVER_DEPLOY_DIR"
echo ""

# Проверка наличия исходной директории
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}❌ Ошибка: исходная директория не найдена: $SOURCE_DIR${NC}"
    exit 1
fi

# Проверка подключения к серверу
echo -e "${YELLOW}🔍 Проверка подключения к серверу...${NC}"
if ! ssh -o ConnectTimeout=5 "$SERVER" "echo 'Подключение успешно'" 2>/dev/null; then
    echo -e "${RED}❌ Ошибка: не удалось подключиться к серверу $SERVER${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Подключение установлено${NC}"
echo ""

# Проверка наличия scalepathology на сервере
echo -e "${YELLOW}🔍 Проверка директории scalepathology на сервере...${NC}"
if ! ssh "$SERVER" "[ -d $SERVER_DEPLOY_DIR ]"; then
    echo -e "${RED}❌ Ошибка: директория $SERVER_DEPLOY_DIR не найдена на сервере${NC}"
    echo "   Сначала выполните: ./prepare_for_deployment.sh на сервере"
    exit 1
fi
echo -e "${GREEN}✅ Директория найдена${NC}"
echo ""

# Синхронизация scale/ - весь код dashboard
echo -e "${YELLOW}📤 Синхронизация кода scale/...${NC}"
echo ""

# Создаем список файлов для синхронизации
# ТОЛЬКО модули, которые реально используются в dashboard
FILES_TO_SYNC=(
    "scale/dashboard.py"
    "scale/dashboard_common.py"
    "scale/dashboard_experiment_selector.py"
    "scale/gdrive_integration.py"
    "scale/gcs_integration.py"
    "scale/aggregate.py"
    "scale/spectral_analysis.py"
    "scale/domain.py"
    "scale/scale_comparison.py"
    "scale/pca_scoring.py"
    "scale/preprocessing.py"
    "scale/eda.py"
    "scale/__init__.py"
)

# Проверяем наличие rsync (если нет - используем scp)
USE_RSYNC=false
if command -v rsync &> /dev/null && ssh "$SERVER" "command -v rsync" &> /dev/null; then
    USE_RSYNC=true
fi

# Синхронизируем каждый файл используя rsync (если доступен) или scp
SYNCED_COUNT=0
for file in "${FILES_TO_SYNC[@]}"; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        echo -e "${GREEN}   ✅ Синхронизирую $file...${NC}"
        if [ "$USE_RSYNC" = true ]; then
            # Используем rsync с правильными правами
            rsync -avz --chmod=644 "$SOURCE_DIR/$file" "$SERVER:$SERVER_DEPLOY_DIR/$file" >/dev/null 2>&1 && {
                SYNCED_COUNT=$((SYNCED_COUNT + 1))
            } || {
                echo -e "${YELLOW}   ⚠️  Не удалось скопировать $file (rsync)${NC}"
            }
        else
            # Fallback на scp
            scp "$SOURCE_DIR/$file" "$SERVER:$SERVER_DEPLOY_DIR/$file" >/dev/null 2>&1 && {
                SYNCED_COUNT=$((SYNCED_COUNT + 1))
                # Устанавливаем права после копирования
                ssh "$SERVER" "chmod 644 $SERVER_DEPLOY_DIR/$file" 2>/dev/null || true
            } || {
                echo -e "${YELLOW}   ⚠️  Не удалось скопировать $file (scp)${NC}"
            }
        fi
    fi
done

# Синхронизируем остальные необходимые файлы из scale/ (только те, что нужны для dashboard)
ADDITIONAL_FILES=(
    "scale/cfg"
)

# Синхронизация results/inference и results/predictions для SVM
echo -e "${YELLOW}📊 Синхронизация данных для SVM (inference и predictions)...${NC}"

# Синхронизация results/inference
if [ -d "$SOURCE_DIR/results/inference" ]; then
    echo -e "${GREEN}   ✅ Синхронизирую results/inference/...${NC}"
    ssh "$SERVER" "mkdir -p $SERVER_DEPLOY_DIR/results/inference" 2>/dev/null || true
    if [ "$USE_RSYNC" = true ]; then
        rsync -avz --chmod=644 --include="*/" --include="*.json" --exclude="*" "$SOURCE_DIR/results/inference/" "$SERVER:$SERVER_DEPLOY_DIR/results/inference/" >/dev/null 2>&1 && {
            INFERENCE_COUNT=$(ls -1 "$SOURCE_DIR/results/inference"/*.json 2>/dev/null | wc -l)
            echo -e "${GREEN}      ✅ Синхронизировано $INFERENCE_COUNT JSON файлов из inference${NC}"
        } || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать inference файлы${NC}"
        }
    else
        # Fallback на scp
        scp "$SOURCE_DIR/results/inference"/*.json "$SERVER:$SERVER_DEPLOY_DIR/results/inference/" >/dev/null 2>&1 && {
            INFERENCE_COUNT=$(ls -1 "$SOURCE_DIR/results/inference"/*.json 2>/dev/null | wc -l)
            echo -e "${GREEN}      ✅ Синхронизировано $INFERENCE_COUNT JSON файлов из inference${NC}"
            ssh "$SERVER" "chmod 644 $SERVER_DEPLOY_DIR/results/inference/*.json" 2>/dev/null || true
        } || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать inference файлы${NC}"
        }
    fi
else
    echo -e "${YELLOW}   ⚠️  Директория results/inference/ не найдена${NC}"
fi

# Синхронизация results/predictions
if [ -d "$SOURCE_DIR/results/predictions" ]; then
    echo -e "${GREEN}   ✅ Синхронизирую results/predictions/...${NC}"
    ssh "$SERVER" "mkdir -p $SERVER_DEPLOY_DIR/results/predictions" 2>/dev/null || true
    if [ "$USE_RSYNC" = true ]; then
        rsync -avz --chmod=644 --include="*/" --include="*.json" --exclude="*" "$SOURCE_DIR/results/predictions/" "$SERVER:$SERVER_DEPLOY_DIR/results/predictions/" >/dev/null 2>&1 && {
            PREDICTIONS_COUNT=$(ls -1 "$SOURCE_DIR/results/predictions"/*.json 2>/dev/null | wc -l)
            echo -e "${GREEN}      ✅ Синхронизировано $PREDICTIONS_COUNT JSON файлов из predictions${NC}"
        } || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать predictions файлы${NC}"
        }
    else
        # Fallback на scp
        scp "$SOURCE_DIR/results/predictions"/*.json "$SERVER:$SERVER_DEPLOY_DIR/results/predictions/" >/dev/null 2>&1 && {
            PREDICTIONS_COUNT=$(ls -1 "$SOURCE_DIR/results/predictions"/*.json 2>/dev/null | wc -l)
            echo -e "${GREEN}      ✅ Синхронизировано $PREDICTIONS_COUNT JSON файлов из predictions${NC}"
            ssh "$SERVER" "chmod 644 $SERVER_DEPLOY_DIR/results/predictions/*.json" 2>/dev/null || true
        } || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать predictions файлы${NC}"
        }
    fi
else
    echo -e "${YELLOW}   ⚠️  Директория results/predictions/ не найдена${NC}"
fi

echo ""

# Опционально: синхронизация моделей (если нужен инференс на сервере)
# По умолчанию НЕ синхронизируем, так как dashboard обычно только читает готовые predictions
SYNC_MODELS=${SYNC_MODELS:-false}
if [ "$SYNC_MODELS" = "true" ]; then
    echo -e "${YELLOW}🤖 Синхронизация моделей (опционально)...${NC}"
        if [ -d "$SOURCE_DIR/models" ]; then
        echo -e "${GREEN}   ✅ Синхронизирую models/...${NC}"
        ssh "$SERVER" "mkdir -p $SERVER_DEPLOY_DIR/models" 2>/dev/null || true
        if [ "$USE_RSYNC" = true ]; then
            rsync -avz --chmod=644 "$SOURCE_DIR/models/" "$SERVER:$SERVER_DEPLOY_DIR/models/" >/dev/null 2>&1 || {
                echo -e "${YELLOW}   ⚠️  Не удалось скопировать модели${NC}"
            }
        else
            scp -r "$SOURCE_DIR/models"/* "$SERVER:$SERVER_DEPLOY_DIR/models/" >/dev/null 2>&1 || {
                echo -e "${YELLOW}   ⚠️  Не удалось скопировать модели${NC}"
            }
        fi
    else
        echo -e "${YELLOW}   ⚠️  Директория models/ не найдена${NC}"
    fi
fi

for item in "${ADDITIONAL_FILES[@]}"; do
    if [ -d "$SOURCE_DIR/$item" ] || [ -f "$SOURCE_DIR/$item" ]; then
        echo -e "${GREEN}   ✅ Синхронизирую $item...${NC}"
        if [ -d "$SOURCE_DIR/$item" ]; then
            # Это директория
            ssh "$SERVER" "mkdir -p $SERVER_DEPLOY_DIR/$item" 2>/dev/null || true
            if [ "$USE_RSYNC" = true ]; then
                rsync -avz --chmod=644 "$SOURCE_DIR/$item/" "$SERVER:$SERVER_DEPLOY_DIR/$item/" >/dev/null 2>&1 || true
            else
                scp -r "$SOURCE_DIR/$item"/* "$SERVER:$SERVER_DEPLOY_DIR/$item/" >/dev/null 2>&1 || true
            fi
        else
            # Это файл
            if [ "$USE_RSYNC" = true ]; then
                rsync -avz --chmod=644 "$SOURCE_DIR/$item" "$SERVER:$SERVER_DEPLOY_DIR/$item" >/dev/null 2>&1 || true
            else
                scp "$SOURCE_DIR/$item" "$SERVER:$SERVER_DEPLOY_DIR/$item" >/dev/null 2>&1 && {
                    ssh "$SERVER" "chmod 644 $SERVER_DEPLOY_DIR/$item" 2>/dev/null || true
                } || true
            fi
        fi
    fi
done

echo ""
echo -e "${GREEN}   ✅ Синхронизировано файлов: $SYNCED_COUNT${NC}"
echo ""

# Синхронизация requirements.txt
echo -e "${YELLOW}📦 Синхронизация зависимостей...${NC}"
if [ -f "$SOURCE_DIR/requirements.txt" ]; then
    echo -e "${GREEN}   ✅ Синхронизирую requirements.txt...${NC}"
    if [ "$USE_RSYNC" = true ]; then
        rsync -avz --chmod=644 "$SOURCE_DIR/requirements.txt" "$SERVER:$SERVER_DEPLOY_DIR/requirements.txt" >/dev/null 2>&1 || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать requirements.txt${NC}"
        }
    else
        scp "$SOURCE_DIR/requirements.txt" "$SERVER:$SERVER_DEPLOY_DIR/requirements.txt" >/dev/null 2>&1 && {
            ssh "$SERVER" "chmod 644 $SERVER_DEPLOY_DIR/requirements.txt" 2>/dev/null || true
        } || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать requirements.txt${NC}"
        }
    fi
fi

# Синхронизация credentials для Google Drive и GCS (копируем в проект для Docker)
echo -e "${YELLOW}🔐 Синхронизация credentials в проект (для Docker образа)...${NC}"

CREDS_SYNCED=false

# Google Drive credentials - копируем в .config/gdrive/ в проекте
GDRIVE_CREDS_SOURCE="/mnt/ai/cnn/.config/gdrive/credentials.json"
if [ -f "$GDRIVE_CREDS_SOURCE" ]; then
    echo -e "${GREEN}   ✅ Найден Google Drive credentials.json${NC}"
    # Создаем директорию в проекте на сервере (чтобы попала в Docker образ)
    ssh "$SERVER" "mkdir -p $SERVER_DEPLOY_DIR/.config/gdrive" 2>/dev/null || true
    # Копируем файл в проект
    if [ "$USE_RSYNC" = true ]; then
        rsync -avz --chmod=600 "$GDRIVE_CREDS_SOURCE" "$SERVER:$SERVER_DEPLOY_DIR/.config/gdrive/credentials.json" >/dev/null 2>&1 && {
            echo -e "${GREEN}   ✅ Синхронизирован Google Drive credentials.json в проект${NC}"
            CREDS_SYNCED=true
        } || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать Google Drive credentials${NC}"
        }
    else
        scp "$GDRIVE_CREDS_SOURCE" "$SERVER:$SERVER_DEPLOY_DIR/.config/gdrive/credentials.json" >/dev/null 2>&1 && {
            ssh "$SERVER" "chmod 600 $SERVER_DEPLOY_DIR/.config/gdrive/credentials.json" 2>/dev/null || true
            echo -e "${GREEN}   ✅ Синхронизирован Google Drive credentials.json в проект${NC}"
            CREDS_SYNCED=true
        } || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать Google Drive credentials${NC}"
        }
    fi
else
    echo -e "${YELLOW}   ⚠️  Google Drive credentials.json не найден: $GDRIVE_CREDS_SOURCE${NC}"
fi

# GCS service account key - копируем в .config/gcs/ в проекте
GCS_CREDS_SOURCE="/mnt/ai/cnn/.config/gcs/service-account-key.json"
if [ -f "$GCS_CREDS_SOURCE" ]; then
    echo -e "${GREEN}   ✅ Найден GCS service-account-key.json${NC}"
    # Создаем директорию в проекте на сервере (чтобы попала в Docker образ)
    ssh "$SERVER" "mkdir -p $SERVER_DEPLOY_DIR/.config/gcs" 2>/dev/null || true
    # Копируем файл в проект
    if [ "$USE_RSYNC" = true ]; then
        rsync -avz --chmod=600 "$GCS_CREDS_SOURCE" "$SERVER:$SERVER_DEPLOY_DIR/.config/gcs/service-account-key.json" >/dev/null 2>&1 && {
            echo -e "${GREEN}   ✅ Синхронизирован GCS service-account-key.json в проект${NC}"
            CREDS_SYNCED=true
        } || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать GCS credentials${NC}"
        }
    else
        scp "$GCS_CREDS_SOURCE" "$SERVER:$SERVER_DEPLOY_DIR/.config/gcs/service-account-key.json" >/dev/null 2>&1 && {
            ssh "$SERVER" "chmod 600 $SERVER_DEPLOY_DIR/.config/gcs/service-account-key.json" 2>/dev/null || true
            echo -e "${GREEN}   ✅ Синхронизирован GCS service-account-key.json в проект${NC}"
            CREDS_SYNCED=true
        } || {
            echo -e "${YELLOW}   ⚠️  Не удалось скопировать GCS credentials${NC}"
        }
    fi
else
    echo -e "${YELLOW}   ⚠️  GCS service-account-key.json не найден: $GCS_CREDS_SOURCE${NC}"
fi

if [ "$CREDS_SYNCED" = true ]; then
    echo -e "${GREEN}   ✅ Credentials будут включены в Docker образ${NC}"
fi
echo ""

# Синхронизация Dockerfile если изменился
echo -e "${YELLOW}🐳 Проверка Dockerfile...${NC}"
if [ -f "$SOURCE_DIR/gcp_deployment/Dockerfile" ]; then
    LOCAL_HASH=$(md5sum "$SOURCE_DIR/gcp_deployment/Dockerfile" 2>/dev/null | awk '{print $1}' || echo "")
    REMOTE_HASH=$(ssh "$SERVER" "md5sum $SERVER_DEPLOY_DIR/Dockerfile 2>/dev/null | awk '{print \$1}'" || echo "")
    
    if [ "$LOCAL_HASH" != "$REMOTE_HASH" ] && [ -n "$LOCAL_HASH" ]; then
        echo -e "${GREEN}   ✅ Обновляю Dockerfile...${NC}"
        if [ "$USE_RSYNC" = true ]; then
            rsync -avz --chmod=644 "$SOURCE_DIR/gcp_deployment/Dockerfile" "$SERVER:$SERVER_DEPLOY_DIR/Dockerfile" >/dev/null 2>&1 || true
        else
            scp "$SOURCE_DIR/gcp_deployment/Dockerfile" "$SERVER:$SERVER_DEPLOY_DIR/Dockerfile" >/dev/null 2>&1 && {
                ssh "$SERVER" "chmod 644 $SERVER_DEPLOY_DIR/Dockerfile" 2>/dev/null || true
            } || true
        fi
    else
        echo -e "${BLUE}   ⏭️  Dockerfile актуален${NC}"
    fi
fi

# Синхронизация deploy_gcp.sh если изменился
if [ -f "$SOURCE_DIR/gcp_deployment/deploy_gcp.sh" ]; then
    LOCAL_HASH=$(md5sum "$SOURCE_DIR/gcp_deployment/deploy_gcp.sh" 2>/dev/null | awk '{print $1}' || echo "")
    REMOTE_HASH=$(ssh "$SERVER" "md5sum $SERVER_DEPLOY_DIR/deploy_gcp.sh 2>/dev/null | awk '{print \$1}'" || echo "")
    
    if [ "$LOCAL_HASH" != "$REMOTE_HASH" ] && [ -n "$LOCAL_HASH" ]; then
        echo -e "${GREEN}   ✅ Обновляю deploy_gcp.sh...${NC}"
        if [ "$USE_RSYNC" = true ]; then
            rsync -avz --chmod=755 "$SOURCE_DIR/gcp_deployment/deploy_gcp.sh" "$SERVER:$SERVER_DEPLOY_DIR/deploy_gcp.sh" >/dev/null 2>&1 || true
        else
            scp "$SOURCE_DIR/gcp_deployment/deploy_gcp.sh" "$SERVER:$SERVER_DEPLOY_DIR/deploy_gcp.sh" >/dev/null 2>&1 && {
                ssh "$SERVER" "chmod 755 $SERVER_DEPLOY_DIR/deploy_gcp.sh" 2>/dev/null || true
            } || true
        fi
    fi
fi

# Синхронизация .dockerignore (ВАЖНО: исключает results/predictions и results/inference из образа)
echo -e "${YELLOW}🐳 Проверка .dockerignore...${NC}"
if [ -f "$SOURCE_DIR/gcp_deployment/.dockerignore" ]; then
    LOCAL_HASH=$(md5sum "$SOURCE_DIR/gcp_deployment/.dockerignore" 2>/dev/null | awk '{print $1}' || echo "")
    REMOTE_HASH=$(ssh "$SERVER" "md5sum $SERVER_DEPLOY_DIR/.dockerignore 2>/dev/null | awk '{print \$1}'" || echo "")
    
    if [ "$LOCAL_HASH" != "$REMOTE_HASH" ] && [ -n "$LOCAL_HASH" ]; then
        echo -e "${GREEN}   ✅ Обновляю .dockerignore...${NC}"
        if [ "$USE_RSYNC" = true ]; then
            rsync -avz --chmod=644 "$SOURCE_DIR/gcp_deployment/.dockerignore" "$SERVER:$SERVER_DEPLOY_DIR/.dockerignore" >/dev/null 2>&1 || true
        else
            scp "$SOURCE_DIR/gcp_deployment/.dockerignore" "$SERVER:$SERVER_DEPLOY_DIR/.dockerignore" >/dev/null 2>&1 && {
                ssh "$SERVER" "chmod 644 $SERVER_DEPLOY_DIR/.dockerignore" 2>/dev/null || true
            } || true
        fi
    else
        echo -e "${BLUE}   ⏭️  .dockerignore актуален${NC}"
    fi
fi


echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Синхронизация завершена!${NC}"
echo ""

# Проверка синхронизированных файлов
echo -e "${BLUE}🔍 Проверка синхронизированных файлов:${NC}"
ssh "$SERVER" "
    echo '   Файлы в scale/:'
    ls -1 $SERVER_DEPLOY_DIR/scale/*.py 2>/dev/null | wc -l | xargs echo '   - Python файлов:'
    echo ''
    echo '   Ключевые файлы:'
    [ -f $SERVER_DEPLOY_DIR/scale/dashboard.py ] && echo '   ✅ dashboard.py' || echo '   ❌ dashboard.py'
    [ -f $SERVER_DEPLOY_DIR/scale/dashboard_common.py ] && echo '   ✅ dashboard_common.py' || echo '   ❌ dashboard_common.py'
    [ -f $SERVER_DEPLOY_DIR/scale/gdrive_integration.py ] && echo '   ✅ gdrive_integration.py' || echo '   ❌ gdrive_integration.py'
    [ -f $SERVER_DEPLOY_DIR/scale/gcs_integration.py ] && echo '   ✅ gcs_integration.py' || echo '   ❌ gcs_integration.py'
    [ -f $SERVER_DEPLOY_DIR/requirements.txt ] && echo '   ✅ requirements.txt' || echo '   ❌ requirements.txt'
    [ -f $SERVER_DEPLOY_DIR/Dockerfile ] && echo '   ✅ Dockerfile' || echo '   ❌ Dockerfile'
    echo ''
    echo '   Credentials:'
    [ -f $SERVER_DEPLOY_DIR/.config/gdrive/credentials.json ] && echo '   ✅ Google Drive credentials.json' || echo '   ❌ Google Drive credentials.json'
    [ -f $SERVER_DEPLOY_DIR/.config/gcs/service-account-key.json ] && echo '   ✅ GCS service-account-key.json' || echo '   ❌ GCS service-account-key.json'
"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Автоматический деплой если указан флаг
if [ "$AUTO_DEPLOY" == "--deploy" ]; then
    echo -e "${YELLOW}🚀 Запуск автоматического деплоя...${NC}"
    echo ""
    ssh "$SERVER" "cd $SERVER_DEPLOY_DIR && ./deploy_gcp.sh <<< 'y'" 2>&1 | tail -30
else
    echo -e "${BLUE}💡 Следующие шаги на сервере:${NC}"
    echo ""
    echo "   ssh $SERVER"
    echo "   cd $SERVER_DEPLOY_DIR"
    echo "   ./deploy_gcp.sh"
    echo ""
    echo -e "${YELLOW}Или запустите автоматический деплой:${NC}"
    echo "   $0 $SERVER --deploy"
    echo ""
fi

