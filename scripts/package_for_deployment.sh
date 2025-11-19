#!/bin/bash
# Скрипт для создания архива проекта для развертывания в Google Cloud

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}📦 Создание архива для развертывания${NC}"
echo "======================================"
echo ""

# Имя архива
ARCHIVE_NAME="dashboard_deployment_$(date +%Y%m%d_%H%M%S).tar.gz"
TEMP_DIR=$(mktemp -d)
ARCHIVE_DIR="$TEMP_DIR/dashboard_deployment"

echo -e "${YELLOW}📁 Создание временной директории...${NC}"
mkdir -p "$ARCHIVE_DIR"

# Копируем необходимые файлы
echo -e "${YELLOW}📋 Копирование файлов...${NC}"

# Основные файлы приложения
cp dashboard_minimal.py "$ARCHIVE_DIR/"
cp requirements_dashboard_minimal.txt "$ARCHIVE_DIR/"
cp Dockerfile.dashboard "$ARCHIVE_DIR/"
cp docker-compose.yml "$ARCHIVE_DIR/"
cp cloudbuild.yaml "$ARCHIVE_DIR/"

# Конфигурационные файлы
cp .dockerignore "$ARCHIVE_DIR/" 2>/dev/null || true
cp .gcloudignore "$ARCHIVE_DIR/" 2>/dev/null || true

# Конфигурация Streamlit
mkdir -p "$ARCHIVE_DIR/.streamlit"
cp .streamlit/config.toml "$ARCHIVE_DIR/.streamlit/" 2>/dev/null || true

# Скрипты
mkdir -p "$ARCHIVE_DIR/scripts"
cp scripts/*.sh "$ARCHIVE_DIR/scripts/" 2>/dev/null || true

# Документация
cp README_DEPLOYMENT*.md "$ARCHIVE_DIR/" 2>/dev/null || true
cp QUICK_START_DEPLOYMENT.md "$ARCHIVE_DIR/" 2>/dev/null || true

# Создаем README для архива
cat > "$ARCHIVE_DIR/README.txt" << 'EOF'
==========================================
Dashboard Deployment Package
==========================================

Этот архив содержит все необходимое для развертывания Dashboard в Google Cloud.

СОДЕРЖИМОЕ:
- dashboard_minimal.py - приложение Dashboard
- requirements_dashboard_minimal.txt - зависимости Python
- Dockerfile.dashboard - Docker образ
- docker-compose.yml - для локального тестирования
- cloudbuild.yaml - конфигурация Cloud Build
- scripts/ - скрипты для развертывания
- .streamlit/config.toml - конфигурация Streamlit

СПОСОБЫ РАЗВЕРТЫВАНИЯ:

1. Через Cloud Shell (рекомендуется):
   - Загрузите архив в Cloud Shell
   - Распакуйте: tar -xzf dashboard_deployment_*.tar.gz
   - Запустите: ./scripts/setup_gcp.sh && ./scripts/deploy_gcp.sh

2. Через Cloud Storage:
   - Загрузите архив в Cloud Storage
   - Используйте Cloud Build для автоматического развертывания

3. Через Cloud Build напрямую:
   - Загрузите архив в Cloud Storage
   - Запустите Cloud Build с cloudbuild.yaml

ПОДРОБНАЯ ДОКУМЕНТАЦИЯ:
См. README_DEPLOYMENT_RU.md или README_DEPLOYMENT.md
EOF

# Создаем архив
echo -e "${YELLOW}🗜️  Создание архива...${NC}"
cd "$TEMP_DIR"
tar -czf "$ARCHIVE_NAME" dashboard_deployment/

# Перемещаем архив в текущую директорию
mv "$ARCHIVE_NAME" "$(pwd)/"

# Очистка
rm -rf "$TEMP_DIR"

ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)

echo ""
echo -e "${GREEN}✅ Архив успешно создан!${NC}"
echo ""
echo "📦 Файл: $ARCHIVE_NAME"
echo "📊 Размер: $ARCHIVE_SIZE"
echo ""
echo "💡 Следующие шаги:"
echo "   1. Загрузите архив в Google Cloud Storage или Cloud Shell"
echo "   2. Распакуйте: tar -xzf $ARCHIVE_NAME"
echo "   3. Следуйте инструкциям в README_DEPLOYMENT_RU.md"

