#!/bin/bash
# Скрипт для подготовки всех файлов для тестирования деплоймента на сервере
# Создает готовую директорию со всеми необходимыми файлами

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}📦 Подготовка файлов для тестирования деплоймента на сервере${NC}"
echo "=========================================================================="
echo ""

# Определяем директории
SOURCE_DIR="${1:-/mnt/ai/cnn/sc}"
OUTPUT_DIR="${2:-$HOME/deployment_test}"
SERVER_USER="${3:-ai8049520}"
SERVER_HOST="${4:-instance-20251117-192323}"

# Расширяем ~ до полного пути
OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"

echo -e "${BLUE}📂 Исходная директория:${NC} $SOURCE_DIR"
echo -e "${BLUE}📂 Директория для копирования:${NC} $OUTPUT_DIR"
echo -e "${BLUE}🖥️  Сервер:${NC} $SERVER_USER@$SERVER_HOST"
echo ""

# Проверка существования исходной директории
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${YELLOW}❌ Ошибка: исходная директория не найдена: $SOURCE_DIR${NC}"
    exit 1
fi

# Создаем директорию для копирования
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo -e "${YELLOW}📋 Копирование файлов...${NC}"
echo ""

# 1. Копируем весь код scale/
echo -e "${GREEN}   ✅ Копирую scale/ (код dashboard)...${NC}"
rsync -av --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    "$SOURCE_DIR/scale/" "$OUTPUT_DIR/scale/" 2>/dev/null || {
    echo "   ⚠️  rsync недоступен, используем cp..."
    mkdir -p "$OUTPUT_DIR/scale"
    cp -r "$SOURCE_DIR/scale"/* "$OUTPUT_DIR/scale/" 2>/dev/null || true
}

# 2. Копируем requirements.txt
echo -e "${GREEN}   ✅ Копирую requirements.txt...${NC}"
if [ -f "$SOURCE_DIR/requirements.txt" ]; then
    cp "$SOURCE_DIR/requirements.txt" "$OUTPUT_DIR/"
fi

# 3. Копируем dashboard_minimal.py (если используется)
if [ -f "$SOURCE_DIR/dashboard_minimal.py" ]; then
    echo -e "${GREEN}   ✅ Копирую dashboard_minimal.py...${NC}"
    cp "$SOURCE_DIR/dashboard_minimal.py" "$OUTPUT_DIR/"
fi

# 4. Копируем requirements_dashboard_minimal.txt
if [ -f "$SOURCE_DIR/requirements_dashboard_minimal.txt" ]; then
    echo -e "${GREEN}   ✅ Копирую requirements_dashboard_minimal.txt...${NC}"
    cp "$SOURCE_DIR/requirements_dashboard_minimal.txt" "$OUTPUT_DIR/"
fi

# 5. Копируем Dockerfile
echo -e "${GREEN}   ✅ Копирую Dockerfile...${NC}"
if [ -f "$SOURCE_DIR/Dockerfile.dashboard" ]; then
    cp "$SOURCE_DIR/Dockerfile.dashboard" "$OUTPUT_DIR/Dockerfile"
elif [ -f "$SOURCE_DIR/deployment_package/Dockerfile" ]; then
    cp "$SOURCE_DIR/deployment_package/Dockerfile" "$OUTPUT_DIR/"
elif [ -f "$SOURCE_DIR/deployment_full/Dockerfile" ]; then
    cp "$SOURCE_DIR/deployment_full/Dockerfile" "$OUTPUT_DIR/"
fi

# 6. Копируем cloudbuild.yaml (если есть)
if [ -f "$SOURCE_DIR/cloudbuild.yaml" ]; then
    echo -e "${GREEN}   ✅ Копирую cloudbuild.yaml...${NC}"
    cp "$SOURCE_DIR/cloudbuild.yaml" "$OUTPUT_DIR/"
fi

# 7. Копируем скрипты деплоймента
echo -e "${GREEN}   ✅ Копирую скрипты деплоймента...${NC}"
mkdir -p "$OUTPUT_DIR/scripts"
if [ -f "$SOURCE_DIR/deployment_package/deploy_gcp.sh" ]; then
    cp "$SOURCE_DIR/deployment_package/deploy_gcp.sh" "$OUTPUT_DIR/"
    chmod +x "$OUTPUT_DIR/deploy_gcp.sh"
fi
if [ -f "$SOURCE_DIR/deployment_package/prepare_for_deployment.sh" ]; then
    cp "$SOURCE_DIR/deployment_package/prepare_for_deployment.sh" "$OUTPUT_DIR/"
    chmod +x "$OUTPUT_DIR/prepare_for_deployment.sh"
fi
if [ -f "$SOURCE_DIR/deployment_package/setup_gcp_apis.sh" ]; then
    cp "$SOURCE_DIR/deployment_package/setup_gcp_apis.sh" "$OUTPUT_DIR/"
    chmod +x "$OUTPUT_DIR/setup_gcp_apis.sh"
fi

# 8. Копируем .streamlit конфигурацию
if [ -d "$SOURCE_DIR/.streamlit" ]; then
    echo -e "${GREEN}   ✅ Копирую .streamlit/...${NC}"
    mkdir -p "$OUTPUT_DIR/.streamlit"
    cp -r "$SOURCE_DIR/.streamlit"/* "$OUTPUT_DIR/.streamlit/" 2>/dev/null || true
else
    # Создаем базовую конфигурацию
    echo -e "${GREEN}   ✅ Создаю базовую конфигурацию .streamlit/...${NC}"
    mkdir -p "$OUTPUT_DIR/.streamlit"
    cat > "$OUTPUT_DIR/.streamlit/config.toml" << 'EOF'
[server]
port = 8080
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOF
fi

# 9. Копируем experiments (только JSON и CSV)
if [ -d "$SOURCE_DIR/experiments" ]; then
    echo -e "${GREEN}   ✅ Копирую experiments/ (JSON и CSV)...${NC}"
    mkdir -p "$OUTPUT_DIR/experiments"
    for exp_dir in "$SOURCE_DIR/experiments"/*/; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            if [ "$exp_name" != "archive" ]; then
                mkdir -p "$OUTPUT_DIR/experiments/$exp_name"
                find "$exp_dir" -maxdepth 1 -name '*.json' -exec cp {} "$OUTPUT_DIR/experiments/$exp_name/" \; 2>/dev/null || true
                find "$exp_dir" -maxdepth 1 -name '*.csv' -exec cp {} "$OUTPUT_DIR/experiments/$exp_name/" \; 2>/dev/null || true
                find "$exp_dir" -maxdepth 1 -name '*.pkl' -exec cp {} "$OUTPUT_DIR/experiments/$exp_name/" \; 2>/dev/null || true
            fi
        fi
    done
fi

# 10. Копируем results/inference (JSON файлы)
if [ -d "$SOURCE_DIR/results/inference" ]; then
    echo -e "${GREEN}   ✅ Копирую results/inference/*.json...${NC}"
    mkdir -p "$OUTPUT_DIR/results/inference"
    find "$SOURCE_DIR/results/inference" -maxdepth 1 -name '*.json' -exec cp {} "$OUTPUT_DIR/results/inference/" \; 2>/dev/null || true
fi

# 11. Создаем необходимые директории
echo -e "${GREEN}   ✅ Создаю структуру директорий...${NC}"
mkdir -p "$OUTPUT_DIR/results/predictions"
mkdir -p "$OUTPUT_DIR/results/visualization"

# 12. Создаем .dockerignore
echo -e "${GREEN}   ✅ Создаю .dockerignore...${NC}"
cat > "$OUTPUT_DIR/.dockerignore" << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
venv/
env/
ENV/
.vscode/
.idea/
*.swp
*.swo
.git/
.gitignore
docs/
*.md
README*
tests/
test_*.py
*_test.py
*.tar.gz
*.zip
*.log
*.tiff
*.tif
*.TIFF
*.TIF
*.svs
*.ndpi
wsi/
notebook/
archive/
model_development/
gcp_deployment/
data/
*.h5
*.hdf5
EOF

# 13. Создаем .gcloudignore
echo -e "${GREEN}   ✅ Создаю .gcloudignore...${NC}"
cat > "$OUTPUT_DIR/.gcloudignore" << 'EOF'
.gcloudignore
.git/
.gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
venv/
env/
ENV/
.vscode/
.idea/
*.swp
*.swo
*.log
*.md
README*
docs/
tests/
test_*.py
*_test.py
notebook/
archive/
model_development/
gcp_deployment/
*.tar.gz
*.zip
*.tiff
*.tif
*.TIFF
*.TIF
*.svs
*.ndpi
wsi/
EOF

# 14. Создаем README с инструкциями
echo -e "${GREEN}   ✅ Создаю README с инструкциями...${NC}"
cat > "$OUTPUT_DIR/README_DEPLOYMENT.md" << EOF
# Инструкция по развертыванию на сервере

## 📦 Подготовка завершена

Все необходимые файлы скопированы в эту директорию.

## 🚀 Быстрый старт

### Вариант 1: Использование prepare_for_deployment.sh (рекомендуется)

Если исходный проект уже есть на сервере в \`/mnt/ai/cnn/sc\`:

\`\`\`bash
cd $OUTPUT_DIR
chmod +x prepare_for_deployment.sh
./prepare_for_deployment.sh /mnt/ai/cnn/sc ~/scalepathology
cd ~/scalepathology
./deploy_gcp.sh
\`\`\`

### Вариант 2: Прямое использование подготовленных файлов

Если вы скопировали эту директорию на сервер:

\`\`\`bash
# На сервере
cd ~/deployment_test  # или путь куда скопировали

# Убедитесь что Dockerfile существует
if [ ! -f "Dockerfile" ]; then
    cp Dockerfile.dashboard Dockerfile 2>/dev/null || true
fi

# Развертывание
chmod +x deploy_gcp.sh
./deploy_gcp.sh
\`\`\`

## 📋 Что включено

- ✅ \`scale/\` - весь код dashboard
- ✅ \`requirements.txt\` - зависимости Python
- ✅ \`Dockerfile\` - конфигурация Docker образа
- ✅ \`deploy_gcp.sh\` - скрипт развертывания
- ✅ \`experiments/\` - эксперименты (JSON, CSV, PKL)
- ✅ \`results/inference/\` - JSON файлы с предсказаниями
- ✅ \`.streamlit/config.toml\` - конфигурация Streamlit

## 🔧 Настройка перед деплоем

1. Убедитесь что проект установлен:
   \`\`\`bash
   gcloud config set project scalepathology
   \`\`\`

2. Проверьте авторизацию:
   \`\`\`bash
   gcloud auth list
   \`\`\`

3. Включите необходимые API (если еще не включены):
   \`\`\`bash
   ./setup_gcp_apis.sh
   \`\`\`

## 📝 Примечания

- Credentials для Google Drive и GCS должны быть настроены отдельно
- Убедитесь что на сервере есть доступ к Google Cloud
- Проверьте что все необходимые API включены в проекте

## 🆘 Помощь

Если возникли проблемы, проверьте:
- Логи: \`gcloud run services logs read dashboard --region us-central1 --follow\`
- Статус сервиса: \`gcloud run services describe dashboard --region us-central1\`
EOF

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Подготовка завершена!${NC}"
echo ""
echo -e "${BLUE}📁 Файлы подготовлены в:${NC} $OUTPUT_DIR"
echo ""

# Статистика
echo -e "${YELLOW}📊 Статистика:${NC}"
echo "   - Python файлов: $(find "$OUTPUT_DIR/scale" -name '*.py' -type f 2>/dev/null | wc -l)"
echo "   - JSON файлов: $(find "$OUTPUT_DIR" -name '*.json' -type f 2>/dev/null | wc -l)"
echo "   - CSV файлов: $(find "$OUTPUT_DIR/experiments" -name '*.csv' -type f 2>/dev/null | wc -l)"
echo "   - Экспериментов: $(find "$OUTPUT_DIR/experiments" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
echo ""

# Команды для копирования на сервер
echo -e "${BLUE}📤 Команды для копирования на сервер:${NC}"
echo ""
echo -e "${YELLOW}Вариант 1: Использование rsync (рекомендуется):${NC}"
echo "   rsync -avz --progress $OUTPUT_DIR/ $SERVER_USER@$SERVER_HOST:~/deployment_test/"
echo ""
echo -e "${YELLOW}Вариант 2: Использование scp (для архива):${NC}"
echo "   cd $OUTPUT_DIR"
echo "   tar -czf deployment_test.tar.gz ."
echo "   scp deployment_test.tar.gz $SERVER_USER@$SERVER_HOST:~/"
echo "   # На сервере: tar -xzf deployment_test.tar.gz"
echo ""
echo -e "${YELLOW}Вариант 3: Использование prepare_for_deployment.sh на сервере:${NC}"
echo "   scp $OUTPUT_DIR/prepare_for_deployment.sh $SERVER_USER@$SERVER_HOST:~/"
echo "   ssh $SERVER_USER@$SERVER_HOST"
echo "   chmod +x ~/prepare_for_deployment.sh"
echo "   ~/prepare_for_deployment.sh /mnt/ai/cnn/sc ~/scalepathology"
echo ""

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

