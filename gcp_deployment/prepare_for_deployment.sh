#!/bin/bash
# Скрипт для подготовки проекта к развертыванию в Google Cloud Run
# Минимальный набор файлов для dashboard: только код, JSON и конфигурации

set -e

echo "📦 Подготовка проекта к развертыванию (минимальный набор)"
echo "=========================================================="
echo ""

# Определяем директории
SOURCE_DIR="${1:-/mnt/ai/cnn/sc}"
DEPLOY_DIR="${2:-$HOME/scalepathology}"

# Расширяем ~ до полного пути
DEPLOY_DIR="${DEPLOY_DIR/#\~/$HOME}"

echo "📂 Исходная директория: $SOURCE_DIR"
echo "📂 Директория для деплоя: $DEPLOY_DIR"
echo ""

# Проверка существования исходной директории
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Ошибка: исходная директория не найдена: $SOURCE_DIR"
    echo ""
    echo "Использование:"
    echo "  $0 [SOURCE_DIR] [DEPLOY_DIR]"
    echo ""
    echo "Примеры:"
    echo "  $0 /mnt/ai/cnn/sc ~/scalepathology"
    echo "  $0 ~/sc ~/scalepathology"
    exit 1
fi

# Создаем директорию для деплоя если её нет
mkdir -p "$DEPLOY_DIR"

echo "📋 Копирование файлов (только необходимое для dashboard)..."
echo ""

# 1. Копируем основной код приложения scale/
if [ -d "$SOURCE_DIR/scale" ]; then
    echo "   ✅ Копирую scale/ (код dashboard)..."
    rsync -av --delete \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        "$SOURCE_DIR/scale/" "$DEPLOY_DIR/scale/" 2>/dev/null || {
        echo "   ⚠️  rsync недоступен, используем cp..."
        mkdir -p "$DEPLOY_DIR/scale"
        find "$SOURCE_DIR/scale" -type f -name '*.py' -exec cp --parents {} "$DEPLOY_DIR/" \; 2>/dev/null || true
        find "$SOURCE_DIR/scale" -type f -name '*.json' -exec cp --parents {} "$DEPLOY_DIR/" \; 2>/dev/null || true
    }
else
    echo "   ❌ Директория scale/ не найдена!"
    exit 1
fi

# 2. Копируем requirements.txt
if [ -f "$SOURCE_DIR/requirements.txt" ]; then
    echo "   ✅ Копирую requirements.txt ..."
    cp "$SOURCE_DIR/requirements.txt" "$DEPLOY_DIR/"
else
    echo "   ⚠️  requirements.txt не найден"
fi

# 3. Копируем JSON файлы из results/inference/ (нужны для dashboard)
if [ -d "$SOURCE_DIR/results/inference" ]; then
    echo "   ✅ Копирую results/inference/*.json ..."
    mkdir -p "$DEPLOY_DIR/results/inference"
    find "$SOURCE_DIR/results/inference" -maxdepth 1 -name '*.json' -exec cp {} "$DEPLOY_DIR/results/inference/" \; 2>/dev/null || true
else
    # Создаем структуру директорий даже если файлов нет
    mkdir -p "$DEPLOY_DIR/results/inference"
    mkdir -p "$DEPLOY_DIR/results/predictions"
    mkdir -p "$DEPLOY_DIR/results/visualization"
fi

# 4. Копируем эксперименты (только JSON конфигурации и CSV данные)
if [ -d "$SOURCE_DIR/experiments" ]; then
    echo "   ✅ Копирую experiments/ (только JSON и CSV)..."
    mkdir -p "$DEPLOY_DIR/experiments"
    
    # Копируем каждый эксперимент, но только нужные файлы
    for exp_dir in "$SOURCE_DIR/experiments"/*/; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            echo "      📁 $exp_name"
            mkdir -p "$DEPLOY_DIR/experiments/$exp_name"
            
            # Копируем JSON конфигурации
            find "$exp_dir" -maxdepth 1 -name '*.json' -exec cp {} "$DEPLOY_DIR/experiments/$exp_name/" \; 2>/dev/null || true
            
            # Копируем CSV файлы с данными
            find "$exp_dir" -maxdepth 1 -name '*.csv' -exec cp {} "$DEPLOY_DIR/experiments/$exp_name/" \; 2>/dev/null || true
            
            # Копируем модели если есть (pkl файлы)
            find "$exp_dir" -maxdepth 1 -name '*.pkl' -exec cp {} "$DEPLOY_DIR/experiments/$exp_name/" \; 2>/dev/null || true
            
            # Копируем scale/cfg/ если есть
            if [ -d "$exp_dir/scale/cfg" ]; then
                mkdir -p "$DEPLOY_DIR/experiments/$exp_name/scale/cfg"
                find "$exp_dir/scale/cfg" -name '*.json' -exec cp {} "$DEPLOY_DIR/experiments/$exp_name/scale/cfg/" \; 2>/dev/null || true
            fi
        fi
    done
fi

# 5. Копируем модели если нужны для инференса
if [ -d "$SOURCE_DIR/models" ]; then
    echo "   ✅ Копирую models/ (только pkl файлы)..."
    mkdir -p "$DEPLOY_DIR/models"
    find "$SOURCE_DIR/models" -name '*.pkl' -exec cp --parents {} "$DEPLOY_DIR/" \; 2>/dev/null || true
fi

# НЕ копируем:
echo ""
echo "   ⏭️  Пропускаю (не нужны для деплоя):"
echo "      - wsi/ (большие .tiff файлы)"
echo "      - docs/ (документация)"
echo "      - tests/ (тесты)"
echo "      - notebook/ (ноутбуки)"
echo "      - archive/ (архивы)"
echo "      - model_development/ (разработка)"

# Копируем конфигурационные файлы если их еще нет
if [ ! -f "$DEPLOY_DIR/Dockerfile" ]; then
    if [ -f "$SOURCE_DIR/gcp_deployment/Dockerfile" ]; then
        echo "   ✅ Копирую Dockerfile ..."
        cp "$SOURCE_DIR/gcp_deployment/Dockerfile" "$DEPLOY_DIR/"
    elif [ -f "$SOURCE_DIR/Dockerfile.dashboard" ]; then
        echo "   ✅ Копирую Dockerfile.dashboard как Dockerfile ..."
        cp "$SOURCE_DIR/Dockerfile.dashboard" "$DEPLOY_DIR/Dockerfile"
    fi
fi

# Копируем cloudbuild.yaml если его нет
if [ ! -f "$DEPLOY_DIR/cloudbuild.yaml" ] && [ -f "$SOURCE_DIR/gcp_deployment/cloudbuild.yaml" ]; then
    echo "   ✅ Копирую cloudbuild.yaml ..."
    cp "$SOURCE_DIR/gcp_deployment/cloudbuild.yaml" "$DEPLOY_DIR/"
fi

# Копируем скрипты деплоя
if [ -f "$SOURCE_DIR/gcp_deployment/deploy_gcp.sh" ]; then
    echo "   ✅ Копирую deploy_gcp.sh ..."
    cp "$SOURCE_DIR/gcp_deployment/deploy_gcp.sh" "$DEPLOY_DIR/"
    chmod +x "$DEPLOY_DIR/deploy_gcp.sh"
fi

if [ -f "$SOURCE_DIR/gcp_deployment/setup_gcp_apis.sh" ]; then
    echo "   ✅ Копирую setup_gcp_apis.sh ..."
    cp "$SOURCE_DIR/gcp_deployment/setup_gcp_apis.sh" "$DEPLOY_DIR/"
    chmod +x "$DEPLOY_DIR/setup_gcp_apis.sh"
fi

# 6. Копируем .streamlit конфигурацию если есть
if [ -d "$SOURCE_DIR/.streamlit" ]; then
    echo "   ✅ Копирую .streamlit/ ..."
    mkdir -p "$DEPLOY_DIR/.streamlit"
    cp -r "$SOURCE_DIR/.streamlit"/* "$DEPLOY_DIR/.streamlit/" 2>/dev/null || true
fi

# 7. Создаем необходимые директории для работы dashboard
echo "   ✅ Создаю структуру директорий..."
mkdir -p "$DEPLOY_DIR/experiments"
mkdir -p "$DEPLOY_DIR/results/inference"
mkdir -p "$DEPLOY_DIR/results/predictions"
mkdir -p "$DEPLOY_DIR/results/visualization"

# Создаем .dockerignore (минимальный набор исключений)
echo "   ✅ Создаю .dockerignore ..."
cat > "$DEPLOY_DIR/.dockerignore" << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# Documentation (не нужна для деплоя)
docs/
*.md
README*

# Tests (не нужны для деплоя)
tests/
test_*.py
*_test.py

# Archives
*.tar.gz
*.zip

# Logs
*.log

# Large image files (WSI - Whole Slide Images) - НЕ нужны
*.tiff
*.tif
*.TIFF
*.TIF
*.svs
*.ndpi
wsi/

# Development directories (не нужны для деплоя)
notebook/
archive/
model_development/
gcp_deployment/

# Large data files (если не нужны)
data/
*.h5
*.hdf5

# Results directories - должны монтироваться извне или загружаться из GCS/GDrive
# НЕ копируем в образ, чтобы данные были актуальными
results/predictions/
results/inference/
results/visualization/

# Keep JSON and CSV files в других местах - они нужны для dashboard!
# Keep .pkl files - модели нужны для инференса
EOF

# Создаем .gcloudignore (минимальный набор исключений)
echo "   ✅ Создаю .gcloudignore ..."
cat > "$DEPLOY_DIR/.gcloudignore" << 'EOF'
# This file specifies files that are *not* uploaded to Google Cloud
# Минимальный набор исключений для dashboard

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

# Documentation (не нужна для деплоя)
*.md
README*
docs/

# Tests (не нужны для деплоя)
tests/
test_*.py
*_test.py

# Development directories (не нужны для деплоя)
notebook/
archive/
model_development/
gcp_deployment/

# Archives
*.tar.gz
*.zip

# Large image files (WSI - Whole Slide Images) - НЕ нужны
*.tiff
*.tif
*.TIFF
*.TIF
*.svs
*.ndpi
wsi/

# Keep JSON, CSV, and PKL files - они нужны для dashboard!
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Подготовка завершена!"
echo ""
echo "📁 Проект готов к развертыванию в: $DEPLOY_DIR"
echo ""

# Проверка наличия ключевых файлов
echo "🔍 Проверка необходимых файлов:"
echo ""

MISSING_FILES=0

if [ -f "$DEPLOY_DIR/Dockerfile" ]; then
    echo "   ✅ Dockerfile"
else
    echo "   ❌ Dockerfile - ОТСУТСТВУЕТ!"
    MISSING_FILES=$((MISSING_FILES + 1))
fi

if [ -f "$DEPLOY_DIR/requirements.txt" ]; then
    echo "   ✅ requirements.txt"
else
    echo "   ⚠️  requirements.txt - не найден"
fi

if [ -d "$DEPLOY_DIR/scale" ]; then
    if [ -f "$DEPLOY_DIR/scale/dashboard.py" ]; then
        echo "   ✅ scale/dashboard.py"
    else
        echo "   ⚠️  scale/dashboard.py - не найден"
    fi
else
    echo "   ⚠️  scale/ - директория не найдена"
fi

if [ -f "$DEPLOY_DIR/cloudbuild.yaml" ]; then
    echo "   ✅ cloudbuild.yaml"
else
    echo "   ⚠️  cloudbuild.yaml - не найден (опционально)"
fi

# Проверка JSON файлов
JSON_COUNT=$(find "$DEPLOY_DIR" -name '*.json' -type f 2>/dev/null | wc -l)
if [ "$JSON_COUNT" -gt 0 ]; then
    echo "   ✅ Найдено JSON файлов: $JSON_COUNT"
else
    echo "   ⚠️  JSON файлы не найдены (могут быть добавлены позже)"
fi

# Проверка CSV файлов
CSV_COUNT=$(find "$DEPLOY_DIR/experiments" -name '*.csv' -type f 2>/dev/null | wc -l)
if [ "$CSV_COUNT" -gt 0 ]; then
    echo "   ✅ Найдено CSV файлов в experiments: $CSV_COUNT"
fi

echo ""
echo "📊 Статистика скопированных файлов:"
echo "   - Python файлов: $(find "$DEPLOY_DIR/scale" -name '*.py' -type f 2>/dev/null | wc -l)"
echo "   - JSON файлов: $JSON_COUNT"
echo "   - CSV файлов: $CSV_COUNT"
echo "   - Экспериментов: $(find "$DEPLOY_DIR/experiments" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
echo ""

if [ $MISSING_FILES -eq 0 ]; then
    echo "✅ Все необходимые файлы на месте!"
    echo ""
    echo "💡 Следующий шаг:"
    echo "   cd $DEPLOY_DIR"
    echo "   ./deploy_gcp.sh"
else
    echo "⚠️  Некоторые файлы отсутствуют. Проверьте выше."
fi

echo ""

