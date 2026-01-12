#!/bin/bash
# Скрипт для создания полного пакета развертывания со всеми файлами
# Создает готовую директорию для переноса на удаленный сервер

set -e

echo "📦 Создание полного пакета для развертывания"
echo "=============================================="
echo ""

SOURCE_DIR="${1:-/mnt/ai/cnn/sc}"
PACKAGE_DIR="${2:-/mnt/ai/cnn/sc/deployment_full}"

echo "📂 Исходная директория: $SOURCE_DIR"
echo "📂 Директория пакета: $PACKAGE_DIR"
echo ""

# Проверка исходной директории
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Ошибка: исходная директория не найдена: $SOURCE_DIR"
    exit 1
fi

# Создаем директорию пакета
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

echo "📋 Копирование файлов..."
echo ""

# 1. Копируем код приложения scale/
if [ -d "$SOURCE_DIR/scale" ]; then
    echo "   ✅ Копирую scale/ ..."
    rsync -av --exclude='__pycache__' --exclude='*.pyc' \
        "$SOURCE_DIR/scale/" "$PACKAGE_DIR/scale/" 2>/dev/null || {
        mkdir -p "$PACKAGE_DIR/scale"
        cp -r "$SOURCE_DIR/scale"/* "$PACKAGE_DIR/scale/" 2>/dev/null || true
    }
else
    echo "   ❌ Директория scale/ не найдена!"
    exit 1
fi

# 2. Копируем requirements.txt
if [ -f "$SOURCE_DIR/requirements.txt" ]; then
    echo "   ✅ Копирую requirements.txt ..."
    cp "$SOURCE_DIR/requirements.txt" "$PACKAGE_DIR/"
else
    echo "   ❌ requirements.txt не найден!"
    exit 1
fi

# 3. Копируем JSON файлы из results/inference/
if [ -d "$SOURCE_DIR/results/inference" ]; then
    echo "   ✅ Копирую results/inference/*.json ..."
    mkdir -p "$PACKAGE_DIR/results/inference"
    find "$SOURCE_DIR/results/inference" -maxdepth 1 -name '*.json' -exec cp {} "$PACKAGE_DIR/results/inference/" \; 2>/dev/null || true
fi
mkdir -p "$PACKAGE_DIR/results/predictions"
mkdir -p "$PACKAGE_DIR/results/visualization"

# 4. Копируем эксперименты (только JSON и CSV)
if [ -d "$SOURCE_DIR/experiments" ]; then
    echo "   ✅ Копирую experiments/ (JSON и CSV)..."
    mkdir -p "$PACKAGE_DIR/experiments"
    
    for exp_dir in "$SOURCE_DIR/experiments"/*/; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            echo "      📁 $exp_name"
            mkdir -p "$PACKAGE_DIR/experiments/$exp_name"
            
            # JSON конфигурации
            find "$exp_dir" -maxdepth 1 -name '*.json' -exec cp {} "$PACKAGE_DIR/experiments/$exp_name/" \; 2>/dev/null || true
            
            # CSV файлы
            find "$exp_dir" -maxdepth 1 -name '*.csv' -exec cp {} "$PACKAGE_DIR/experiments/$exp_name/" \; 2>/dev/null || true
            
            # PKL модели
            find "$exp_dir" -maxdepth 1 -name '*.pkl' -exec cp {} "$PACKAGE_DIR/experiments/$exp_name/" \; 2>/dev/null || true
            
            # scale/cfg/ если есть
            if [ -d "$exp_dir/scale/cfg" ]; then
                mkdir -p "$PACKAGE_DIR/experiments/$exp_name/scale/cfg"
                find "$exp_dir/scale/cfg" -name '*.json' -exec cp {} "$PACKAGE_DIR/experiments/$exp_name/scale/cfg/" \; 2>/dev/null || true
            fi
        fi
    done
fi

# 5. Копируем модели если есть
if [ -d "$SOURCE_DIR/models" ]; then
    echo "   ✅ Копирую models/ (pkl файлы)..."
    mkdir -p "$PACKAGE_DIR/models"
    find "$SOURCE_DIR/models" -name '*.pkl' -exec cp --parents {} "$PACKAGE_DIR/" \; 2>/dev/null || true
fi

# 5.5. Копируем model_development (нужен для dashboard)
if [ -d "$SOURCE_DIR/model_development" ]; then
    echo "   ✅ Копирую model_development/ ..."
    rsync -av --exclude='__pycache__' --exclude='*.pyc' \
        "$SOURCE_DIR/model_development/" "$PACKAGE_DIR/model_development/" 2>/dev/null || {
        mkdir -p "$PACKAGE_DIR/model_development"
        cp -r "$SOURCE_DIR/model_development"/* "$PACKAGE_DIR/model_development/" 2>/dev/null || true
    }
fi

# 6. Копируем конфигурационные файлы деплоя
echo "   ✅ Копирую файлы деплоя..."
cp "$SOURCE_DIR/gcp_deployment/Dockerfile" "$PACKAGE_DIR/" 2>/dev/null || \
    cp "$SOURCE_DIR/gcp_deployment/Dockerfile" "$PACKAGE_DIR/" 2>/dev/null || true

cp "$SOURCE_DIR/gcp_deployment/cloudbuild.yaml" "$PACKAGE_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/gcp_deployment/deploy_gcp.sh" "$PACKAGE_DIR/" 2>/dev/null || true
chmod +x "$PACKAGE_DIR/deploy_gcp.sh" 2>/dev/null || true

# 7. Копируем .streamlit конфигурацию если есть
if [ -d "$SOURCE_DIR/.streamlit" ]; then
    echo "   ✅ Копирую .streamlit/ ..."
    mkdir -p "$PACKAGE_DIR/.streamlit"
    cp -r "$SOURCE_DIR/.streamlit"/* "$PACKAGE_DIR/.streamlit/" 2>/dev/null || true
fi

# 8. Создаем .dockerignore
echo "   ✅ Создаю .dockerignore ..."
cat > "$PACKAGE_DIR/.dockerignore" << 'EOF'
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

# Documentation
docs/
*.md
README*

# Tests
tests/
test_*.py
*_test.py

# Archives
*.tar.gz
*.zip

# Logs
*.log

# Large image files (WSI)
*.tiff
*.tif
*.TIFF
*.TIF
*.svs
*.ndpi
wsi/

# Development directories
notebook/
archive/
gcp_deployment/
# model_development/ - НЕ исключаем, нужен для dashboard!

# Large data files
data/
*.h5
*.hdf5

# Results directories - должны монтироваться извне или загружаться из GCS/GDrive
# НЕ копируем в образ, чтобы данные были актуальными
results/predictions/
results/inference/
results/visualization/
EOF

# 9. Создаем .gcloudignore
echo "   ✅ Создаю .gcloudignore ..."
cat > "$PACKAGE_DIR/.gcloudignore" << 'EOF'
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
gcp_deployment/
# model_development/ - НЕ исключаем, нужен для dashboard!
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

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Пакет создан!"
echo ""

# Проверка
echo "🔍 Проверка созданного пакета:"
echo ""

MISSING=0

if [ -f "$PACKAGE_DIR/Dockerfile" ]; then
    echo "   ✅ Dockerfile"
else
    echo "   ❌ Dockerfile - ОТСУТСТВУЕТ!"
    MISSING=$((MISSING + 1))
fi

if [ -f "$PACKAGE_DIR/requirements.txt" ]; then
    echo "   ✅ requirements.txt"
else
    echo "   ❌ requirements.txt - ОТСУТСТВУЕТ!"
    MISSING=$((MISSING + 1))
fi

if [ -d "$PACKAGE_DIR/scale" ] && [ -f "$PACKAGE_DIR/scale/dashboard.py" ]; then
    echo "   ✅ scale/dashboard.py"
else
    echo "   ❌ scale/dashboard.py - ОТСУТСТВУЕТ!"
    MISSING=$((MISSING + 1))
fi

if [ -f "$PACKAGE_DIR/deploy_gcp.sh" ]; then
    echo "   ✅ deploy_gcp.sh"
else
    echo "   ⚠️  deploy_gcp.sh - не найден"
fi

echo ""
echo "📊 Статистика:"
echo "   - Python файлов: $(find "$PACKAGE_DIR/scale" -name '*.py' -type f 2>/dev/null | wc -l)"
echo "   - JSON файлов: $(find "$PACKAGE_DIR" -name '*.json' -type f 2>/dev/null | wc -l)"
echo "   - CSV файлов: $(find "$PACKAGE_DIR/experiments" -name '*.csv' -type f 2>/dev/null | wc -l)"
echo "   - Размер: $(du -sh "$PACKAGE_DIR" | cut -f1)"
echo ""

if [ $MISSING -eq 0 ]; then
    echo "✅ Все необходимые файлы на месте!"
    echo ""
    echo "📦 Создание архива..."
    cd "$(dirname "$PACKAGE_DIR")"
    tar -czf "$(basename "$PACKAGE_DIR").tar.gz" "$(basename "$PACKAGE_DIR")"
    echo "✅ Архив создан: $(dirname "$PACKAGE_DIR")/$(basename "$PACKAGE_DIR").tar.gz"
    echo ""
    echo "💡 Следующий шаг:"
    echo "   scp $(dirname "$PACKAGE_DIR")/$(basename "$PACKAGE_DIR").tar.gz ai8049520@instance-20251117-192323:~/"
    echo "   # На сервере:"
    echo "   tar -xzf $(basename "$PACKAGE_DIR").tar.gz"
    echo "   cd $(basename "$PACKAGE_DIR")"
    echo "   ./deploy_gcp.sh"
else
    echo "⚠️  Некоторые файлы отсутствуют. Проверьте выше."
fi

echo ""

