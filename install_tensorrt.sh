#!/bin/bash
# Скрипт для установки TensorRT различными способами

echo "=========================================="
echo "Установка TensorRT для Python 3.10"
echo "=========================================="
echo ""

# Проверяем Python версию
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python версия: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "⚠️  Внимание: TensorRT лучше работает с Python 3.10"
    echo "   Текущая версия: $PYTHON_VERSION"
    echo ""
fi

echo "Выберите метод установки:"
echo "1) NVIDIA PyIndex (рекомендуется для pip)"
echo "2) Conda (если используете conda)"
echo "3) Пропустить (TensorRT уже установлен или будет установлен вручную)"
echo ""
read -p "Ваш выбор (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Установка через NVIDIA PyIndex..."
        pip install nvidia-pyindex
        if [ $? -eq 0 ]; then
            echo "✅ nvidia-pyindex установлен"
            pip install nvidia-tensorrt
            if [ $? -eq 0 ]; then
                echo "✅ TensorRT установлен успешно!"
                python -c "import tensorrt as trt; print(f'TensorRT версия: {trt.__version__}')"
            else
                echo "❌ Ошибка установки nvidia-tensorrt"
                exit 1
            fi
        else
            echo "❌ Ошибка установки nvidia-pyindex"
            exit 1
        fi
        ;;
    2)
        echo ""
        echo "Установка через conda..."
        conda install -c nvidia tensorrt -y
        if [ $? -eq 0 ]; then
            echo "✅ TensorRT установлен успешно!"
            python -c "import tensorrt as trt; print(f'TensorRT версия: {trt.__version__}')"
        else
            echo "❌ Ошибка установки через conda"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo "Пропуск установки. Убедитесь что TensorRT установлен перед конвертацией моделей."
        ;;
    *)
        echo "❌ Неверный выбор"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Проверка установки TensorRT"
echo "=========================================="
python -c "
try:
    import tensorrt as trt
    print(f'✅ TensorRT установлен, версия: {trt.__version__}')
except ImportError:
    print('❌ TensorRT не установлен')
    print('   Установите TensorRT одним из способов выше')
    exit(1)
"

