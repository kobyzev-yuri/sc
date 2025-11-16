# Решение проблем с установкой TensorRT

## Проблема: Не удается установить TensorRT

Если вы столкнулись с проблемами установки TensorRT через pip или conda, это нормально. TensorRT - это **опциональная оптимизация**, а не обязательное требование.

## ✅ Решение: Использовать PyTorch модели

**Хорошая новость:** Система полностью работает без TensorRT! PyTorch модели работают отлично и обеспечивают хорошую производительность.

### Что делать:

1. **Продолжайте использовать PyTorch модели** - они работают без TensorRT
2. Система автоматически использует PyTorch fallback если TensorRT недоступен
3. TensorRT дает ускорение 2-5x, но для большинства случаев PyTorch достаточно быстрый

### Использование без TensorRT:

```python
from scale.model_config import create_model_configs

# Использовать PyTorch модели (по умолчанию)
model_configs = create_model_configs(use_tensorrt=False)

# Или просто не указывать use_tensorrt - будет использован PyTorch
model_configs = create_model_configs()
```

## Альтернативные способы установки TensorRT (если все же нужен)

### Способ 1: Официальный пакет NVIDIA (наиболее надежно)

1. Зарегистрируйтесь на [NVIDIA Developer](https://developer.nvidia.com/)
2. Скачайте TensorRT с [официального сайта](https://developer.nvidia.com/tensorrt)
3. Распакуйте архив
4. Установите Python пакет:
   ```bash
   cd TensorRT-<version>/python
   pip install tensorrt-*-cp310-linux_x86_64.whl
   ```

### Способ 2: Docker контейнер с TensorRT

Используйте официальный Docker образ NVIDIA с предустановленным TensorRT:
```bash
docker pull nvcr.io/nvidia/tensorrt:23.12-py3
```

### Способ 3: Использовать ONNX Runtime (альтернатива)

ONNX Runtime может дать ускорение без TensorRT:
```bash
pip install onnxruntime-gpu
```

## Когда TensorRT действительно нужен?

TensorRT полезен если:
- Обрабатываете очень большие объемы данных
- Нужна максимальная производительность
- Работаете в production с высокими требованиями к скорости

Для большинства исследовательских задач PyTorch достаточно.

## Проверка работы без TensorRT

Система автоматически определит отсутствие TensorRT и использует PyTorch:

```python
from scale.model_config import create_model_configs

# Автоматически использует PyTorch если TensorRT недоступен
model_configs = create_model_configs(use_tensorrt=True, tensorrt_fallback=True)
```

## Вывод

**Не беспокойтесь если TensorRT не устанавливается!** Система полностью функциональна с PyTorch моделями. TensorRT - это дополнительная оптимизация, а не требование.



