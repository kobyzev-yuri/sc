"""
Конфигурация моделей для детекции патологий.

Все модели находятся в директории models/ и используются для анализа WSI изображений.
Конфигурация основана на notebook/predict.ipynb
"""

from pathlib import Path

from . import aimodels, predict


# Базовый путь к моделям
MODELS_DIR = Path(__file__).parent.parent / "models"


def create_model_configs(
    models_dir: Path = MODELS_DIR,
    use_tensorrt: bool = False,
    tensorrt_fallback: bool = True,
) -> list[predict.ModelConfig]:
    """
    Создает конфигурации всех моделей для детекции патологий.
    
    Args:
        models_dir: Путь к директории с моделями
        use_tensorrt: Если True, использует TensorRT модели (если доступны)
        tensorrt_fallback: Если True и TensorRT модель не найдена, использует обычную PyTorch модель
        
    Returns:
        Список конфигураций моделей
    """
    models_dir = Path(models_dir)
    
    # Выбираем класс модели в зависимости от параметра use_tensorrt
    ModelClass = aimodels.YOLOTensorRTModel if use_tensorrt else aimodels.YOLOModel
    
    # Инициализация моделей с порогами уверенности и маппингом классов
    if use_tensorrt:
        model_mild = ModelClass(
            str(models_dir / "nn_seg_data_outputs_mild_train.pt"),
            0.6,
            {0: "Mild"},
            fallback_to_pt=tensorrt_fallback,
        )
    else:
        model_mild = ModelClass(
            str(models_dir / "nn_seg_data_outputs_mild_train.pt"),
            0.6,
            {0: "Mild"}
        )
    
    if use_tensorrt:
        model_mod = ModelClass(
            str(models_dir / "moderate_seg_train6_acc2.pt"),
            0.7,
            {0: "Moderate"},
            fallback_to_pt=tensorrt_fallback,
        )
        
        model_gran = ModelClass(
            str(models_dir / "nn_det2_data_outputs_gran_train5.pt"),
            0.2,
            {0: "Granulomas"},
            fallback_to_pt=tensorrt_fallback,
        )
        
        model_dysp = ModelClass(
            str(models_dir / "nn_det2_data_outputs_dysplasia_ibd_seg_train3.pt"),
            0.2,
            {0: "Dysplasia"},
            fallback_to_pt=tensorrt_fallback,
        )
        
        model_eoe = ModelClass(
            str(models_dir / "nn_det2_data_outputs_eoe_train6.pt"),
            0.4,
            {0: "EoE"},
            fallback_to_pt=tensorrt_fallback,
        )
        
        model_meta = ModelClass(
            str(models_dir / "nn_det2_data_outputs_meta_train4.pt"),
            0.2,
            {0: "Meta"},
            fallback_to_pt=tensorrt_fallback,
        )
        
        model_neutrophils = ModelClass(
            str(models_dir / "nn_det2_data_outputs_neutrophils_train7.pt"),
            0.3,
            {0: "Neutrophils"},
            fallback_to_pt=tensorrt_fallback,
        )
        
        model_plasma = ModelClass(
            str(models_dir / "nn_det2_data_outputs_plasma-transformed_train3.pt"),
            0.3,
            {0: "Plasma Cells"},
            fallback_to_pt=tensorrt_fallback,
        )
        
        model_lp = ModelClass(
            str(models_dir / "nn_det2_data_outputs_lp_train3.pt"),
            0.5,
            {0: "Crypts", 1: "Muscularis mucosae", 2: "Surface epithelium", 3: "White space"},
            fallback_to_pt=tensorrt_fallback,
        )
        
        model_enterocytes = ModelClass(
            str(models_dir / "nn_det2_data_outputs_enterocytes_train2.pt"),
            0.4,
            {0: "Enterocytes"},
            fallback_to_pt=tensorrt_fallback,
        )
        
        model_paneth = ModelClass(
            str(models_dir / "nn_det2_data_outputs_paneth_train5.pt"),
            0.4,
            {0: "Paneth"},
            fallback_to_pt=tensorrt_fallback,
        )
    else:
        model_mod = ModelClass(
            str(models_dir / "moderate_seg_train6_acc2.pt"),
            0.7,
            {0: "Moderate"}
        )
        
        model_gran = ModelClass(
            str(models_dir / "nn_det2_data_outputs_gran_train5.pt"),
            0.2,
            {0: "Granulomas"}
        )
        
        model_dysp = ModelClass(
            str(models_dir / "nn_det2_data_outputs_dysplasia_ibd_seg_train3.pt"),
            0.2,
            {0: "Dysplasia"}
        )
        
        model_eoe = ModelClass(
            str(models_dir / "nn_det2_data_outputs_eoe_train6.pt"),
            0.4,
            {0: "EoE"}
        )
        
        model_meta = ModelClass(
            str(models_dir / "nn_det2_data_outputs_meta_train4.pt"),
            0.2,
            {0: "Meta"}
        )
        
        model_neutrophils = ModelClass(
            str(models_dir / "nn_det2_data_outputs_neutrophils_train7.pt"),
            0.3,
            {0: "Neutrophils"}
        )
        
        model_plasma = ModelClass(
            str(models_dir / "nn_det2_data_outputs_plasma-transformed_train3.pt"),
            0.3,
            {0: "Plasma Cells"}
        )
        
        model_lp = ModelClass(
            str(models_dir / "nn_det2_data_outputs_lp_train3.pt"),
            0.5,
            {0: "Crypts", 1: "Muscularis mucosae", 2: "Surface epithelium", 3: "White space"}
        )
        
        model_enterocytes = ModelClass(
            str(models_dir / "nn_det2_data_outputs_enterocytes_train2.pt"),
            0.4,
            {0: "Enterocytes"}
        )
        
        model_paneth = ModelClass(
            str(models_dir / "nn_det2_data_outputs_paneth_train5.pt"),
            0.4,
            {0: "Paneth"}
        )
    
    # Конфигурации моделей с размерами окон
    model_configs = [
        predict.ModelConfig(model_mild, 514 * 2),
        
        predict.ModelConfig(model_mod, 514 * 6),
        predict.ModelConfig(model_gran, 514 * 6),
        predict.ModelConfig(model_dysp, 514 * 6),
        predict.ModelConfig(model_lp, 514 * 6),
        
        predict.ModelConfig(model_eoe, 514),
        predict.ModelConfig(model_meta, 514),
        predict.ModelConfig(model_neutrophils, 514),
        predict.ModelConfig(model_plasma, 514),
        
        predict.ModelConfig(model_enterocytes, 514),
        predict.ModelConfig(model_paneth, 514),
    ]
    
    return model_configs


def create_tensorrt_model_configs(
    models_dir: Path = MODELS_DIR,
    fallback_to_pt: bool = True,
) -> list[predict.ModelConfig]:
    """
    Создает конфигурации всех моделей с использованием TensorRT оптимизации.
    
    Это удобная функция-обертка для create_model_configs с use_tensorrt=True.
    
    Args:
        models_dir: Путь к директории с моделями
        fallback_to_pt: Если True и TensorRT модель не найдена, использует обычную PyTorch модель
        
    Returns:
        Список конфигураций моделей с TensorRT оптимизацией
    """
    return create_model_configs(
        models_dir=models_dir,
        use_tensorrt=True,
        tensorrt_fallback=fallback_to_pt,
    )


def get_postprocess_settings() -> dict[str, predict.Postprocessing]:
    """
    Создает настройки постобработки для каждого типа патологии.
    
    Returns:
        Словарь с настройками постобработки для каждого класса
    """
    return {
        "Mild": predict.Postprocessing(nms_thres=0.7, cov_thres=0.4),
        "Moderate": predict.Postprocessing(nms_thres=0.7, cov_thres=0.4),
        "Granulomas": predict.Postprocessing(nms_thres=0.7, cov_thres=0.4),
        "Dysplasia": predict.Postprocessing(nms_thres=0.7, cov_thres=0.4),
        "EoE": predict.Postprocessing(nms_thres=0.4, cov_thres=None),
        "Meta": predict.Postprocessing(nms_thres=0.4, cov_thres=None),
        "Neutrophils": predict.Postprocessing(nms_thres=0.4, cov_thres=None),
        "Plasma Cells": predict.Postprocessing(nms_thres=0.4, cov_thres=None),
        "Crypts": predict.Postprocessing(nms_thres=0.7, cov_thres=0.5),
        "Muscularis mucosae": predict.Postprocessing(nms_thres=None, cov_thres=0.2),
        "Surface epithelium": predict.Postprocessing(nms_thres=None, cov_thres=0.5),
        "White space": predict.Postprocessing(nms_thres=None, cov_thres=0.1),
        "Enterocytes": predict.Postprocessing(nms_thres=0.4, cov_thres=None),
        "Paneth": predict.Postprocessing(nms_thres=0.4, cov_thres=None),
    }


# Список всех детектируемых патологий
DETECTED_PATHOLOGIES = [
    "Mild",
    "Moderate",
    "Crypts",
    "White space",
    "Surface epithelium",
    "Dysplasia",
    "Meta",
    "Plasma Cells",
    "Neutrophils",
    "EoE",
    "Granulomas",
    "Paneth cells",  # Модель есть, но не используется в ноутбуке
    "Enterocytes",
    "LP",  # Lamina propria - возможно, это часть model_lp
]

