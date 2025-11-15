from . import domain

import numpy as np
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


class YOLOModel:
    def __init__(
        self,
        model_path: str,
        min_conf: float,
        cls_index_to_name: dict[int, str],
    ):
        self.min_conf = min_conf
        self.model = YOLO(model_path)
        self.cls_index_to_name = cls_index_to_name

    def predict(self, images: list[np.ndarray]) -> dict[str, list[domain.Prediction]]:
        predictions = self.model.predict(images, verbose=False, conf=self.min_conf)
        return self._parse_yolo_predictions(predictions)

    def _parse_yolo_predictions(
        self, preds: list
    ) -> dict[str, list[domain.Prediction]]:
        parsed_preds = defaultdict(list)

        for pred in preds:
            single_pred = self._parse_single_yolo_prediction(pred)
            for cls_name, preds_list in single_pred.items():
                parsed_preds[cls_name].extend(preds_list)

        return parsed_preds

    def _parse_single_yolo_prediction(
        self, pred
    ) -> dict[str, list[domain.Prediction]]:
        """Парсит результаты YOLO для одного изображения."""
        parsed_preds = defaultdict(list)
        
        boxes = pred.boxes.xyxy.tolist()
        confs = pred.boxes.conf.tolist()
        cls_indexes = pred.boxes.cls.tolist()

        has_masks = hasattr(pred, "masks") and pred.masks is not None

        polygons = []
        if has_masks:
            polygons = pred.masks.xy

        for i, (box, cls_index, conf) in enumerate(zip(boxes, cls_indexes, confs)):
            if conf < self.min_conf:
                continue

            x1, y1, x2, y2 = box
            pathology_type = self.cls_index_to_name[cls_index]

            polygon = None
            if has_masks:
                coords = [domain.Coords(float(x), float(y)) for x, y in polygons[i]]
                try:
                    polygon = domain.Polygon(coords=coords)
                except domain.BadPolygonError as e:
                    print(e)
                    continue

            parsed_box = domain.Prediction(
                domain.Box(start=domain.Coords(x1, y1), end=domain.Coords(x2, y2)),
                conf=conf,
                polygon=polygon,
            )
            parsed_preds[pathology_type].append(parsed_box)

        return parsed_preds


class YOLOTensorRTModel:
    """
    YOLO модель с оптимизацией TensorRT для ускорения инференса на NVIDIA GPU.
    
    TensorRT модели (.engine файлы) должны быть сгенерированы заранее с помощью
    скрипта convert_to_tensorrt.py. Эти файлы специфичны для конкретной GPU и
    версии TensorRT, поэтому их нужно генерировать на целевой системе.
    """
    def __init__(
        self,
        model_path: str,
        min_conf: float,
        cls_index_to_name: dict[int, str],
        fallback_to_pt: bool = True,
    ):
        """
        Инициализирует TensorRT модель YOLO.
        
        Args:
            model_path: Путь к .engine файлу TensorRT или .pt файлу (если fallback_to_pt=True)
            min_conf: Минимальный порог уверенности для детекций
            cls_index_to_name: Маппинг индексов классов на имена
            fallback_to_pt: Если True, загрузит обычную PyTorch модель если .engine не найден
        """
        self.min_conf = min_conf
        self.cls_index_to_name = cls_index_to_name
        self.fallback_to_pt = fallback_to_pt
        
        model_path_obj = Path(model_path)
        
        # Проверяем наличие TensorRT engine файла
        if model_path_obj.suffix == '.engine' and model_path_obj.exists():
            # Загружаем TensorRT engine напрямую
            self.model = YOLO(str(model_path_obj))
            self.is_tensorrt = True
        elif model_path_obj.suffix == '.pt' and fallback_to_pt:
            # Пытаемся найти соответствующий .engine файл
            engine_path = model_path_obj.with_suffix('.engine')
            if engine_path.exists():
                self.model = YOLO(str(engine_path))
                self.is_tensorrt = True
            else:
                # Fallback на обычную PyTorch модель
                self.model = YOLO(str(model_path_obj))
                self.is_tensorrt = False
                if fallback_to_pt:
                    print(f"Warning: TensorRT engine not found for {model_path_obj.name}, "
                          f"using PyTorch model instead")
        else:
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"For TensorRT, provide .engine file or .pt file with corresponding .engine"
            )

    def predict(self, images: list[np.ndarray]) -> dict[str, list[domain.Prediction]]:
        """
        Выполняет предсказание на списке изображений.
        
        TensorRT модели обычно работают быстрее, особенно на больших batch sizes.
        """
        predictions = self.model.predict(images, verbose=False, conf=self.min_conf)
        return self._parse_yolo_predictions(predictions)

    def _parse_yolo_predictions(
        self, preds: list
    ) -> dict[str, list[domain.Prediction]]:
        """Парсит результаты YOLO (совместимо с обычным YOLOModel)."""
        parsed_preds = defaultdict(list)

        for pred in preds:
            single_pred = self._parse_single_yolo_prediction(pred)
            for cls_name, preds_list in single_pred.items():
                parsed_preds[cls_name].extend(preds_list)

        return parsed_preds

    def _parse_single_yolo_prediction(
        self, pred
    ) -> dict[str, list[domain.Prediction]]:
        """Парсит результаты YOLO для одного изображения (совместимо с YOLOModel)."""
        parsed_preds = defaultdict(list)
        
        boxes = pred.boxes.xyxy.tolist()
        confs = pred.boxes.conf.tolist()
        cls_indexes = pred.boxes.cls.tolist()

        has_masks = hasattr(pred, "masks") and pred.masks is not None

        polygons = []
        if has_masks:
            polygons = pred.masks.xy

        for i, (box, cls_index, conf) in enumerate(zip(boxes, cls_indexes, confs)):
            if conf < self.min_conf:
                continue

            x1, y1, x2, y2 = box
            pathology_type = self.cls_index_to_name[cls_index]

            polygon = None
            if has_masks:
                coords = [domain.Coords(float(x), float(y)) for x, y in polygons[i]]
                try:
                    polygon = domain.Polygon(coords=coords)
                except domain.BadPolygonError as e:
                    print(e)
                    continue

            parsed_box = domain.Prediction(
                domain.Box(start=domain.Coords(x1, y1), end=domain.Coords(x2, y2)),
                conf=conf,
                polygon=polygon,
            )
            parsed_preds[pathology_type].append(parsed_box)

        return parsed_preds
