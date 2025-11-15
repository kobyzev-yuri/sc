from . import domain

from dataclasses import dataclass
from collections import defaultdict
from typing import Protocol, Iterator, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from contextlib import contextmanager

import numpy as np
import torch
from torchvision.ops import nms


@contextmanager
def timer(name: str, enabled: bool = True):
    start = time.perf_counter()
    yield
    if enabled:
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {name}: {elapsed:.3f} s")


class WSIIterator(Protocol):
    def iter_first_section_bgr(
        self, window_size: int, overlap_ratio: float
    ) -> Iterator[tuple[np.ndarray, domain.Coords]]: ...
    
    def iter_section_bgr(
        self, index: int, window_size: int, overlap_ratio: float
    ) -> Iterator[tuple[np.ndarray, domain.Coords]]: ...
    
    def iter_subsection_bgr(
        self, section_index: int, subsection_index: int, window_size: int, overlap_ratio: float
    ) -> Iterator[tuple[np.ndarray, domain.Coords]]: ...


class Model(Protocol):
    def predict(
        self, images: list[np.ndarray]
    ) -> dict[str, list[domain.Prediction]]: ...


@dataclass
class Postprocessing:
    nms_thres: Optional[float]
    cov_thres: Optional[float]


@dataclass
class ModelConfig:
    model: Model
    window_size: int


class WSIPredictor:
    def __init__(
        self,
        wsi_iterator: WSIIterator,
        model_configs: list[ModelConfig],
        postprocess_settings: dict[str, Postprocessing],
        overlap_ratio: float = 0.5,
        parallel_subsections: bool = True,
        max_workers: Optional[int] = None,
        enable_timing: bool = False,
    ):
        self.iterator = wsi_iterator
        self.model_configs = model_configs
        self.overlap_ratio = overlap_ratio
        self.postprocess_settings = postprocess_settings
        self.parallel_subsections = parallel_subsections
        # По умолчанию используем количество подсекций или оптимальное для A100
        self.max_workers = max_workers
        # YOLO потокобезопасен для работы с GPU, lock не нужен
        self.enable_timing = enable_timing

    def predict_first_section(self) -> dict[str, list[domain.Prediction]]:
        with timer("_predict_section(index=0)", self.enable_timing):
            preds_all = self._predict_section(sec_index=0)
        return self._postprocess(preds_all)

    def predict_first_section_via_subsections(self) -> dict[str, list[domain.Prediction]]:
        """Предсказывает первую секцию (индекс 0) через обработку подсекций.
        
        Удобный метод для быстрой обработки первой секции через подсекции.
        """
        with timer("_predict_section_via_subsections(index=0)", self.enable_timing):
            result = self.predict_section_via_subsections(0)
        return result

    def predict_section(self, section_index: int) -> dict[str, list[domain.Prediction]]:
        with timer(f"_predict_section(index={section_index})", self.enable_timing):
            preds_all = self._predict_section(sec_index=section_index)
        return self._postprocess(preds_all)

    def predict_subsection(
        self, section_index: int, subsection_index: int
    ) -> dict[str, list[domain.Prediction]]:
        preds_all = self._predict_subsection(section_index, subsection_index)
        return self._postprocess(preds_all)

    def predict_section_via_subsections(
        self, section_index: int
    ) -> dict[str, list[domain.Prediction]]:
        """Предсказывает секцию через обработку ТОЛЬКО подсекций (кусочков).
        
        ПРОЦЕСС РАБОТЫ:
        
        1. РАЗБИЕНИЕ СЕКЦИИ НА ПОДСЕКЦИИ:
           - Сначала применяем разбиение к указанной секции (обычно секция 0)
           - Разбиение может быть автоматическим (num_subsections=None) или 
             с указанием количества (num_subsections=N)
           - Метод extract_subsection_bounds находит отдельные кусочки ткани внутри секции
           - Полученные подсекции МОГУТ ПЕРЕСЕКАТЬСЯ (это нормально!)
           
        2. ОБРАБОТКА ПОДСЕКЦИЙ:
           - Обрабатываем КАЖДУЮ подсекцию отдельно (не всю секцию целиком!)
           - iter_subsection_bgr обрабатывает только границы конкретной подсекции
           - Это ускоряет обработку, так как каждая подсекция меньше всей секции
           
        3. ПОСТПРОЦЕССИНГ С УЧЕТОМ ПЕРЕСЕЧЕНИЙ:
           - Подсекции могут пересекаться, поэтому один участок ткани может быть
             обработан в нескольких подсекциях, что приводит к дубликатам
           - Специальный постпроцессинг:
             * Использует разделители для определения принадлежности предиктов к подсекциям
             * Фильтрует дубликаты в областях пересечения подсекций
             * Применяет стандартный NMS и merge для окончательной очистки
        
        Args:
            section_index: Индекс секции для обработки (обычно 0 - первая секция)
            
        Returns:
            Словарь с предсказаниями по классам (аналогично predict_section)
        """
        # Проверяем, что iterator имеет метод для получения подсекций
        if not hasattr(self.iterator, 'extract_subsection_bounds'):
            raise ValueError(
                "Iterator должен иметь метод extract_subsection_bounds для использования "
                "predict_section_via_subsections"
            )
        
        # ЭТАП 1: Разбиваем секцию на подсекции (кусочки)
        # extract_subsection_bounds применяет разбиение к секции:
        # - Если num_subsections=None: автоматически находит все крупные кусочки ткани
        # - Если num_subsections=N: берет N самых больших кусочков
        # ВАЖНО: Полученные подсекции МОГУТ ПЕРЕСЕКАТЬСЯ (bounding boxes могут пересекаться)
        subsection_bounds = self.iterator.extract_subsection_bounds(section_index)
        
        if not subsection_bounds:
            # Если подсекций нет, обрабатываем как обычную секцию
            return self.predict_section(section_index)
        
        # ЭТАП 2: Обрабатываем КАЖДУЮ подсекцию отдельно (не всю секцию!)
        # iter_subsection_bgr обрабатывает только границы конкретной подсекции
        
        timer_name = f"_predict_subsections(section={section_index}, count={len(subsection_bounds)})"
        with timer(timer_name, self.enable_timing):
            if self.parallel_subsections and len(subsection_bounds) > 1:
                # Параллельная обработка подсекций для лучшей загрузки GPU
                all_predictions = self._predict_subsections_parallel(
                    section_index, subsection_bounds
                )
            else:
                # Последовательная обработка (по умолчанию или если подсекция одна)
                all_predictions = defaultdict(list[domain.Prediction])
                
                for subsection_index in range(len(subsection_bounds)):
                    # Получаем предсказания ТОЛЬКО для этой подсекции (кусочка)
                    with timer(f"_predict_subsection(section={section_index}, subsection={subsection_index})", self.enable_timing):
                        subsection_preds_raw = self._predict_subsection(section_index, subsection_index)
                    
                    # Собираем все предикты из всех подсекций
                    # ВАЖНО: В областях пересечения подсекций могут быть дубликаты!
                    for cls_name, preds in subsection_preds_raw.items():
                        all_predictions[cls_name].extend(preds)
        
        # ЭТАП 3: Применяем специальный постпроцессинг для пересекающихся подсекций
        # который учитывает, что подсекции сами пересекаются и могут давать дубликаты
        with timer(f"_postprocess_subsections(section={section_index})", self.enable_timing):
            result = self._postprocess_subsections(all_predictions, section_index)
        return result

    def _postprocess(
        self, preds_all: dict[str, list[domain.Prediction]]
    ) -> dict[str, list[domain.Prediction]]:
        result = {}
        for cls_name, preds in preds_all.items():
            with timer(f"_postprocess_predictions({cls_name})", self.enable_timing):
                result[cls_name] = self._postprocess_predictions(preds, cls_name)

        return result

    def _postprocess_subsections(
        self,
        preds_all: dict[str, list[domain.Prediction]],
        section_index: int
    ) -> dict[str, list[domain.Prediction]]:
        """Специальный постпроцессинг для пересекающихся подсекций.
        
        ВАЖНО: Использует стандартный постпроцессинг (_postprocess) для обеспечения
        консистентности с обычным методом predict_section.
        
        Подсекции могут пересекаться, что приводит к дубликатам предиктов.
        Стандартный NMS и merge по coverage обрабатывают эти дубликаты так же,
        как и дубликаты из перекрывающихся патчей в обычном методе.
        
        Args:
            preds_all: Словарь с предсказаниями по классам из всех подсекций
            section_index: Индекс секции (не используется, но оставлен для совместимости)
            
        Returns:
            Словарь с отфильтрованными предсказаниями по классам
        """
        # Используем стандартный постпроцессинг для обеспечения консистентности
        # Стандартный NMS и merge по coverage обработают дубликаты из пересекающихся подсекций
        # так же, как и дубликаты из перекрывающихся патчей в обычном методе
        return self._postprocess(preds_all)

    def _filter_subsection_overlaps(
        self,
        preds: list[domain.Prediction],
        section_index: int,
        cls_name: str
    ) -> list[domain.Prediction]:
        """Фильтрует дубликаты предиктов в областях пересечения подсекций.
        
        ВАЖНО: Подсекции пересекаются, поэтому один и тот же участок ткани
        может быть обработан в нескольких подсекциях, что приводит к дубликатам.
        
        Использует тот же подход, что и стандартный постпроцессинг YOLO:
        1. NMS (Non-Maximum Suppression) с порогом из настроек класса
        2. Merge по coverage с порогом из настроек класса
        
        Это обеспечивает консистентность с обработкой обычных секций.
        
        Args:
            preds: Список предиктов из всех подсекций (могут быть дубликаты)
            section_index: Индекс секции
            cls_name: Имя класса (для получения настроек постпроцессинга)
            
        Returns:
            Отфильтрованный список предиктов без дубликатов в областях пересечения
        """
        if not preds:
            return []
        
        # Получаем границы подсекций для проверки пересечений
        subsection_bounds = self.iterator.extract_subsection_bounds(section_index)
        
        if len(subsection_bounds) <= 1:
            # Если подсекция одна или нет подсекций, возвращаем как есть
            return preds
        
        # Используем тот же подход, что и стандартный постпроцессинг
        # Получаем настройки для этого класса
        settings = self.postprocess_settings.get(cls_name)
        if settings is None:
            # Если нет настроек, используем стандартные пороги
            # Применяем NMS с порогом 0.5 (стандартный для YOLO)
            preds = self.__bbox_nms(preds, 0.5)
            return preds
        
        # Применяем NMS с порогом из настроек класса
        if settings.nms_thres:
            preds = self.__bbox_nms(preds, settings.nms_thres)
        
        # Применяем merge по coverage с порогом из настроек класса
        if settings.cov_thres:
            preds = self.__merge_by_coverage(preds, settings.cov_thres)
        
        return preds

    def _predict_section(self, sec_index: int, batch_size: int = 32) -> dict[str, list[domain.Prediction]]:
        """Предсказывает секцию с батчингом для лучшей загрузки GPU.
        
        Args:
            sec_index: Индекс секции
            batch_size: Размер батча для обработки патчей (по умолчанию 32 для A100)
        """
        size_to_models = self._group_models_by_window_size()
        predictions = defaultdict(list[domain.Prediction])

        for window_size, models in size_to_models.items():
            # Собираем патчи в батчи для лучшей загрузки GPU
            batch_regions = []
            batch_starts = []
            
            for region, start in self.iterator.iter_section_bgr(
                index=sec_index,
                window_size=window_size,
                overlap_ratio=self.overlap_ratio,
            ):
                batch_regions.append(region)
                batch_starts.append(start)
                
                # Когда набрали батч, обрабатываем
                if len(batch_regions) >= batch_size:
                    for model in models:
                        # Обрабатываем батч патчей одновременно
                        batch_preds_raw = model.model.predict(
                            batch_regions, verbose=False, conf=model.min_conf
                        )
                        
                        # Распределяем результаты по патчам
                        for pred_result, start_coords in zip(batch_preds_raw, batch_starts):
                            # Парсим результаты YOLO для этого патча
                            window_preds = model._parse_single_yolo_prediction(pred_result)
                            
                            for cls_name, preds in window_preds.items():
                                for p in preds:
                                    predictions[cls_name].append(
                                        domain.Prediction(
                                            box=self._to_absolute_box(start_coords, p.box),
                                            polygon=self._to_absolute_polygon(start_coords, p.polygon),
                                            conf=p.conf,
                                        )
                                    )
                    
                    # Очищаем батч
                    batch_regions = []
                    batch_starts = []
            
            # Обрабатываем оставшиеся патчи (если есть)
            if batch_regions:
                for model in models:
                    batch_preds_raw = model.model.predict(
                        batch_regions, verbose=False, conf=model.min_conf
                    )
                    
                    for pred_result, start_coords in zip(batch_preds_raw, batch_starts):
                        window_preds = model._parse_single_yolo_prediction(pred_result)
                        
                        for cls_name, preds in window_preds.items():
                            for p in preds:
                                predictions[cls_name].append(
                                    domain.Prediction(
                                        box=self._to_absolute_box(start_coords, p.box),
                                        polygon=self._to_absolute_polygon(start_coords, p.polygon),
                                        conf=p.conf,
                                    )
                                )

        return predictions

    def _predict_subsection(
        self, section_index: int, subsection_index: int, batch_size: int = 32
    ) -> dict[str, list[domain.Prediction]]:
        """Предсказывает ТОЛЬКО для одной подсекции (кусочка), не для всей секции.
        
        Использует iter_subsection_bgr, который обрабатывает только границы
        конкретной подсекции через скользящее окно.
        
        Для лучшей загрузки GPU собирает патчи в батчи перед обработкой.
        
        Args:
            section_index: Индекс секции
            subsection_index: Индекс подсекции
            batch_size: Размер батча для обработки патчей (по умолчанию 8)
        """
        size_to_models = self._group_models_by_window_size()
        predictions = defaultdict(list[domain.Prediction])

        # iter_subsection_bgr обрабатывает ТОЛЬКО границы подсекции (кусочка),
        # а не всей секции. Это ускоряет обработку.
        for window_size, models in size_to_models.items():
            # Собираем патчи в батчи для лучшей загрузки GPU
            batch_regions = []
            batch_starts = []
            
            for region, start in self.iterator.iter_subsection_bgr(
                section_index=section_index,
                subsection_index=subsection_index,
                window_size=window_size,
                overlap_ratio=self.overlap_ratio,
            ):
                batch_regions.append(region)
                batch_starts.append(start)
                
                # Когда набрали батч, обрабатываем
                if len(batch_regions) >= batch_size:
                    for model in models:
                        # Обрабатываем батч патчей одновременно
                        # model.predict принимает список изображений и возвращает словарь с объединенными результатами
                        # Но нам нужны результаты по каждому патчу отдельно, поэтому используем внутренний метод
                        batch_preds_raw = model.model.predict(
                            batch_regions, verbose=False, conf=model.min_conf
                        )
                        
                        # Распределяем результаты по патчам
                        for pred_result, start_coords in zip(batch_preds_raw, batch_starts):
                            # Парсим результаты YOLO для этого патча
                            window_preds = model._parse_single_yolo_prediction(pred_result)
                            
                            for cls_name, preds in window_preds.items():
                                for p in preds:
                                    predictions[cls_name].append(
                                        domain.Prediction(
                                            box=self._to_absolute_box(start_coords, p.box),
                                            polygon=self._to_absolute_polygon(start_coords, p.polygon),
                                            conf=p.conf,
                                        )
                                    )
                    
                    # Очищаем батч
                    batch_regions = []
                    batch_starts = []
            
            # Обрабатываем оставшиеся патчи (если есть)
            if batch_regions:
                for model in models:
                    batch_preds_raw = model.model.predict(
                        batch_regions, verbose=False, conf=model.min_conf
                    )
                    
                    for pred_result, start_coords in zip(batch_preds_raw, batch_starts):
                        window_preds = model._parse_single_yolo_prediction(pred_result)
                        
                        for cls_name, preds in window_preds.items():
                            for p in preds:
                                predictions[cls_name].append(
                                    domain.Prediction(
                                        box=self._to_absolute_box(start_coords, p.box),
                                        polygon=self._to_absolute_polygon(start_coords, p.polygon),
                                        conf=p.conf,
                                    )
                                )

        return predictions

    def _predict_subsections_parallel(
        self,
        section_index: int,
        subsection_bounds: list
    ) -> dict[str, list[domain.Prediction]]:
        """Параллельно обрабатывает подсекции для лучшей загрузки GPU.
        
        Использует ThreadPoolExecutor для параллельной обработки подсекций.
        YOLO модели могут работать параллельно на одном GPU через потоки.
        
        Args:
            section_index: Индекс секции
            subsection_bounds: Список границ подсекций
            
        Returns:
            Словарь с предсказаниями из всех подсекций
        """
        all_predictions = defaultdict(list[domain.Prediction])
        
        # Определяем количество воркеров
        num_workers = self.max_workers
        if num_workers is None:
            # По умолчанию: минимум из количества подсекций и 4 (для A100 оптимально)
            num_workers = min(len(subsection_bounds), 4)
        
        # Функция для обработки одной подсекции
        def process_subsection(subsection_idx: int) -> tuple[int, dict[str, list[domain.Prediction]]]:
            """Обрабатывает одну подсекцию и возвращает результаты с индексом.
            
            YOLO модели потокобезопасны для работы с GPU, поэтому можно обрабатывать
            несколько подсекций параллельно без блокировок.
            """
            try:
                subsection_preds_raw = self._predict_subsection(
                    section_index, subsection_idx
                )
                return subsection_idx, subsection_preds_raw
            except Exception as e:
                print(f"⚠️  Ошибка при обработке подсекции {subsection_idx}: {e}")
                import traceback
                traceback.print_exc()
                return subsection_idx, {}
        
        # Параллельная обработка подсекций
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Запускаем обработку всех подсекций
            future_to_subsection = {
                executor.submit(process_subsection, idx): idx
                for idx in range(len(subsection_bounds))
            }
            
            # Собираем результаты по мере готовности
            for future in as_completed(future_to_subsection):
                subsection_idx, subsection_preds_raw = future.result()
                
                # Объединяем предикты из всех подсекций
                # ВАЖНО: В областях пересечения подсекций могут быть дубликаты!
                for cls_name, preds in subsection_preds_raw.items():
                    all_predictions[cls_name].extend(preds)
        
        return all_predictions

    def _postprocess_predictions(
        self, preds: list[domain.Prediction], cls_name: str
    ) -> list[domain.Prediction]:
        settings = self.postprocess_settings[cls_name]
        if settings is None:
            raise ValueError(f"No postprocessing settings found for class '{cls_name}'")

        if settings.nms_thres:
            preds = self.__bbox_nms(preds, settings.nms_thres)
        if settings.cov_thres:
            preds = self.__merge_by_coverage(preds, settings.cov_thres)

        return preds

    def __bbox_nms(self, preds: list[domain.Prediction], iou_threshold: float):
        if not preds:
            return []

        boxes = torch.tensor(
            [[p.box.start.x, p.box.start.y, p.box.end.x, p.box.end.y] for p in preds]
        )
        scores = torch.tensor([p.conf for p in preds])
        keep_indices = nms(boxes, scores, iou_threshold)

        return [preds[i] for i in keep_indices]

    def __merge_by_coverage(
        self,
        preds: list[domain.Prediction],
        coverage_threshold: float,
    ) -> list[domain.Prediction]:
        if not preds:
            return []

        preds = sorted(preds, key=lambda x: (x.box.area(), x.conf), reverse=True)
        result: list[domain.Prediction] = []
        for pred in preds:
            merged = pred
            merged_indices = []

            for i, selected_pred in enumerate(result):
                max_coverage = max(
                    pred.covered_by(selected_pred), selected_pred.covered_by(pred)
                )

                if max_coverage > coverage_threshold:
                    merged = merged.merge_with(selected_pred)
                    merged_indices.append(i)

            if merged_indices:
                for i in sorted(merged_indices, reverse=True):
                    result.pop(i)
                result.append(merged)
            else:
                result.append(pred)

        return result

    def _group_models_by_window_size(self) -> dict[int, list[Model]]:
        size_to_models = defaultdict(list[Model])
        for cfg in self.model_configs:
            size_to_models[cfg.window_size].append(cfg.model)
        return size_to_models

    def _to_absolute_polygon(
        self, start: domain.Coords, poly: Optional[domain.Polygon]
    ) -> Optional[domain.Polygon]:
        if not poly:
            return None

        coords = [domain.Coords(c.x + start.x, c.y + start.y) for c in poly.to_coords()]
        return domain.Polygon(coords=coords)

    def _to_absolute_box(self, start: domain.Coords, box: domain.Box) -> domain.Box:
        return domain.Box(
            start=domain.Coords(box.start.x + start.x, box.start.y + start.y),
            end=domain.Coords(box.end.x + start.x, box.end.y + start.y),
        )
