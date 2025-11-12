from . import domain

import cv2
import numpy as np
from typing import Iterator, Optional, Tuple

from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cucim.clara import CuImage


class WSI:
    def __init__(
        self,
        wsi_path: str,
        num_sections: Optional[int] = None,
        num_subsections: Optional[int] = None,
    ):
        self.num_sections = num_sections
        self.num_subsections = num_subsections
        self.min_thumb_width = 1024

        self.wsi = CuImage(wsi_path)
        self.wsi_size = self.wsi.resolutions["level_dimensions"][0]

        self.thumb = self._create_wsi_thumbnail()
        self.thumb_size = (self.thumb.shape[1], self.thumb.shape[0])
        
        # Словарь для хранения разделяющих полигонов между подсекциями
        # Ключ: (section_index, subsection_index1, subsection_index2), значение: domain.Polygon
        self._subsection_separators: dict[tuple[int, int, int], domain.Polygon] = {}
        # Словарь для хранения отфильтрованных точек разделяющих линий
        # Ключ: (section_index, subsection_index1, subsection_index2), значение: list[domain.Coords]
        self._subsection_separator_points: dict[tuple[int, int, int], list[domain.Coords]] = {}

    def iter_first_section_bgr(
        self, window_size: int, overlap_ratio: float
    ) -> Iterator[Tuple[np.ndarray, domain.Coords]]:
        return self.iter_section_bgr(0, window_size, overlap_ratio)

    def iter_section_bgr(
        self, index: int, window_size: int, overlap_ratio: float
    ) -> Iterator[Tuple[np.ndarray, domain.Coords]]:
        assert 0 <= overlap_ratio < 1, "overlap_ratio должен быть в диапазоне [0, 1)"

        bound = self.extract_biopsy_bound(index)

        x_start, y_start = int(bound.start.x), int(bound.start.y)
        w, h = map(int, bound.size())

        stride = int(window_size * (1 - overlap_ratio))

        for window_y in range(0, h, stride):
            for window_x in range(0, w, stride):
                size_x = min(window_size, w - window_x)
                size_y = min(window_size, h - window_y)

                region = self.wsi.read_region(
                    location=(x_start + window_x, y_start + window_y),
                    size=(size_x, size_y),
                    level=0,
                )
                region = cv2.cvtColor(np.asarray(region)[..., :3], cv2.COLOR_RGB2BGR)

                # Координаты верхнего левого угла текущего окна в глобальной системе координат WSI.
                start = domain.Coords((x_start + window_x), (y_start + window_y))

                yield region, start

    def iter_subsection_bgr(
        self, section_index: int, subsection_index: int, window_size: int, overlap_ratio: float
    ) -> Iterator[Tuple[np.ndarray, domain.Coords]]:
        assert 0 <= overlap_ratio < 1, "overlap_ratio должен быть в диапазоне [0, 1)"

        bound = self.extract_subsection_bound(section_index, subsection_index)

        x_start, y_start = int(bound.start.x), int(bound.start.y)
        w, h = map(int, bound.size())

        stride = int(window_size * (1 - overlap_ratio))

        for window_y in range(0, h, stride):
            for window_x in range(0, w, stride):
                size_x = min(window_size, w - window_x)
                size_y = min(window_size, h - window_y)

                region = self.wsi.read_region(
                    location=(x_start + window_x, y_start + window_y),
                    size=(size_x, size_y),
                    level=0,
                )
                region = cv2.cvtColor(np.asarray(region)[..., :3], cv2.COLOR_RGB2BGR)

                # Координаты верхнего левого угла текущего окна в глобальной системе координат WSI.
                start = domain.Coords((x_start + window_x), (y_start + window_y))

                yield region, start

    def visualize_wsi_sections(self) -> np.ndarray:
        bounds = self.extract_biopsy_bounds()

        # Конвертируем thumbnail из RGB в BGR для cv2
        img = self.thumb.copy()
        # Проверяем формат
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]  # Убираем альфа-канал
            # CuImage возвращает RGB, конвертируем в BGR для cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        scale_x = self.thumb_size[0] / self.wsi_size[0]
        scale_y = self.thumb_size[1] / self.wsi_size[1]

        for i, box in enumerate(bounds):
            x1 = int(box.start.x * scale_x)
            y1 = int(box.start.y * scale_y)
            x2 = int(box.end.x * scale_x)
            y2 = int(box.end.y * scale_y)

            # Рисуем прямоугольник (BGR: синий цвет)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Отображение индекса секции (белый текст с черной обводкой)
            text_pos = (x1 + 5, y1 + 30)
            cv2.putText(
                img,
                str(i),
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),  # Белый текст
                3,  # Толщина
                cv2.LINE_AA,
            )
            # Черная обводка для лучшей видимости
            cv2.putText(
                img,
                str(i),
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),  # Черная обводка
                1,
                cv2.LINE_AA,
            )

        return img

    def visualize_wsi_subsections(self, section_index: int) -> np.ndarray:
        """Визуализирует подсекции для указанной секции"""
        section_bound = self.extract_biopsy_bound(section_index)
        subsection_bounds = self.extract_subsection_bounds(section_index)

        # Извлекаем секцию как изображение (RGB)
        section_img = self.extract_biopsy_section(section_index)
        
        # Проверяем формат изображения
        if section_img.ndim == 2:
            # Если grayscale, конвертируем в RGB
            section_img = cv2.cvtColor(section_img, cv2.COLOR_GRAY2RGB)
        elif section_img.shape[2] == 4:
            # Если RGBA, берем только RGB
            section_img = section_img[:, :, :3]
        
        # Конвертируем в BGR для cv2
        section_img = cv2.cvtColor(section_img, cv2.COLOR_RGB2BGR)

        # Размеры изображения и секции
        img_h, img_w = section_img.shape[:2]
        section_w, section_h = map(int, section_bound.size())

        # Масштаб для перевода координат WSI в координаты изображения
        # Изображение секции имеет размеры (img_w, img_h), а секция в WSI имеет размеры (section_w, section_h)
        scale_x = img_w / section_w
        scale_y = img_h / section_h

        for i, box in enumerate(subsection_bounds):
            # Переводим координаты подсекции из WSI в локальную систему секции
            # box.start и box.end - это координаты в WSI
            # section_bound.start - начало секции в WSI
            local_x1 = int((box.start.x - section_bound.start.x) * scale_x)
            local_y1 = int((box.start.y - section_bound.start.y) * scale_y)
            local_x2 = int((box.end.x - section_bound.start.x) * scale_x)
            local_y2 = int((box.end.y - section_bound.start.y) * scale_y)
            
            # Ограничиваем координаты границами изображения
            local_x1 = max(0, min(local_x1, img_w - 1))
            local_y1 = max(0, min(local_y1, img_h - 1))
            local_x2 = max(0, min(local_x2, img_w - 1))
            local_y2 = max(0, min(local_y2, img_h - 1))

            # Рисуем прямоугольник (BGR: зеленый цвет)
            cv2.rectangle(section_img, (local_x1, local_y1), (local_x2, local_y2), (0, 255, 0), 5)

            # Отображение индекса подсекции (белый текст с черной обводкой)
            text_pos = (max(5, local_x1 + 5), max(30, local_y1 + 30))
            # Черная обводка сначала (толще)
            cv2.putText(
                section_img,
                str(i),
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 0),  # Черная обводка
                5,
                cv2.LINE_AA,
            )
            # Белый текст поверх
            cv2.putText(
                section_img,
                str(i),
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (255, 255, 255),  # Белый текст
                3,
                cv2.LINE_AA,
            )
        
        # Рисуем разделяющие ломаные (красные линии)
        # Используем сохраненные отфильтрованные точки разделителя
        for (sec_idx, sub_idx1, sub_idx2) in self._subsection_separator_points.keys():
            if sec_idx == section_index:
                separator_points = self._subsection_separator_points[(sec_idx, sub_idx1, sub_idx2)]
                if len(separator_points) >= 2:
                    # Переводим координаты отфильтрованных точек из WSI в локальную систему секции
                    center_line_points = []
                    for coord in separator_points:
                        local_x = int((coord.x - section_bound.start.x) * scale_x)
                        local_y = int((coord.y - section_bound.start.y) * scale_y)
                        # Ограничиваем координаты
                        local_x = max(0, min(local_x, img_w - 1))
                        local_y = max(0, min(local_y, img_h - 1))
                        center_line_points.append((local_x, local_y))
                    
                    if len(center_line_points) >= 2:
                        # Рисуем ломаную линию (красный цвет, очень толстая линия)
                        # Толщина зависит от размера изображения
                        line_thickness = max(20, int(min(img_w, img_h) / 40))
                        for k in range(len(center_line_points) - 1):
                            cv2.line(section_img, center_line_points[k], center_line_points[k + 1], (0, 0, 255), line_thickness)

        return section_img

    def extract_first_biopsy_section(self) -> np.ndarray:
        return self.extract_biopsy_section(0)

    def extract_biopsy_section(self, index: int) -> np.ndarray:
        bound = self.extract_biopsy_bound(index)

        crop = self.wsi.read_region(
            location=(bound.start.x, bound.start.y),
            size=bound.size(),
            level=0,
        )

        return np.asarray(crop)[..., :3]

    def extract_biopsy_sections(self) -> list[np.ndarray]:
        bounds = self.extract_biopsy_bounds()

        sections = []
        for bound in bounds:
            crop = self.wsi.read_region(
                location=(bound.start.x, bound.start.y), size=bound.size(), level=0
            )

            crop_rgb = np.asarray(crop)[..., :3]
            sections.append(crop_rgb)

        return sections

    def extract_first_biopsy_bound(self) -> domain.Box:
        return self.extract_biopsy_bound(0)

    def extract_biopsy_bound(self, index: int) -> domain.Box:
        return self.extract_biopsy_bounds()[index]

    def extract_biopsy_bounds(self) -> list[domain.Box]:
        clusters = self._cluster_biopsies(self.thumb)

        return self._extract_biopsy_bounds(clusters, self.thumb_size)

    def extract_subsection_bound(self, section_index: int, subsection_index: int) -> domain.Box:
        return self.extract_subsection_bounds(section_index)[subsection_index]

    def extract_subsection_bounds(self, section_index: int) -> list[domain.Box]:
        """Извлекает границы подсекций для указанной секции.
        
        Подсекции - это отдельные связанные компоненты ткани внутри секции.
        Каждый крупный кусок ткани становится отдельной подсекцией.
        Если подсекции пересекаются, они разделяются линией так, чтобы не было пересечений.
        """
        section_bound = self.extract_biopsy_bound(section_index)
        
        # Извлекаем секцию как изображение для анализа
        section_img = self.extract_biopsy_section(section_index)
        
        # Находим отдельные кусочки ткани внутри секции
        # Используем более агрессивную фильтрацию, чтобы найти только крупные компоненты
        regions = self._detect_tissue_regions_for_subsections(section_img)
        
        if not regions:
            # Если регионов нет, возвращаем саму секцию как одну подсекцию
            return [section_bound]
        
        # Сортируем регионы по размеру (от большего к меньшему)
        regions = sorted(regions, key=lambda r: r.area, reverse=True)
        
        # Если указано количество подсекций, берем N самых больших
        if self.num_subsections:
            regions = regions[:self.num_subsections]
        
        # Преобразуем регионы в границы подсекций и получаем контуры
        subsection_bounds = []
        subsection_contours = []  # Сохраняем контуры для каждого кусочка
        section_h, section_w = section_img.shape[:2]
        section_w_actual, section_h_actual = map(int, section_bound.size())
        
        # Масштаб для перевода координат изображения в координаты WSI
        scale_x = section_w_actual / section_w
        scale_y = section_h_actual / section_h
        
        # Создаем маску для каждого региона и находим контуры
        for region in regions:
            # Получаем bbox региона (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = region.bbox
            
            # Создаем маску для этого региона
            region_mask = np.zeros((section_h, section_w), dtype=np.uint8)
            # Заполняем маску пикселями региона
            coords = region.coords  # (row, col) координаты пикселей региона
            if len(coords) > 0:
                region_mask[coords[:, 0], coords[:, 1]] = 255
            
            # Находим контуры региона
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Берем самый большой контур (основной контур региона)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                subsection_contours.append(main_contour)
            else:
                subsection_contours.append(None)
            
            # Переводим координаты в систему координат WSI
            subsection_start = domain.Coords(
                int(section_bound.start.x + min_col * scale_x),
                int(section_bound.start.y + min_row * scale_y)
            )
            subsection_end = domain.Coords(
                int(section_bound.start.x + max_col * scale_x),
                int(section_bound.start.y + max_row * scale_y)
            )
            
            box = domain.Box(start=subsection_start, end=subsection_end)
            subsection_bounds.append(box)
        
        # Разделяем пересекающиеся подсекции, используя контуры
        subsection_bounds = self._split_overlapping_subsections(
            subsection_bounds, section_bound, subsection_contours, section_img.shape, scale_x, scale_y, section_index
        )
        
        return sorted(subsection_bounds, key=lambda x: (x.start.y, x.start.x))

    def _split_overlapping_subsections(
        self, 
        subsection_bounds: list[domain.Box], 
        section_bound: domain.Box,
        subsection_contours: list,
        section_shape: tuple,
        scale_x: float,
        scale_y: float,
        section_index: int = 0
    ) -> list[domain.Box]:
        """Разделяет пересекающиеся подсекции линией, используя контуры регионов ткани.
        
        Использует cv2.findContours для получения контуров и центры масс контуров
        для построения разделяющей линии между двумя подсекциями.
        """
        if len(subsection_bounds) < 2:
            return subsection_bounds
        
        result_bounds = []
        result_contours = []
        
        # Проверяем каждую пару подсекций на пересечение
        for i, box1 in enumerate(subsection_bounds):
            if i == 0:
                # Первая подсекция добавляется как есть
                result_bounds.append(box1)
                result_contours.append(subsection_contours[i] if i < len(subsection_contours) else None)
                continue
            
            # Проверяем пересечение с уже обработанными подсекциями
            current_box = box1
            current_contour = subsection_contours[i] if i < len(subsection_contours) else None
            
            for j, box2 in enumerate(result_bounds):
                if current_box.intersects_with(box2):
                    # Находим область пересечения
                    inter_start = domain.Coords(
                        max(current_box.start.x, box2.start.x),
                        max(current_box.start.y, box2.start.y)
                    )
                    inter_end = domain.Coords(
                        min(current_box.end.x, box2.end.x),
                        min(current_box.end.y, box2.end.y)
                    )
                    
                    if inter_start.x < inter_end.x and inter_start.y < inter_end.y:
                        # Есть пересечение - используем контуры для построения разделяющей линии
                        contour2 = result_contours[j] if j < len(result_contours) else None
                        
                        # Вычисляем центры масс контуров
                        if current_contour is not None and contour2 is not None:
                            # Используем cv2.moments для вычисления центра масс контура
                            M1 = cv2.moments(current_contour)
                            M2 = cv2.moments(contour2)
                            
                            if M1["m00"] > 0 and M2["m00"] > 0:
                                # Центр масс контура в координатах изображения (col, row)
                                c1_col = M1["m10"] / M1["m00"]
                                c1_row = M1["m01"] / M1["m00"]
                                c2_col = M2["m10"] / M2["m00"]
                                c2_row = M2["m01"] / M2["m00"]
                                
                                # Переводим в координаты WSI
                                center1 = domain.Coords(
                                    section_bound.start.x + c1_col * scale_x,
                                    section_bound.start.y + c1_row * scale_y
                                )
                                center2 = domain.Coords(
                                    section_bound.start.x + c2_col * scale_x,
                                    section_bound.start.y + c2_row * scale_y
                                )
                            else:
                                # Fallback: используем центры боксов
                                center1 = current_box.center()
                                center2 = box2.center()
                        else:
                            # Fallback: используем центры боксов
                            center1 = current_box.center()
                            center2 = box2.center()
                        
                        # Линия между центрами масс: вектор (dx, dy)
                        dx = center2.x - center1.x
                        dy = center2.y - center1.y
                        
                        # Разделяем пересекающуюся область по диагонали через центр пересечения
                        # Используем наклон линии между центрами масс контуров
                        inter_center_x = (inter_start.x + inter_end.x) / 2
                        inter_center_y = (inter_start.y + inter_end.y) / 2
                        
                        # Создаем разделяющий полигон (ломаная) для области пересечения
                        # Переводим координаты контуров для проверки пересечений
                        section_start_coords = section_bound.start
                        separator_polygon, filtered_separator_points = self._create_separator_polygon(
                            inter_start, inter_end, center1, center2, dx, dy,
                            current_contour, contour2, scale_x, scale_y, section_start_coords
                        )
                        
                        # Сохраняем разделяющий полигон и отфильтрованные точки для использования при предикте и визуализации
                        self._subsection_separators[(section_index, j, i)] = separator_polygon
                        self._subsection_separator_points[(section_index, j, i)] = filtered_separator_points
                        
                        # Разделяем пересечение по диагонали
                        # Упрощенный подход: разделяем по линии, перпендикулярной к линии между центрами
                        # через центр пересечения
                        
                        # Перпендикуляр к линии между центрами имеет наклон -dx/dy
                        if abs(dy) > 1e-6:
                            k_perp = -dx / dy
                            
                            # Разделяем по перпендикуляру через центр пересечения
                            # Определяем, с какой стороны перпендикуляра находится каждый центр
                            def perp_side(x, y):
                                return y - k_perp * x + k_perp * inter_center_x - inter_center_y
                            
                            center1_perp_side = perp_side(center1.x, center1.y)
                            center2_perp_side = perp_side(center2.x, center2.y)
                            
                            # Разделяем в зависимости от направления перпендикуляра
                            if abs(k_perp) < 1:
                                # Более горизонтальная линия - разделяем вертикально
                                split_x = inter_center_x
                                if center1_perp_side > center2_perp_side:
                                    # center1 справа - current_box получает правую часть пересечения
                                    if box2.end.x > split_x:
                                        box2 = domain.Box(
                                            start=box2.start,
                                            end=domain.Coords(split_x, box2.end.y)
                                        )
                                        result_bounds[j] = box2
                                    if current_box.start.x < split_x:
                                        current_box = domain.Box(
                                            start=domain.Coords(split_x, current_box.start.y),
                                            end=current_box.end
                                        )
                                else:
                                    # center2 справа - box2 получает правую часть пересечения
                                    if current_box.end.x > split_x:
                                        current_box = domain.Box(
                                            start=current_box.start,
                                            end=domain.Coords(split_x, current_box.end.y)
                                        )
                                    if box2.start.x < split_x:
                                        box2 = domain.Box(
                                            start=domain.Coords(split_x, box2.start.y),
                                            end=box2.end
                                        )
                                        result_bounds[j] = box2
                            else:
                                # Более вертикальная линия - разделяем горизонтально
                                split_y = inter_center_y
                                if center1_perp_side > center2_perp_side:
                                    # center1 выше - current_box получает верхнюю часть пересечения
                                    if box2.end.y > split_y:
                                        box2 = domain.Box(
                                            start=box2.start,
                                            end=domain.Coords(box2.end.x, split_y)
                                        )
                                        result_bounds[j] = box2
                                    if current_box.start.y < split_y:
                                        current_box = domain.Box(
                                            start=domain.Coords(current_box.start.x, split_y),
                                            end=current_box.end
                                        )
                                else:
                                    # center2 выше - box2 получает верхнюю часть пересечения
                                    if current_box.end.y > split_y:
                                        current_box = domain.Box(
                                            start=current_box.start,
                                            end=domain.Coords(current_box.end.x, split_y)
                                        )
                                    if box2.start.y < split_y:
                                        box2 = domain.Box(
                                            start=domain.Coords(box2.start.x, split_y),
                                            end=box2.end
                                        )
                                        result_bounds[j] = box2
                        else:
                            # Почти горизонтальная линия между центрами - разделяем вертикально
                            split_x = inter_center_x
                            if center1.x > center2.x:
                                # center1 правее - current_box получает правую часть пересечения
                                if box2.end.x > split_x:
                                    box2 = domain.Box(
                                        start=box2.start,
                                        end=domain.Coords(split_x, box2.end.y)
                                    )
                                    result_bounds[j] = box2
                                if current_box.start.x < split_x:
                                    current_box = domain.Box(
                                        start=domain.Coords(split_x, current_box.start.y),
                                        end=current_box.end
                                    )
                            else:
                                # center2 правее - box2 получает правую часть пересечения
                                if current_box.end.x > split_x:
                                    current_box = domain.Box(
                                        start=current_box.start,
                                        end=domain.Coords(split_x, current_box.end.y)
                                    )
                                if box2.start.x < split_x:
                                    box2 = domain.Box(
                                        start=domain.Coords(split_x, box2.start.y),
                                        end=box2.end
                                    )
                                    result_bounds[j] = box2
                    else:
                        # Почти вертикальная линия между центрами - разделяем горизонтально
                        split_y = inter_center_y
                        if center1.y < center2.y:
                            # center1 выше - current_box получает верхнюю часть пересечения
                            if box2.end.y > split_y:
                                box2 = domain.Box(
                                    start=box2.start,
                                    end=domain.Coords(box2.end.x, split_y)
                                )
                                result_bounds[j] = box2
                            if current_box.start.y < split_y:
                                current_box = domain.Box(
                                    start=domain.Coords(current_box.start.x, split_y),
                                    end=current_box.end
                                )
                        else:
                            # center2 выше - box2 получает верхнюю часть пересечения
                            if current_box.end.y > split_y:
                                current_box = domain.Box(
                                    start=current_box.start,
                                    end=domain.Coords(current_box.end.x, split_y)
                                )
                            if box2.start.y < split_y:
                                box2 = domain.Box(
                                    start=domain.Coords(box2.start.x, split_y),
                                    end=box2.end
                                )
                                result_bounds[j] = box2
            
            result_bounds.append(current_box)
            result_contours.append(current_contour)
        
        return result_bounds

    def _create_separator_polygon(
        self,
        inter_start: domain.Coords,
        inter_end: domain.Coords,
        center1: domain.Coords,
        center2: domain.Coords,
        dx: float,
        dy: float,
        contour1: Optional[np.ndarray] = None,
        contour2: Optional[np.ndarray] = None,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        section_start: Optional[domain.Coords] = None
    ) -> tuple[domain.Polygon, list[domain.Coords]]:
        """Создает разделяющий полигон (ломаная) для области пересечения.
        
        Полигон разделяет область пересечения на две части, каждая из которых
        принадлежит соответствующей подсекции. Удаляет звенья ломаной, которые
        пересекают сами кусочки ткани.
        """
        inter_center_x = (inter_start.x + inter_end.x) / 2
        inter_center_y = (inter_start.y + inter_end.y) / 2
        
        # Создаем начальную ломаную через область пересечения
        # Используем наклон, перпендикулярный к линии между центрами
        if abs(dy) > 1e-6:
            k_perp = -dx / dy  # Наклон перпендикуляра
            
            # Создаем ломаную линию через область пересечения
            # Начинаем с простой диагонали через центр пересечения
            # Создаем несколько точек для ломаной
            num_points = 10  # Количество точек для ломаной
            polyline_points = []
            
            if abs(k_perp) < 1:
                # Более горизонтальная линия - ломаная идет вертикально через центр
                for i in range(num_points + 1):
                    y = inter_start.y + (inter_end.y - inter_start.y) * i / num_points
                    x = inter_center_x
                    polyline_points.append(domain.Coords(x, y))
            else:
                # Более вертикальная линия - ломаная идет горизонтально через центр
                for i in range(num_points + 1):
                    x = inter_start.x + (inter_end.x - inter_start.x) * i / num_points
                    y = inter_center_y
                    polyline_points.append(domain.Coords(x, y))
        else:
            # Почти горизонтальная линия между центрами - ломаная идет вертикально
            num_points = 10
            polyline_points = []
            for i in range(num_points + 1):
                y = inter_start.y + (inter_end.y - inter_start.y) * i / num_points
                x = inter_center_x
                polyline_points.append(domain.Coords(x, y))
        
        # Фильтруем звенья ломаной: удаляем те, которые пересекают кусочки ткани
        if contour1 is not None and contour2 is not None and section_start is not None:
            filtered_points = []
            
            # НЕ добавляем первую точку автоматически - будем добавлять только те точки,
            # звенья от которых не пересекают кусочки
            
            # Проверяем каждое звено ломаной на пересечение с контурами
            for i in range(len(polyline_points)):
                point = polyline_points[i]
                
                # Переводим координаты в систему координат изображения для проверки
                img_x = (point.x - section_start.x) / scale_x
                img_y = (point.y - section_start.y) / scale_y
                
                # Проверяем, находится ли точка внутри контуров
                point_inside_contour1 = cv2.pointPolygonTest(contour1, (img_x, img_y), False) >= 0
                point_inside_contour2 = cv2.pointPolygonTest(contour2, (img_x, img_y), False) >= 0
                
                # Проверяем звено (отрезок) на пересечение с контурами
                segment_goes_through_contour1 = False
                segment_goes_through_contour2 = False
                
                if len(filtered_points) > 0:
                    # Проверяем звено от последней добавленной точки до текущей
                    prev_point = filtered_points[-1]
                    prev_img_x = (prev_point.x - section_start.x) / scale_x
                    prev_img_y = (prev_point.y - section_start.y) / scale_y
                    
                    # Проверяем расстояние до контуров для обеих точек звена
                    prev_dist1 = cv2.pointPolygonTest(contour1, (prev_img_x, prev_img_y), True)
                    prev_dist2 = cv2.pointPolygonTest(contour2, (prev_img_x, prev_img_y), True)
                    curr_dist1 = cv2.pointPolygonTest(contour1, (img_x, img_y), True)
                    curr_dist2 = cv2.pointPolygonTest(contour2, (img_x, img_y), True)
                    
                    # Строгая проверка: звено проходит через кусок, если оно пересекает контур
                    # Проверяем, пересекает ли отрезок контур, используя более точный метод
                    
                    # Создаем маску для проверки пересечения
                    # Проверяем несколько точек на отрезке с высокой точностью
                    num_checks = 200  # Очень много точек для точной проверки
                    points_inside1 = 0
                    points_inside2 = 0
                    max_dist_inside1 = -float('inf')
                    max_dist_inside2 = -float('inf')
                    
                    for j in range(num_checks + 1):
                        t = j / num_checks
                        check_x = prev_img_x + (img_x - prev_img_x) * t
                        check_y = prev_img_y + (img_y - prev_img_y) * t
                        
                        # Проверяем расстояние до контуров (положительное = внутри, отрицательное = снаружи)
                        dist1 = cv2.pointPolygonTest(contour1, (check_x, check_y), True)
                        dist2 = cv2.pointPolygonTest(contour2, (check_x, check_y), True)
                        
                        # Если точка внутри контура (расстояние > 0), считаем её
                        if dist1 > 0:
                            points_inside1 += 1
                            max_dist_inside1 = max(max_dist_inside1, dist1)
                        if dist2 > 0:
                            points_inside2 += 1
                            max_dist_inside2 = max(max_dist_inside2, dist2)
                    
                    # Звено проходит через кусок, если ЛЮБАЯ его точка внутри контура
                    # (даже если это всего одна точка из 200)
                    # Это очень строгий критерий - удаляем звено, если оно хоть немного заходит внутрь кусочка
                    if points_inside1 > 0:
                        segment_goes_through_contour1 = True
                    if points_inside2 > 0:
                        segment_goes_through_contour2 = True
                    
                    # Дополнительная проверка: если хотя бы одна из конечных точек звена внутри контура
                    if prev_dist1 > 0 or curr_dist1 > 0:
                        segment_goes_through_contour1 = True
                    if prev_dist2 > 0 or curr_dist2 > 0:
                        segment_goes_through_contour2 = True
                
                # Удаляем точку, если:
                # - Точка внутри любого из контуров (внутри кусочка), ИЛИ
                # - Звено пересекает любой из контуров (проходит через кусочки)
                # Мы хотим оставить только те звенья, которые проходят через область пересечения,
                # но не заходят внутрь самих кусочков
                
                # Проверяем, находится ли точка внутри контуров
                point_inside_contour1 = cv2.pointPolygonTest(contour1, (img_x, img_y), False) >= 0
                point_inside_contour2 = cv2.pointPolygonTest(contour2, (img_x, img_y), False) >= 0
                
                # Удаляем точку, если:
                # 1. Точка внутри любого из контуров (внутри кусочка), ИЛИ
                # 2. Звено пересекает любой из контуров (проходит через кусочки)
                # Исключение: если точка находится на границе обоих контуров одновременно
                # (это область пересечения), то оставляем её
                
                # Проверяем, находится ли точка на границе (расстояние близко к 0)
                dist_to_contour1 = cv2.pointPolygonTest(contour1, (img_x, img_y), True)
                dist_to_contour2 = cv2.pointPolygonTest(contour2, (img_x, img_y), True)
                
                # Порог для определения "на границе" (в пикселях)
                boundary_threshold = 2.0
                on_boundary1 = abs(dist_to_contour1) < boundary_threshold
                on_boundary2 = abs(dist_to_contour2) < boundary_threshold
                
                # Удаляем точку, если:
                # - Точка глубоко внутри любого из контуров (внутри кусочка), ИЛИ
                # - Звено проходит через любой из контуров (проходит через кусочки)
                # НО оставляем, если точка на границе обоих контуров (область пересечения)
                
                # Удаляем звено целиком, если оно проходит через кусочки
                # Это главный критерий - если звено проходит через внутреннюю часть кусочка, не добавляем текущую точку
                if segment_goes_through_contour1 or segment_goes_through_contour2:
                    continue  # Пропускаем эту точку - звено проходит через кусочки, не добавляем её
                
                # Дополнительная проверка: если точка глубоко внутри контура (не на границе)
                # и не в области пересечения, тоже не добавляем
                deep_threshold = 2.0
                deep_inside1 = dist_to_contour1 > deep_threshold
                deep_inside2 = dist_to_contour2 > deep_threshold
                
                # Не добавляем точку, если она глубоко внутри одного из контуров
                # НО добавляем, если она на границе обоих контуров (область пересечения)
                if (deep_inside1 or deep_inside2) and not (on_boundary1 and on_boundary2):
                    continue  # Пропускаем эту точку - она глубоко внутри кусочка
                
                # Добавляем точку только если звено не проходит через кусочки
                filtered_points.append(point)
            
            # Если после фильтрации осталось меньше 2 точек, значит все звенья пересекают кусочки
            # В этом случае не рисуем разделяющую линию вообще (возвращаем пустой список)
            if len(filtered_points) < 2:
                filtered_points = []  # Не рисуем линию, если все звенья пересекают кусочки
        else:
            filtered_points = polyline_points
        
        # Создаем полигон из отфильтрованных точек
        # Используем небольшую ширину для создания "толстой" линии
        if len(filtered_points) < 2:
            # Если точек недостаточно, создаем простой прямоугольник
            strip_width = min(inter_end.x - inter_start.x, inter_end.y - inter_start.y) * 0.01
            coords = [
                domain.Coords(inter_center_x - strip_width, inter_start.y),
                domain.Coords(inter_center_x - strip_width, inter_end.y),
                domain.Coords(inter_center_x + strip_width, inter_end.y),
                domain.Coords(inter_center_x + strip_width, inter_start.y),
            ]
        else:
            # Создаем полигон-полосу из ломаной
            strip_width = min(inter_end.x - inter_start.x, inter_end.y - inter_start.y) * 0.01
            
            # Создаем полигон, "расширяя" ломаную в перпендикулярном направлении
            coords = []
            
            # Определяем направление перпендикуляра
            if abs(dy) > 1e-6:
                k_perp = -dx / dy
                if abs(k_perp) < 1:
                    # Горизонтальная полоса
                    for point in filtered_points:
                        coords.append(domain.Coords(point.x, point.y - strip_width))
                    # Добавляем точки в обратном порядке для второй стороны
                    for point in reversed(filtered_points):
                        coords.append(domain.Coords(point.x, point.y + strip_width))
                else:
                    # Вертикальная полоса
                    for point in filtered_points:
                        coords.append(domain.Coords(point.x - strip_width, point.y))
                    for point in reversed(filtered_points):
                        coords.append(domain.Coords(point.x + strip_width, point.y))
            else:
                # Вертикальная полоса
                for point in filtered_points:
                    coords.append(domain.Coords(point.x - strip_width, point.y))
                for point in reversed(filtered_points):
                    coords.append(domain.Coords(point.x + strip_width, point.y))
        
        try:
            return domain.Polygon(coords)
        except domain.BadPolygonError:
            # Если не получилось создать полигон, создаем простой прямоугольник
            strip_width = min(inter_end.x - inter_start.x, inter_end.y - inter_start.y) * 0.01
            coords = [
                domain.Coords(inter_center_x - strip_width, inter_start.y),
                domain.Coords(inter_center_x - strip_width, inter_end.y),
                domain.Coords(inter_center_x + strip_width, inter_end.y),
                domain.Coords(inter_center_x + strip_width, inter_start.y),
            ]
            polygon = domain.Polygon(coords)
            return polygon, filtered_points

    def get_subsection_for_prediction(
        self, section_index: int, prediction: domain.Prediction
    ) -> Optional[int]:
        """Определяет, к какой подсекции относится предикт.
        
        Возвращает индекс подсекции или None, если не удалось определить.
        """
        subsections = self.extract_subsection_bounds(section_index)
        
        # Проверяем, в какую подсекцию попадает центр предикта
        pred_center = prediction.box.center()
        
        for i, subsection in enumerate(subsections):
            # Проверяем, находится ли центр предикта в подсекции
            if (subsection.start.x <= pred_center.x <= subsection.end.x and
                subsection.start.y <= pred_center.y <= subsection.end.y):
                
                # Проверяем, не находится ли предикт в разделяющей области
                # (в этом случае нужно проверить, с какой стороны разделителя он находится)
                for (sec_idx, sub_idx1, sub_idx2), separator in self._subsection_separators.items():
                    if sec_idx == section_index and (sub_idx1 == i or sub_idx2 == i):
                        # Проверяем, пересекается ли предикт с разделителем
                        if prediction.polygon:
                            # Если есть полигон, проверяем пересечение
                            # TODO: более точная проверка с использованием shapely
                            pass
                        else:
                            # Если нет полигона, используем центр
                            # TODO: проверка принадлежности центра к разделителю
                            pass
                
                return i
        
        return None

    def _detect_tissue_regions_for_subsections(self, img: np.ndarray) -> list:
        """Находит отдельные крупные кусочки ткани для подсекций.
        
        Использует более агрессивную фильтрацию, чем _detect_tissue_regions,
        чтобы найти только крупные отдельные компоненты.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        lower_tissue = np.array([0, 30, 30])
        upper_tissue = np.array([180, 255, 255])
        tissue_mask = cv2.inRange(hsv, lower_tissue, upper_tissue)
        
        # Более агрессивная морфология для объединения близких регионов
        kernel_large = np.ones((15, 15), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel_large)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel_large)
        
        labeled_image = label(tissue_mask)
        regions = regionprops(labeled_image)
        
        if not regions:
            return []
        
        # Фильтруем по площади - берем только крупные компоненты
        areas = np.array([r.area for r in regions])
        # Используем более высокий порог - например, медиана * 0.5 или минимум 10% от максимальной площади
        min_area = max(np.median(areas) * 0.5, np.max(areas) * 0.1)
        
        large_regions = [r for r in regions if r.area >= min_area]
        
        return large_regions

    def _extract_biopsy_bounds(
        self, biopsy_clusters, thumbnail_size
    ) -> list[domain.Box]:
        biopsy_regions = []

        for cluster in biopsy_clusters:
            min_row = min(r.bbox[0] for r in cluster)
            min_col = min(r.bbox[1] for r in cluster)
            max_row = max(r.bbox[2] for r in cluster)
            max_col = max(r.bbox[3] for r in cluster)

            scale_x = self.wsi_size[0] / thumbnail_size[0]
            scale_y = self.wsi_size[1] / thumbnail_size[1]

            box = domain.Box(
                domain.Coords(int(min_col * scale_x), int(min_row * scale_y)),
                domain.Coords(int(max_col * scale_x), int(max_row * scale_y)),
            )

            biopsy_regions.append(box)

        return sorted(biopsy_regions, key=lambda x: (x.start.y, x.start.x))

    def _detect_tissue_regions(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_tissue = np.array([0, 30, 30])
        upper_tissue = np.array([180, 255, 255])
        tissue_mask = cv2.inRange(hsv, lower_tissue, upper_tissue)

        kernel = np.ones((5, 5), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

        labeled_image = label(tissue_mask)
        regions = regionprops(labeled_image)

        areas = np.array([r.area for r in regions])
        min_area = np.median(areas) * 0.3

        return [r for r in regions if r.area >= min_area]

    def _cluster_biopsies(self, img):
        regions = self._detect_tissue_regions(img)

        if not regions:
            return []

        centroids = np.array([[r.centroid[1], r.centroid[0]] for r in regions])

        if self.num_sections:
            n_clusters = min(self.num_sections, len(regions))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(centroids)
        else:
            labels = self._find_optimal_clusters(centroids)

        n_clusters = len(set(labels))
        clusters = [[] for _ in range(n_clusters)]
        for region, label in zip(regions, labels):
            clusters[label].append(region)

        return clusters

    def _find_optimal_clusters(self, centroids):
        n_min = 1
        n_max = n_max = min(16, len(centroids) - 1)

        best_score = -1
        best_labels = None

        for n in range(n_min, n_max + 1):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(centroids)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(centroids, labels)
            if score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is None:
            kmeans = KMeans(n_clusters=n_min, random_state=42, n_init=10)
            best_labels = kmeans.fit_predict(centroids)

        return best_labels

    def _create_wsi_thumbnail(self):
        target_level = self._choose_best_level()
        width, height = self.wsi.resolutions["level_dimensions"][target_level]

        thumbnail = self.wsi.read_region(
            location=(0, 0),
            size=(width, height),
            level=target_level,
        )

        return np.asarray(thumbnail)

    def _choose_best_level(self) -> int:
        widths = [res[0] for res in self.wsi.resolutions["level_dimensions"]]
        candidates = [
            (level, w) for level, w in enumerate(widths) if w >= self.min_thumb_width
        ]

        if candidates:
            return min(candidates, key=lambda x: x[1])[0]

        return int(np.argmax(widths))
