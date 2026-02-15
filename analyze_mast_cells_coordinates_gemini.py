#!/usr/bin/env python3
"""
Скрипт для анализа координат мастоцитов через Gemini 3 Pro.

Задача:
1. Напомнить Gemini о явно и неявно выраженных мастоцитах из предыдущего теста
2. Попросить определить координаты мастоцитов на картинках с расширением _no
3. Сделать анализ ошибок на основании тех же картинок с разметкой _yes
4. Выяснить, что нужно Gemini для улучшения обнаружения неявно выраженных мастоцитов
"""

import os
import sys
import base64
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx
from PIL import Image
import io
import json

# Добавляем путь к brats для импорта клиента Gemini
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brats" / "kb-service"))

# Загружаем конфигурацию из ../brats/config.env
brats_config_path = Path(__file__).resolve().parent.parent / "brats" / "config.env"
kb_service_config_path = Path(__file__).resolve().parent.parent / "brats" / "kb-service" / "config.env"

# Загружаем конфигурацию (приоритет: kb-service/config.env, затем brats/config.env)
if kb_service_config_path.exists():
    load_dotenv(dotenv_path=kb_service_config_path, override=True)
if brats_config_path.exists():
    load_dotenv(dotenv_path=brats_config_path, override=False)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiVisionService:
    """Клиент для работы с Gemini 3 Pro Vision API через ProxyAPI.ru"""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("GEMINI_BASE_URL", "https://api.proxyapi.ru/google")
        self.model = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
        self.timeout = int(os.getenv("GEMINI_TIMEOUT", "120"))
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY или OPENAI_API_KEY не настроен. "
                "Проверьте config.env в kb-service или brats директории."
            )
        
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        
        logger.info(
            f"✅ GeminiVisionService инициализирован (model={self.model}, base_url={self.base_url})"
        )
    
    def _encode_image(self, image_path: Path, max_size_mb: float = 4.0, preserve_resolution: bool = False) -> Dict[str, str]:
        """
        Кодирует изображение в base64 для Gemini API.
        Уменьшает размер изображения, если оно слишком большое.
        
        Args:
            image_path: Путь к изображению
            max_size_mb: Максимальный размер в МБ после кодирования (по умолчанию 4 МБ)
            preserve_resolution: Если True, пытается сохранить большее разрешение (до 8 МБ)
        
        Returns:
            Словарь с inline_data для Gemini API
        """
        # Определяем MIME тип по расширению
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        
        # Для сохранения разрешения увеличиваем лимит
        if preserve_resolution:
            max_size_mb = 8.0
            max_dimension = 4096
        else:
            max_dimension = 2048
        
        # Читаем и обрабатываем изображение
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            logger.info(f"📐 Исходное разрешение {image_path.name}: {original_width}x{original_height}")
            
            # Конвертируем в RGB, если нужно (для PNG с альфа-каналом)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Проверяем размер и уменьшаем, если нужно
            original_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            logger.debug(f"Размер изображения {image_path.name}: {original_size_mb:.2f} МБ")
            
            # Уменьшаем изображение, если оно слишком большое
            quality = 95
            
            # Если изображение большое, уменьшаем его
            if img.width > max_dimension or img.height > max_dimension:
                ratio = min(max_dimension / img.width, max_dimension / img.height)
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"🖼️ Изображение {image_path.name} уменьшено до {new_width}x{new_height}")
            
            # Сохраняем в буфер с оптимизацией качества
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            image_data = buffer.getvalue()
            
            # Проверяем размер после кодирования
            encoded_size_mb = len(image_data) / (1024 * 1024)
            logger.debug(f"Размер после обработки {image_path.name}: {encoded_size_mb:.2f} МБ")
            
            # Если все еще слишком большое, уменьшаем качество
            if encoded_size_mb > max_size_mb:
                for q in [85, 75, 65, 55]:
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=q, optimize=True)
                    test_data = buffer.getvalue()
                    if len(test_data) / (1024 * 1024) <= max_size_mb:
                        image_data = test_data
                        logger.info(f"🖼️ Качество {image_path.name} снижено до {q}% для уменьшения размера")
                        break
        
        return {
            "inline_data": {
                "mime_type": "image/jpeg",  # Всегда JPEG после обработки
                "data": base64.b64encode(image_data).decode("utf-8")
            }
        }
    
    async def analyze_images(
        self,
        image_paths: List[Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        image_labels: Optional[List[str]] = None,
        preserve_resolution: bool = False,
    ) -> str:
        """
        Анализирует изображения через Gemini Vision API.
        
        Args:
            image_paths: Список путей к изображениям
            prompt: Промпт для анализа
            system_prompt: Системный промпт (опционально)
            image_labels: Метки для изображений (для идентификации в промпте)
            preserve_resolution: Сохранять большее разрешение для детального анализа
        
        Returns:
            Текст ответа от Gemini
        """
        # Формируем parts: сначала текст, затем изображения
        parts = [{"text": prompt}]
        
        # Добавляем изображения
        for idx, img_path in enumerate(image_paths):
            if not img_path.exists():
                logger.warning(f"⚠️ Изображение не найдено: {img_path}")
                continue
            parts.append(self._encode_image(img_path, preserve_resolution=preserve_resolution))
        
        request_data: Dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 8192,  # Увеличиваем для детальных ответов
            },
        }
        
        if system_prompt:
            request_data["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }
        
        model_endpoint = f"/v1beta/models/{self.model}:generateContent"
        
        try:
            logger.info(f"📤 Отправка запроса к Gemini с {len(image_paths)} изображениями...")
            response = await self._client.post(
                model_endpoint,
                json=request_data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            candidates = data.get("candidates", []) or []
            if not candidates:
                logger.warning("⚠️ Gemini вернул пустой список candidates")
                return "Ошибка: пустой ответ от модели"
            
            content = candidates[0].get("content", {})
            parts_out = content.get("parts", []) or []
            if not parts_out:
                logger.warning("⚠️ Gemini вернул пустые parts в content")
                return "Ошибка: пустой контент в ответе"
            
            text = parts_out[0].get("text", "") or ""
            if not text:
                logger.warning("⚠️ Gemini вернул пустой text")
                return "Ошибка: пустой текст в ответе"
            
            return text
        
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            logger.error(f"❌ HTTP ошибка: {e.response.status_code} - {error_text}")
            
            # Для ошибки 402 возвращаем специальный маркер
            if e.response.status_code == 402:
                return f"ERROR_402: {error_text}"
            
            return f"Ошибка HTTP {e.response.status_code}: {error_text}"
        except Exception as e:
            logger.error(f"❌ Ошибка запроса к Gemini: {e}", exc_info=True)
            return f"Ошибка: {str(e)}"
    
    async def close(self):
        """Закрывает HTTP клиент"""
        await self._client.aclose()


def load_previous_analysis_summary() -> str:
    """Загружает краткое резюме предыдущего анализа для контекста"""
    result_file = Path(__file__).parent / "mast_cells_analysis_result.txt"
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Извлекаем ключевые выводы из предыдущего анализа
        summary = """
КРАТКОЕ РЕЗЮМЕ ПРЕДЫДУЩЕГО АНАЛИЗА:

Из предыдущего теста ты уже знаешь о мастоцитах следующее:

1. ЯВНО ВЫРАЖЕННЫЕ МАСТОЦИТЫ (зеленые аннотации):
   - Округлая или слегка овальная форма
   - Центрально расположенное, округлое или овальное ядро темно-фиолетового цвета
   - Эозинофильная (розоватая) зернистая цитоплазма вокруг ядра
   - Четкие границы, хороший контраст с фоном
   - Расположение в строме (соединительной ткани) между криптами
   - Визуальный образ: "глазунья" - темный "желток" (ядро) и мутноватый розовый "белок" (цитоплазма)

2. НЕЯВНО ВЫРАЖЕННЫЕ МАСТОЦИТЫ (синие аннотации):
   - Овальная или слегка вытянутая форма
   - Центрально расположенное ядро (ключевой признак для отличия от плазмоцитов)
   - Розовая цитоплазма, но с размытыми границами
   - Зернистость плохо различима или не видна
   - Сливаются с фоном, низкий контраст
   - Легко спутать с фибробластами, гистиоцитами или макрофагами
   - Могут быть веретеновидной формы (мимикрируют под фибробласты)

3. КЛЮЧЕВЫЕ ПРИЗНАКИ ДЛЯ ОТЛИЧЕНИЯ:
   - От плазмоцитов: у мастоцитов ядро по центру, у плазмоцитов - сдвинуто к краю
   - От лимфоцитов: у мастоцитов есть ободок розовой цитоплазмы, у лимфоцитов цитоплазмы почти нет
   - От фибробластов: у мастоцитов ядро округлое, у фибробластов - вытянутое веретеновидное

4. ОГРАНИЧЕНИЯ ПРЕДЫДУЩЕГО АНАЛИЗА:
   - На г/э уверенно определялись только "классические" мастоциты
   - Часть мастоцитов терялась (особенно дегранулированных или срезанных по краю)
   - ИГХ показывало больше клеток, чем было видно на г/э
   - Для неявных мастоцитов требовался ИГХ контроль для 100% уверенности
"""
        return summary
    return ""


async def analyze_coordinates_and_errors():
    """Основная функция анализа координат и ошибок"""
    
    # Путь к директории с изображениями
    mast_dir = Path(__file__).parent / "MAST_GEMINI"
    
    if not mast_dir.exists():
        logger.error(f"❌ Директория не найдена: {mast_dir}")
        return
    
    # Находим пары изображений _no и _yes
    images_no = sorted(mast_dir.glob("*_no.png"))
    images_yes = sorted(mast_dir.glob("*_yes.png"))
    
    logger.info(f"Найдено изображений _no: {len(images_no)}")
    logger.info(f"Найдено изображений _yes: {len(images_yes)}")
    
    if not images_no:
        logger.error("❌ Не найдено изображений с расширением _no")
        return
    
    # Инициализируем клиент Gemini
    try:
        gemini = GeminiVisionService()
    except ValueError as e:
        logger.error(f"❌ {e}")
        return
    
    # Загружаем резюме предыдущего анализа
    previous_summary = load_previous_analysis_summary()
    
    # Системный промпт
    system_prompt = """Ты эксперт-патолог, специализирующийся на анализе гистологических изображений.
Твоя задача - найти мастоциты на неокрашенных патчах (гематоксилин-эозин) и определить их координаты.
Ты уже знаешь о явно и неявно выраженных мастоцитах из предыдущего анализа.
Будь максимально точным и детальным. Используй все свои знания о морфологии мастоцитов."""
    
    results = []
    
    try:
        for img_no in images_no:
            # Находим соответствующее изображение _yes
            img_yes = None
            base_name = img_no.stem.replace("_no", "")
            for yes_img in images_yes:
                if yes_img.stem.replace("_yes", "") == base_name:
                    img_yes = yes_img
                    break
            
            if not img_yes:
                logger.warning(f"⚠️ Не найдено соответствующее изображение _yes для {img_no.name}")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Обработка пары: {img_no.name} / {img_yes.name}")
            logger.info(f"{'='*80}\n")
            
            # ЭТАП 1: Определение координат на _no изображении
            prompt_step1 = f"""{previous_summary}

ЗАДАЧА ЭТАПА 1: ОПРЕДЕЛЕНИЕ КООРДИНАТ МАСТОЦИТОВ

Тебе предоставлено изображение {img_no.name} - это неокрашенный патч (гематоксилин-эозин) БЕЗ разметки.

ТВОЯ ЗАДАЧА:
1. Внимательно изучи изображение и найди ВСЕ мастоциты, которые ты можешь обнаружить
2. Для каждого найденного мастоцита определи:
   - Координаты центра ядра (x, y) в пикселях
   - Тип: "явный" или "неявный" (на основе признаков, которые ты описал ранее)
   - Уверенность: "высокая", "средняя" или "низкая"
   - Краткое описание визуальных признаков, по которым ты его определил

3. ВАЖНО: Постарайся найти не только явно выраженные мастоциты, но и неявно выраженные (расплывчатые, с низким контрастом)

ФОРМАТ ОТВЕТА:
Для каждого найденного мастоцита укажи:
- Мастоцит #N: координаты (x, y), тип: [явный/неявный], уверенность: [высокая/средняя/низкая]
  Признаки: [краткое описание]

Если мастоциты не найдены, укажи это явно и объясни почему.

После списка найденных мастоцитов добавь раздел:
"ОБЩИЕ НАБЛЮДЕНИЯ:"
- Сколько мастоцитов найдено всего
- Сколько явных и сколько неявных
- Какие признаки были наиболее полезны для поиска
- Что затрудняло поиск (если были трудности)
"""
            
            logger.info("📤 ЭТАП 1: Отправка запроса на определение координат...")
            result_step1 = await gemini.analyze_images(
                image_paths=[img_no],
                prompt=prompt_step1,
                system_prompt=system_prompt,
                preserve_resolution=True,  # Сохраняем разрешение для точных координат
            )
            
            # ЭТАП 2: Анализ ошибок с использованием _yes изображения
            prompt_step2 = f"""{previous_summary}

ЗАДАЧА ЭТАПА 2: АНАЛИЗ ОШИБОК

Тебе предоставлены ДВА изображения:
1. {img_no.name} - неокрашенный патч БЕЗ разметки (тот же, что на этапе 1)
2. {img_yes.name} - тот же патч, но С РАЗМЕТКОЙ мастоцитов

На изображении {img_yes.name}:
- Зеленым выделены мастоциты с ЯВНЫМИ признаками
- Синим выделены мастоциты с НЕ ОЧЕНЬ ЯВНЫМИ признаками

ТВОЯ ЗАДАЧА:
1. Сравни свои результаты с этапа 1 с реальной разметкой на {img_yes.name}

2. Определи:
   - СКОЛЬКО мастоцитов ты НАШЕЛ на этапе 1
   - СКОЛЬКО мастоцитов РЕАЛЬНО есть на разметке (зеленые + синие)
   - СКОЛЬКО мастоцитов ты ПРОПУСТИЛ (False Negatives)
   - СКОЛЬКО объектов ты НЕПРАВИЛЬНО определил как мастоциты (False Positives, если были)

3. Для каждого ПРОПУЩЕННОГО мастоцита (который есть в разметке, но ты не нашел):
   - Укажи его координаты из разметки
   - Объясни, ПОЧЕМУ ты его пропустил (слишком слабые признаки, слился с фоном, похож на другую клетку и т.д.)
   - Опиши, какие признаки были бы нужны, чтобы его найти

4. Для каждого НЕПРАВИЛЬНО ОПРЕДЕЛЕННОГО объекта (если были):
   - Укажи его координаты
   - Объясни, на что он похож на самом деле (фибробласт, плазмоцит, лимфоцит и т.д.)
   - Опиши, как можно было бы избежать ошибки

5. ОБЩИЙ АНАЛИЗ:
   - Какие типы мастоцитов ты находил лучше (явные или неявные)?
   - Какие визуальные признаки были наиболее надежными?
   - Какие признаки оказались недостаточными для неявных мастоцитов?

ФОРМАТ ОТВЕТА:
Структурированный анализ с конкретными числами и объяснениями для каждого случая.
"""
            
            logger.info("📤 ЭТАП 2: Отправка запроса на анализ ошибок...")
            result_step2 = await gemini.analyze_images(
                image_paths=[img_no, img_yes],
                prompt=prompt_step2,
                system_prompt=system_prompt,
                preserve_resolution=True,
            )
            
            # ЭТАП 3: Вопросы о том, что нужно для улучшения
            prompt_step3 = f"""{previous_summary}

ЗАДАЧА ЭТАПА 3: РЕФЛЕКСИЯ И РЕКОМЕНДАЦИИ

На основе результатов этапов 1 и 2, ответь на следующие вопросы:

1. СМОГ ЛИ ТЫ НАЙТИ НЕЯВНО ВЫРАЖЕННЫЕ МАСТОЦИТЫ?
   - Да/Нет/Частично
   - Если частично или нет - объясни, какие именно неявные мастоциты ты пропустил и почему

2. ЧТО ТЕБЕ НУЖНО ДЛЯ УЛУЧШЕНИЯ ОБНАРУЖЕНИЯ НЕЯВНЫХ МАСТОЦИТОВ?
   
   а) БОЛЬШЕ АННОТИРОВАННЫХ КАРТИНОК?
   - Помогло бы тебе больше примеров с разметкой?
   - Сколько примерно примеров нужно для обучения?
   - Какие типы примеров были бы наиболее полезны (больше неявных случаев, разные стадии заболевания и т.д.)?
   
   б) ДРУГОЕ РАЗРЕШЕНИЕ?
   - Достаточно ли текущего разрешения изображения для поиска неявных мастоцитов?
   - Нужно ли большее разрешение (больше деталей)?
   - Или наоборот, меньшее разрешение помогает видеть общую картину?
   - В прошлый раз изображения сильно сжимали - повлияло ли это на твою способность находить неявные мастоциты?
   
   в) ДРУГИЕ УЛУЧШЕНИЯ?
   - Нужны ли дополнительные типы окрашивания для сравнения?
   - Помогли бы тебе описания морфологических признаков от экспертов?
   - Нужны ли контекстные подсказки (например, где искать мастоциты в ткани)?
   - Что еще могло бы помочь?

3. ПОЯВИЛИСЬ ЛИ У ТЕБЯ ИНСАЙТЫ?
   - Какие новые закономерности ты заметил при поиске неявных мастоцитов?
   - Есть ли какие-то "скрытые" признаки, которые не очевидны, но помогают?
   - Можешь ли ты сформулировать алгоритм или правила для поиска неявных мастоцитов?
   - Что отличает успешное обнаружение от пропуска?

4. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:
   - Что нужно изменить в процессе разметки/обучения, чтобы улучшить обнаружение неявных мастоцитов?
   - Какие метаданные или аннотации были бы полезны?
   - Какой формат данных был бы оптимальным для обучения?

ФОРМАТ ОТВЕТА:
Детальные ответы на каждый вопрос с конкретными рекомендациями и обоснованиями.
"""
            
            logger.info("📤 ЭТАП 3: Отправка запроса на рефлексию и рекомендации...")
            result_step3 = await gemini.analyze_images(
                image_paths=[img_no, img_yes],
                prompt=prompt_step3,
                system_prompt=system_prompt,
                preserve_resolution=True,
            )
            
            # Сохраняем результаты для этой пары
            results.append({
                "image_no": img_no.name,
                "image_yes": img_yes.name,
                "step1_coordinates": result_step1,
                "step2_error_analysis": result_step2,
                "step3_recommendations": result_step3,
            })
            
            logger.info(f"✅ Обработка пары {img_no.name} завершена")
        
        # Формируем итоговый отчет
        full_result = []
        full_result.append("="*80)
        full_result.append("ИТОГОВЫЙ ОТЧЕТ: АНАЛИЗ КООРДИНАТ И ОШИБОК МАСТОЦИТОВ")
        full_result.append("="*80)
        full_result.append(f"\nПроанализировано пар изображений: {len(results)}")
        full_result.append(f"Изображений _no (без разметки): {len(images_no)}")
        full_result.append(f"Изображений _yes (с разметкой): {len(images_yes)}\n")
        
        # Добавляем результаты по каждой паре
        for idx, result in enumerate(results, 1):
            full_result.append("\n" + "="*80)
            full_result.append(f"ПАРА #{idx}: {result['image_no']} / {result['image_yes']}")
            full_result.append("="*80)
            
            full_result.append("\n" + "-"*80)
            full_result.append("ЭТАП 1: ОПРЕДЕЛЕНИЕ КООРДИНАТ МАСТОЦИТОВ")
            full_result.append("-"*80)
            full_result.append(result['step1_coordinates'])
            
            full_result.append("\n" + "-"*80)
            full_result.append("ЭТАП 2: АНАЛИЗ ОШИБОК")
            full_result.append("-"*80)
            full_result.append(result['step2_error_analysis'])
            
            full_result.append("\n" + "-"*80)
            full_result.append("ЭТАП 3: РЕФЛЕКСИЯ И РЕКОМЕНДАЦИИ")
            full_result.append("-"*80)
            full_result.append(result['step3_recommendations'])
        
        # Добавляем обобщающий раздел
        full_result.append("\n" + "="*80)
        full_result.append("ОБОБЩЕНИЕ ПО ВСЕМ ПАРАМ")
        full_result.append("="*80)
        full_result.append("\n(См. детальные результаты по каждой паре выше)")
        
        result_text = "\n".join(full_result)
        
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТ АНАЛИЗА КООРДИНАТ И ОШИБОК:")
        print("="*80)
        print(result_text)
        print("="*80 + "\n")
        
        # Сохраняем результат в файл
        output_file = Path(__file__).parent / "mast_cells_coordinates_analysis_result.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_text)
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"✅ Результат сохранен в: {output_file}")
        
        # Сохраняем также в JSON для структурированного доступа
        json_output_file = Path(__file__).parent / "mast_cells_coordinates_analysis_result.json"
        with open(json_output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ JSON результат сохранен в: {json_output_file}")
        
    finally:
        await gemini.close()


if __name__ == "__main__":
    asyncio.run(analyze_coordinates_and_errors())

