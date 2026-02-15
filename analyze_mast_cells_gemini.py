#!/usr/bin/env python3
"""
Скрипт для анализа изображений мастоцитов через Gemini 3 Pro.

Задача: определить, видит ли модель признаки для разметки мастоцитов
на неокрашенных патчах (стандартное окрашивание гематоксилин-эозин).
Может ли модель сама их найти или дать внятное описание для разметчика.
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
    
    def _encode_image(self, image_path: Path, max_size_mb: float = 4.0) -> Dict[str, str]:
        """
        Кодирует изображение в base64 для Gemini API.
        Уменьшает размер изображения, если оно слишком большое.
        
        Args:
            image_path: Путь к изображению
            max_size_mb: Максимальный размер в МБ после кодирования (по умолчанию 4 МБ)
        
        Returns:
            Словарь с inline_data для Gemini API
        """
        # Определяем MIME тип по расширению
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        
        # Читаем и обрабатываем изображение
        with Image.open(image_path) as img:
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
            max_dimension = 2048  # Максимальный размер по большей стороне
            
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
    ) -> str:
        """
        Анализирует изображения через Gemini Vision API.
        
        Args:
            image_paths: Список путей к изображениям
            prompt: Промпт для анализа
            system_prompt: Системный промпт (опционально)
            image_labels: Метки для изображений (для идентификации в промпте)
        
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
            parts.append(self._encode_image(img_path))
        
        request_data: Dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 4096,
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
    
    async def analyze_batch(
        self,
        image_pairs: List[tuple],
        prompt_template: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Анализирует пары изображений в batch режиме (параллельно).
        
        Args:
            image_pairs: Список кортежей (img_g_e_path, img_ihc_path, pair_id)
            prompt_template: Шаблон промпта с плейсхолдерами {pair_id}, {img_g_e_name}, {img_ihc_name}
            system_prompt: Системный промпт (опционально)
        
        Returns:
            Список результатов анализа для каждой пары
        """
        async def analyze_single_pair(img_g_e, img_ihc, pair_id):
            """Анализирует одну пару изображений"""
            try:
                # Формируем промпт для конкретной пары
                prompt = prompt_template.format(
                    pair_id=pair_id,
                    img_g_e_name=img_g_e.name,
                    img_ihc_name=img_ihc.name if img_ihc and img_ihc.exists() else "отсутствует"
                )
                
                # Создаем список изображений для анализа
                image_list = [img_g_e]
                if img_ihc and img_ihc.exists():
                    image_list.append(img_ihc)
                
                # Выполняем анализ
                result_text = await self.analyze_images(
                    image_paths=image_list,
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                
                # Определяем статус на основе результата
                if result_text.startswith("ERROR_402"):
                    status = "error_402"
                elif result_text.startswith("Ошибка"):
                    status = "error"
                else:
                    status = "success"
                
                return {
                    "pair_id": pair_id,
                    "status": status,
                    "result": result_text
                }
            except Exception as e:
                logger.error(f"❌ Пара {pair_id}: ошибка - {e}")
                return {
                    "pair_id": pair_id,
                    "status": "error",
                    "result": f"Ошибка: {str(e)}"
                }
        
        # Создаем задачи для всех пар
        tasks = [
            analyze_single_pair(img_g_e, img_ihc, pair_id)
            for img_g_e, img_ihc, pair_id in image_pairs
        ]
        
        # Выполняем все запросы параллельно
        logger.info(f"📦 Отправка batch запросов для {len(tasks)} пар изображений (параллельно)...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обрабатываем исключения
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pair_id = image_pairs[i][2]
                logger.error(f"❌ Пара {pair_id}: исключение - {result}")
                final_results.append({
                    "pair_id": pair_id,
                    "status": "error",
                    "result": f"Исключение: {str(result)}"
                })
            else:
                final_results.append(result)
                if result["status"] == "success":
                    logger.info(f"✅ Пара {result['pair_id']}: анализ завершен")
        
        return final_results
    
    async def close(self):
        """Закрывает HTTP клиент"""
        await self._client.aclose()


async def analyze_mast_cells():
    """Основная функция анализа мастоцитов"""
    
    # Путь к директории с изображениями
    mast_dir = Path(__file__).parent / "MAST_GEMINI"
    
    if not mast_dir.exists():
        logger.error(f"❌ Директория не найдена: {mast_dir}")
        return
    
    # Находим пары изображений
    images_g_e = sorted(mast_dir.glob("[0-9].png"))  # Стандартное окрашивание г/э
    images_ihc = sorted(mast_dir.glob("[0-9]_игх.png"))  # Окрашивание ИГХ
    
    logger.info(f"Найдено изображений г/э: {len(images_g_e)}")
    logger.info(f"Найдено изображений ИГХ: {len(images_ihc)}")
    
    if not images_g_e:
        logger.error("❌ Не найдено изображений со стандартным окрашиванием")
        return
    
    # Инициализируем клиент Gemini
    try:
        gemini = GeminiVisionService()
    except ValueError as e:
        logger.error(f"❌ {e}")
        return
    
    # Системный промпт
    system_prompt = """Ты эксперт-патолог, специализирующийся на анализе гистологических изображений.
Твоя задача - помочь разметчику найти признаки мастоцитов на неокрашенных патчах.
Будь точным и детальным в описании визуальных признаков.
Отвечай структурированно и конкретно."""
    
    # Шаблон промпта для batch-анализа (для каждой пары отдельно)
    prompt_template = """Проанализируй пару изображений мастоцитов (пара #{pair_id}).

ИЗОБРАЖЕНИЯ:
1. {img_g_e_name} - стандартное окрашивание гематоксилин-эозин (г/э) - это неокрашенный патч
2. {img_ihc_name} - окрашивание ИГХ (иммуногистохимия) - эталонное изображение тех же мастоцитов

АННОТАЦИИ на изображении г/э:
- Зеленым выделены мастоциты с ЯВНЫМИ признаками
- Синим выделены мастоциты с НЕ ОЧЕНЬ ЯВНЫМИ признаками

ЗАДАЧА (отвечай кратко, но конкретно для этой пары):

1. ВИДИШЬ ЛИ ТЫ признаки мастоцитов на неокрашенном патче (г/э)?
   - Если да: перечисли КОНКРЕТНЫЕ визуальные признаки (форма, размер, цвет, текстура, расположение)
   - Если нет: объясни почему (слишком слабо видно, нет контраста, и т.д.)

2. МОЖЕШЬ ЛИ ТЫ САМА найти и разметить мастоциты на этом патче?
   - Если да: опиши краткий алгоритм (на что смотреть, какие признаки использовать)
   - Если нет: что нужно дополнительно (увеличение, другой тип окрашивания, и т.д.)

3. МОЖЕШЬ ЛИ ДАТЬ ВНЯТНОЕ ОПИСАНИЕ для разметчика?
   - Опиши визуальные характеристики мастоцитов на г/э для этой пары
   - Какие морфологические признаки наиболее важны?
   - В чем разница между зелеными (явные) и синими (не очень явные) аннотациями?

4. СРАВНЕНИЕ с ИГХ:
   - Насколько хорошо видны мастоциты на г/э по сравнению с ИГХ для этой пары?
   - Какие признаки сохраняются, какие теряются?

Формат ответа: структурированный список с конкретными наблюдениями для этой пары изображений."""
    
    try:
        # Подготавливаем пары изображений для batch-анализа
        image_pairs = []
        for i, img_g_e in enumerate(images_g_e):
            pair_id = i + 1
            img_ihc = images_ihc[i] if i < len(images_ihc) else None
            image_pairs.append((img_g_e, img_ihc, pair_id))
        
        logger.info("\n" + "="*80)
        logger.info("BATCH АНАЛИЗ ИЗОБРАЖЕНИЙ МАСТОЦИТОВ")
        logger.info("="*80 + "\n")
        logger.info(f"Подготовлено {len(image_pairs)} пар изображений для batch-анализа")
        
        # Выполняем batch-анализ (параллельно)
        batch_results = await gemini.analyze_batch(
            image_pairs=image_pairs,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
        )
        
        # Формируем итоговый отчет
        full_result = []
        full_result.append("="*80)
        full_result.append("ИТОГОВЫЙ ОТЧЕТ: АНАЛИЗ МАСТОЦИТОВ ЧЕРЕЗ GEMINI 3 PRO (BATCH РЕЖИМ)")
        full_result.append("="*80)
        full_result.append(f"\nПроанализировано пар изображений: {len(batch_results)}")
        full_result.append(f"Изображений г/э: {len(images_g_e)}")
        full_result.append(f"Изображений ИГХ: {len(images_ihc)}\n")
        
        # Добавляем результаты по каждой паре
        for result in batch_results:
            pair_id = result["pair_id"]
            status = result["status"]
            result_text = result["result"]
            
            full_result.append("\n" + "="*80)
            full_result.append(f"ПАРА #{pair_id} - {images_g_e[pair_id-1].name}")
            full_result.append("="*80)
            full_result.append(f"Статус: {status}")
            full_result.append("-"*80)
            full_result.append(result_text)
        
        # Добавляем обобщающий раздел
        full_result.append("\n" + "="*80)
        full_result.append("ОБОБЩЕНИЕ ПО ВСЕМ ПАРАМ")
        full_result.append("="*80)
        full_result.append("\n(Обобщение будет добавлено после анализа всех пар)")
        
        result = "\n".join(full_result)
        
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТ BATCH АНАЛИЗА GEMINI 3 PRO:")
        print("="*80)
        print(result)
        print("="*80 + "\n")
        
        # Сохраняем результат в файл
        output_file = Path(__file__).parent / "mast_cells_analysis_result.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"✅ Результат сохранен в: {output_file}")
        
        # Статистика
        successful = sum(1 for r in batch_results if r["status"] == "success")
        error_402 = sum(1 for r in batch_results if r["status"] == "error_402")
        other_errors = sum(1 for r in batch_results if r["status"] == "error")
        total_errors = error_402 + other_errors
        
        logger.info(f"📊 Статистика: успешно {successful}/{len(batch_results)}, ошибок {total_errors} (402: {error_402}, другие: {other_errors})")
        
        if error_402 > 0:
            logger.warning("⚠️ Обнаружены ошибки 402 (недостаточно баланса)")
            logger.warning("💡 Возможные причины:")
            logger.warning("   - Изображения слишком большие (попробуйте уменьшить max_size_mb)")
            logger.warning("   - Модель gemini-3-pro-preview очень дорогая для vision запросов")
            logger.warning("   - Проверьте баланс на ProxyAPI.ru")
        
    finally:
        await gemini.close()


if __name__ == "__main__":
    asyncio.run(analyze_mast_cells())

