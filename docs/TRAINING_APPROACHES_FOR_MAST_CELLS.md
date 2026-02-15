# Практические подходы к обучению моделей для обнаружения мастоцитов

## Обзор проблемы

На основе анализа Gemini 3 Pro выявлено, что для улучшения обнаружения неявных мастоцитов нужно:
1. Больше аннотированных примеров (50-1000 парных патчей Г/Э + ИГХ)
2. Выше разрешение (lossless форматы, 40x увеличение)
3. Парные данные для обучения на скрытых паттернах

**Ограничение:** Gemini не поддерживает файнтюнинг (fine-tuning) или LoRA.

## Два подхода к решению

### ПОДХОД 1: Knowledge Base / RAG для Gemini

**Преимущества:**
- ✅ Не требует обучения модели
- ✅ Можно использовать сразу
- ✅ Легко обновлять базу знаний
- ✅ Работает с существующим Gemini API

**Недостатки:**
- ❌ Ограничен контекстом промпта
- ❌ Может быть медленнее (нужно искать в базе)
- ❌ Зависит от качества поиска в базе

### ПОДХОД 2: Файнтюнинг другой Vision модели (Qwen-VL, LLaVA, etc.)

**Преимущества:**
- ✅ Полный контроль над моделью
- ✅ Можно использовать LoRA для эффективного обучения
- ✅ Модель "запоминает" паттерны
- ✅ Не зависит от внешних API

**Недостатки:**
- ❌ Требует вычислительных ресурсов для обучения
- ❌ Нужно больше данных для обучения
- ❌ Требует настройки инфраструктуры

---

## ПОДХОД 1: Knowledge Base / RAG для Gemini

### Архитектура решения

```
┌─────────────────┐
│  Изображение    │
│   (Г/Э патч)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gemini Vision  │
│   API (3 Pro)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  RAG System     │◄─────│  Knowledge Base  │
│  (Retrieval)    │      │  (Vector Store)  │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐
│  Enhanced       │
│  Prompt +       │
│  Context        │
└─────────────────┘
```

### Шаг 1: Создание Knowledge Base

#### 1.1. Подготовка данных для базы знаний

Структура данных должна включать:

```python
# Структура записи в Knowledge Base
{
    "id": "mast_cell_explicit_001",
    "type": "explicit",  # или "implicit"
    "image_path": "MAST_GEMINI/01_yes.png",
    "ihc_image_path": "MAST_GEMINI/1_игх.png",  # парное ИГХ изображение
    "coordinates": {"x": 195, "y": 190},
    "confidence": "high",
    "morphological_features": {
        "nucleus": "central, round, hyperchromatic",
        "cytoplasm": "eosinophilic, granular, abundant",
        "shape": "round, 'fried egg' pattern",
        "location": "stroma, between crypts"
    },
    "distinguishing_features": {
        "vs_lymphocytes": "has visible cytoplasm, nucleus not bare",
        "vs_plasmocytes": "nucleus central (not eccentric)",
        "vs_fibroblasts": "round nucleus (not elongated)"
    },
    "gemini_insights": [
        "Rule of 'Dirty Halo': creates space with muddy pink substance",
        "Nuclear criterion: ovoid, 'plump' nucleus",
        "Law of neighborhood: rarely solitary"
    ],
    "difficulty_level": "easy",  # easy, medium, hard
    "annotations": {
        "explicit": True,
        "implicit": False
    }
}
```

#### 1.2. Реализация Vector Store

**Вариант A: Использование ChromaDB (рекомендуется)**

```python
# train_knowledge_base.py
import chromadb
from chromadb.config import Settings
from pathlib import Path
import json
from PIL import Image
import base64
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

class MastCellsKnowledgeBase:
    def __init__(self, db_path: str = "./mast_cells_kb"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="mast_cells",
            metadata={"description": "Knowledge base for mast cell detection"}
        )
        # Модель для эмбеддингов текста
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # Модель для эмбеддингов изображений (опционально)
        # self.image_embedder = SentenceTransformer('clip-ViT-B-32')
    
    def add_example(
        self,
        example_id: str,
        image_path: str,
        ihc_image_path: str = None,
        morphological_features: Dict = None,
        gemini_insights: List[str] = None,
        difficulty: str = "medium"
    ):
        """Добавляет пример в базу знаний"""
        
        # Формируем текстовое описание
        text_parts = []
        
        if morphological_features:
            text_parts.append(f"Nucleus: {morphological_features.get('nucleus', 'N/A')}")
            text_parts.append(f"Cytoplasm: {morphological_features.get('cytoplasm', 'N/A')}")
            text_parts.append(f"Shape: {morphological_features.get('shape', 'N/A')}")
            text_parts.append(f"Location: {morphological_features.get('location', 'N/A')}")
        
        if gemini_insights:
            text_parts.append("Insights: " + "; ".join(gemini_insights))
        
        text_parts.append(f"Difficulty: {difficulty}")
        
        full_text = " | ".join(text_parts)
        
        # Создаем эмбеддинг
        embedding = self.embedder.encode(full_text).tolist()
        
        # Метаданные
        metadata = {
            "example_id": example_id,
            "image_path": image_path,
            "ihc_image_path": ihc_image_path or "",
            "difficulty": difficulty,
            "text": full_text
        }
        
        # Добавляем в коллекцию
        self.collection.add(
            ids=[example_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[full_text]
        )
    
    def search_similar(
        self,
        query_text: str,
        n_results: int = 5,
        filter_difficulty: str = None
    ) -> List[Dict]:
        """Ищет похожие примеры в базе знаний"""
        
        query_embedding = self.embedder.encode(query_text).tolist()
        
        where_clause = {}
        if filter_difficulty:
            where_clause["difficulty"] = filter_difficulty
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]

# Использование
if __name__ == "__main__":
    kb = MastCellsKnowledgeBase()
    
    # Добавляем примеры из анализа Gemini
    kb.add_example(
        example_id="explicit_001",
        image_path="MAST_GEMINI/01_yes.png",
        ihc_image_path="MAST_GEMINI/1_игх.png",
        morphological_features={
            "nucleus": "central, round, hyperchromatic, dark purple",
            "cytoplasm": "eosinophilic, granular, abundant, pink halo",
            "shape": "round, 'fried egg' pattern",
            "location": "stroma, between crypts, isolated"
        },
        gemini_insights=[
            "Rule of 'Dirty Halo': creates space with muddy pink substance",
            "Classic 'fried egg' pattern: dark yolk (nucleus) + pink white (cytoplasm)"
        ],
        difficulty="easy"
    )
    
    # Поиск похожих примеров
    results = kb.search_similar(
        "mast cell with central nucleus and pink cytoplasm",
        n_results=3
    )
    print(results)
```

**Вариант B: Использование Pinecone (облачное решение)**

```python
import pinecone
from sentence_transformers import SentenceTransformer

class PineconeKnowledgeBase:
    def __init__(self, api_key: str, index_name: str = "mast-cells-kb"):
        pinecone.init(api_key=api_key, environment="us-west1-gcp")
        self.index = pinecone.Index(index_name)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_example(self, example_id: str, text: str, metadata: Dict):
        embedding = self.embedder.encode(text).tolist()
        self.index.upsert(
            vectors=[(example_id, embedding, metadata)]
        )
    
    def search(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.encode(query).tolist()
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results
```

#### 1.3. Интеграция с Gemini API

```python
# analyze_with_kb.py
from analyze_mast_cells_coordinates_gemini import GeminiVisionService
from train_knowledge_base import MastCellsKnowledgeBase
from pathlib import Path

class EnhancedMastCellsAnalyzer:
    def __init__(self, kb_path: str = "./mast_cells_kb"):
        self.gemini = GeminiVisionService()
        self.kb = MastCellsKnowledgeBase(db_path=kb_path)
    
    async def analyze_with_context(
        self,
        image_path: Path,
        query: str = "Find mast cells in this image"
    ):
        """Анализирует изображение с использованием Knowledge Base"""
        
        # 1. Ищем похожие примеры в базе знаний
        similar_examples = self.kb.search_similar(
            query_text=query,
            n_results=5
        )
        
        # 2. Формируем контекстный промпт
        context_parts = [
            "Based on previous successful analyses, here are key patterns:",
            ""
        ]
        
        for i, example in enumerate(similar_examples, 1):
            context_parts.append(f"Example {i}:")
            context_parts.append(example["text"])
            context_parts.append("")
        
        context_parts.append("Use these patterns to analyze the provided image.")
        
        context = "\n".join(context_parts)
        
        # 3. Создаем расширенный промпт
        enhanced_prompt = f"""{context}

ANALYSIS TASK:
{query}

Pay special attention to:
- Central nucleus position (key differentiator from plasmocytes)
- Presence of pink/rosy cytoplasm halo (differentiator from lymphocytes)
- Round/ovoid shape (differentiator from fibroblasts)
- 'Dirty halo' effect around cell (indicator of implicit mast cells)
- Neighborhood effect: if you find one explicit mast cell, look for implicit ones nearby

Provide coordinates and confidence levels for each found mast cell.
"""
        
        # 4. Отправляем запрос к Gemini
        result = await self.gemini.analyze_images(
            image_paths=[image_path],
            prompt=enhanced_prompt,
            system_prompt="You are an expert pathologist with access to a knowledge base of successful mast cell detection cases.",
            preserve_resolution=True
        )
        
        return result

# Использование
async def main():
    analyzer = EnhancedMastCellsAnalyzer()
    
    result = await analyzer.analyze_with_context(
        image_path=Path("MAST_GEMINI/01_no.png"),
        query="Find all mast cells, including implicit ones"
    )
    
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Шаг 2: Пополнение базы знаний

```python
# populate_kb_from_analysis.py
"""
Пополняет базу знаний на основе результатов анализа Gemini
"""
import json
from pathlib import Path
from train_knowledge_base import MastCellsKnowledgeBase

def populate_from_gemini_analysis():
    """Извлекает данные из результатов анализа и добавляет в KB"""
    
    kb = MastCellsKnowledgeBase()
    
    # Читаем результаты анализа
    with open("mast_cells_coordinates_analysis_result.json", "r") as f:
        analysis_results = json.load(f)
    
    for result in analysis_results:
        image_no = result["image_no"]
        image_yes = result["image_yes"]
        
        # Извлекаем инсайты из этапа 3
        step3_text = result["step3_recommendations"]
        
        # Парсим координаты из этапа 1
        step1_text = result["step1_coordinates"]
        
        # Добавляем в базу знаний
        kb.add_example(
            example_id=f"pair_{image_no.replace('.png', '')}",
            image_path=f"MAST_GEMINI/{image_yes}",
            ihc_image_path=None,  # Можно добавить если есть
            morphological_features={
                "nucleus": "central, round/ovoid",
                "cytoplasm": "eosinophilic, pink, granular",
                "shape": "round or ovoid",
                "location": "stroma, between crypts"
            },
            gemini_insights=[
                "Rule of 'Dirty Halo'",
                "Nuclear criterion: ovoid, 'plump' nucleus",
                "Law of neighborhood",
                "Effect of halo: space around cell"
            ],
            difficulty="medium"
        )
    
    print("Knowledge base populated successfully!")

if __name__ == "__main__":
    populate_from_gemini_analysis()
```

### Шаг 3: Установка зависимостей

```bash
# requirements_kb.txt
chromadb>=0.4.0
sentence-transformers>=2.2.0
pinecone-client>=2.2.0  # опционально, для облачного решения
```

---

## ПОДХОД 2: Файнтюнинг Vision модели с LoRA

### Выбор модели

**Рекомендуемые модели:**
1. **Qwen-VL** (Alibaba) - хорошая поддержка vision, открытый код
2. **LLaVA** (Microsoft) - популярная, много примеров
3. **InstructBLIP** - хорошая для детальных инструкций
4. **InternVL** - новая, эффективная

**Рекомендация: Qwen-VL** - хороший баланс качества и доступности.

### Шаг 1: Подготовка данных для обучения

#### 1.1. Формат данных

```python
# prepare_training_data.py
from pathlib import Path
import json
from PIL import Image
import base64
from typing import List, Dict

def prepare_qwen_vl_dataset(
    mast_dir: Path = Path("MAST_GEMINI"),
    output_file: str = "mast_cells_training_data.jsonl"
):
    """
    Подготавливает данные в формате для Qwen-VL fine-tuning
    """
    
    training_examples = []
    
    # Читаем результаты анализа
    with open("mast_cells_coordinates_analysis_result.json", "r") as f:
        analysis_results = json.load(f)
    
    for result in analysis_results:
        image_no_path = mast_dir / result["image_no"]
        image_yes_path = mast_dir / result["image_yes"]
        
        # Загружаем изображение
        img = Image.open(image_no_path)
        
        # Конвертируем в base64
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Извлекаем координаты и описания из анализа
        step1_text = result["step1_coordinates"]
        
        # Формируем промпт и ответ в формате Qwen-VL
        conversation = [
            {
                "from": "user",
                "value": [
                    {
                        "type": "image",
                        "image": f"data:image/png;base64,{img_base64}"
                    },
                    {
                        "type": "text",
                        "text": """Analyze this H&E stained histology image and find all mast cells.

Pay attention to:
1. Explicit mast cells: round/ovoid cells with central nucleus and pink granular cytoplasm ("fried egg" pattern)
2. Implicit mast cells: cells with central nucleus but blurred/pale cytoplasm boundaries

For each found mast cell, provide:
- Coordinates (x, y) of nucleus center
- Type: "explicit" or "implicit"
- Confidence: "high", "medium", or "low"
- Brief description of morphological features"""
                    }
                ]
            },
            {
                "from": "assistant",
                "value": step1_text  # Ответ Gemini из анализа
            }
        ]
        
        training_examples.append({
            "id": f"mast_cell_{result['image_no'].replace('.png', '')}",
            "conversations": conversation
        })
    
    # Сохраняем в JSONL формат
    with open(output_file, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Created {len(training_examples)} training examples in {output_file}")

if __name__ == "__main__":
    prepare_qwen_vl_dataset()
```

#### 1.2. Расширение датасета

```python
# augment_training_data.py
"""
Создает дополнительные примеры на основе рекомендаций Gemini
"""
import json
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

def augment_image(image_path: Path, output_dir: Path):
    """Создает аугментированные версии изображения"""
    
    img = Image.open(image_path)
    augmented_images = []
    
    # 1. Яркость
    for factor in [0.9, 1.1]:
        enhancer = ImageEnhance.Brightness(img)
        aug_img = enhancer.enhance(factor)
        aug_path = output_dir / f"{image_path.stem}_bright_{factor}.png"
        aug_img.save(aug_path)
        augmented_images.append(aug_path)
    
    # 2. Контраст
    for factor in [0.9, 1.1]:
        enhancer = ImageEnhance.Contrast(img)
        aug_img = enhancer.enhance(factor)
        aug_path = output_dir / f"{image_path.stem}_contrast_{factor}.png"
        aug_img.save(aug_path)
        augmented_images.append(aug_path)
    
    # 3. Поворот (небольшой)
    for angle in [-5, 5]:
        aug_img = img.rotate(angle, fillcolor='white')
        aug_path = output_dir / f"{image_path.stem}_rotate_{angle}.png"
        aug_img.save(aug_path)
        augmented_images.append(aug_path)
    
    return augmented_images
```

### Шаг 2: Файнтюнинг Qwen-VL с LoRA

#### 2.1. Установка зависимостей

```bash
# requirements_qwen_vl.txt
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0  # для LoRA
accelerate>=0.24.0
bitsandbytes>=0.41.0  # для 8-bit training
qwen-vl  # или установка из исходников
```

#### 2.2. Скрипт обучения

```python
# train_qwen_vl_lora.py
"""
Файнтюнинг Qwen-VL с LoRA для обнаружения мастоцитов
"""
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
from pathlib import Path

def load_mast_cells_dataset(jsonl_file: str):
    """Загружает датасет из JSONL файла"""
    
    def process_example(example):
        # Парсим conversations
        conversations = json.loads(example["text"])["conversations"]
        
        # Извлекаем изображение и текст
        image = None
        user_text = ""
        assistant_text = ""
        
        for msg in conversations:
            if msg["from"] == "user":
                for item in msg["value"]:
                    if item["type"] == "image":
                        # Декодируем base64
                        import base64
                        from io import BytesIO
                        from PIL import Image
                        
                        img_data = item["image"].split(",")[1]
                        img_bytes = base64.b64decode(img_data)
                        image = Image.open(BytesIO(img_bytes))
                    elif item["type"] == "text":
                        user_text = item["text"]
            elif msg["from"] == "assistant":
                assistant_text = msg["value"]
        
        return {
            "image": image,
            "user_text": user_text,
            "assistant_text": assistant_text
        }
    
    dataset = load_dataset("json", data_files=jsonl_file, split="train")
    return dataset.map(process_example)

def train_qwen_vl_lora(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",  # или 7B, зависит от ресурсов
    train_file: str = "mast_cells_training_data.jsonl",
    output_dir: str = "./qwen_vl_mast_cells_lora",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4
):
    """Обучает Qwen-VL с LoRA"""
    
    # 1. Загружаем модель и процессор
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    
    # 2. Подготовка для LoRA
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # для Qwen-VL
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. Загружаем датасет
    print("Loading dataset...")
    dataset = load_mast_cells_dataset(train_file)
    
    # 4. Подготовка данных
    def preprocess_function(examples):
        # Токенизация и обработка изображений
        # (упрощенная версия, нужно адаптировать под Qwen-VL)
        inputs = processor(
            text=examples["user_text"],
            images=examples["image"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        labels = processor.tokenizer(
            examples["assistant_text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )["input_ids"]
        
        inputs["labels"] = labels
        return inputs
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=3,
        remove_unused_columns=False,
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda x: preprocess_function(x),
    )
    
    # 7. Обучение
    print("Starting training...")
    trainer.train()
    
    # 8. Сохранение
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_qwen_vl_lora(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        train_file="mast_cells_training_data.jsonl",
        output_dir="./qwen_vl_mast_cells_lora",
        num_epochs=3
    )
```

### Шаг 3: Использование обученной модели

```python
# use_trained_model.py
"""
Использование обученной Qwen-VL модели для обнаружения мастоцитов
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from PIL import Image
from pathlib import Path

class TrainedMastCellsDetector:
    def __init__(self, base_model: str, lora_path: str):
        # Загружаем базовую модель
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Загружаем LoRA веса
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        
        self.processor = Qwen2VLProcessor.from_pretrained(base_model)
    
    def detect_mast_cells(self, image_path: Path) -> str:
        """Обнаруживает мастоциты на изображении"""
        
        image = Image.open(image_path)
        
        prompt = """Analyze this H&E stained histology image and find all mast cells.

Pay attention to:
1. Explicit mast cells: round/ovoid cells with central nucleus and pink granular cytoplasm
2. Implicit mast cells: cells with central nucleus but blurred cytoplasm

For each found mast cell, provide coordinates, type, and confidence."""
        
        # Обработка
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2
            )
        
        # Декодирование
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response

# Использование
if __name__ == "__main__":
    detector = TrainedMastCellsDetector(
        base_model="Qwen/Qwen2-VL-2B-Instruct",
        lora_path="./qwen_vl_mast_cells_lora"
    )
    
    result = detector.detect_mast_cells(Path("MAST_GEMINI/01_no.png"))
    print(result)
```

---

## Сравнение подходов

| Критерий | Knowledge Base (RAG) | Файнтюнинг с LoRA |
|----------|---------------------|-------------------|
| **Сложность реализации** | Низкая | Средняя-Высокая |
| **Требуемые ресурсы** | Минимальные | GPU для обучения |
| **Скорость внедрения** | Быстро (дни) | Медленно (недели) |
| **Точность** | Зависит от качества KB | Может быть выше |
| **Обновляемость** | Легко обновлять | Нужно переобучать |
| **Стоимость** | Низкая (только API) | Средняя (GPU время) |
| **Масштабируемость** | Хорошая | Ограничена ресурсами |

## Рекомендации

### Для быстрого старта:
1. **Начните с Knowledge Base** - быстрее внедрить, можно использовать сразу
2. Пополните базу знаний результатами анализа Gemini
3. Добавьте парные данные Г/Э + ИГХ по мере поступления

### Для долгосрочного решения:
1. **Параллельно готовьте данные** для файнтюнинга
2. Когда наберется 50-100 примеров, запустите обучение Qwen-VL
3. Используйте оба подхода: KB для быстрых ответов, обученная модель для точности

### Гибридный подход (рекомендуется):
1. Knowledge Base для контекста и быстрых ответов
2. Обученная модель для финальной детекции
3. Объединить результаты обоих подходов для максимальной точности

---

## Следующие шаги

1. ✅ Создать структуру Knowledge Base
2. ✅ Пополнить базу знаний из анализа Gemini
3. ✅ Интегрировать RAG с Gemini API
4. ⏳ Подготовить данные для файнтюнинга
5. ⏳ Настроить инфраструктуру для обучения (GPU)
6. ⏳ Обучить Qwen-VL с LoRA
7. ⏳ Сравнить результаты обоих подходов

