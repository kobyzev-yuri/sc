#!/usr/bin/env python3
"""
Создание и управление Knowledge Base для обнаружения мастоцитов через RAG.

Использует ChromaDB для хранения примеров и эмбеддингов.
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MastCellsKnowledgeBase:
    """База знаний для обнаружения мастоцитов с использованием RAG"""
    
    def __init__(self, db_path: str = "./mast_cells_kb"):
        """
        Инициализирует базу знаний
        
        Args:
            db_path: Путь к директории для хранения базы данных
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="mast_cells",
            metadata={"description": "Knowledge base for mast cell detection"}
        )
        # Модель для эмбеддингов текста
        logger.info("Loading sentence transformer model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Knowledge base initialized")
    
    def add_example(
        self,
        example_id: str,
        image_path: str,
        ihc_image_path: Optional[str] = None,
        morphological_features: Optional[Dict] = None,
        gemini_insights: Optional[List[str]] = None,
        difficulty: str = "medium",
        coordinates: Optional[Dict] = None,
        cell_type: Optional[str] = None,  # "explicit" or "implicit"
        confidence: Optional[str] = None
    ):
        """
        Добавляет пример в базу знаний
        
        Args:
            example_id: Уникальный идентификатор примера
            image_path: Путь к изображению Г/Э
            ihc_image_path: Путь к парному ИГХ изображению (опционально)
            morphological_features: Словарь с морфологическими признаками
            gemini_insights: Список инсайтов от Gemini
            difficulty: Уровень сложности ("easy", "medium", "hard")
            coordinates: Координаты мастоцита {"x": int, "y": int}
            cell_type: Тип мастоцита ("explicit" or "implicit")
            confidence: Уровень уверенности ("high", "medium", "low")
        """
        # Формируем текстовое описание
        text_parts = []
        
        if cell_type:
            text_parts.append(f"Cell type: {cell_type} mast cell")
        
        if morphological_features:
            if "nucleus" in morphological_features:
                text_parts.append(f"Nucleus: {morphological_features['nucleus']}")
            if "cytoplasm" in morphological_features:
                text_parts.append(f"Cytoplasm: {morphological_features['cytoplasm']}")
            if "shape" in morphological_features:
                text_parts.append(f"Shape: {morphological_features['shape']}")
            if "location" in morphological_features:
                text_parts.append(f"Location: {morphological_features['location']}")
        
        if gemini_insights:
            text_parts.append("Key insights: " + "; ".join(gemini_insights))
        
        if confidence:
            text_parts.append(f"Confidence: {confidence}")
        
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
            "cell_type": cell_type or "",
            "confidence": confidence or "",
            "text": full_text
        }
        
        if coordinates:
            metadata["coordinates_x"] = str(coordinates.get("x", ""))
            metadata["coordinates_y"] = str(coordinates.get("y", ""))
        
        # Добавляем в коллекцию
        self.collection.add(
            ids=[example_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[full_text]
        )
        
        logger.info(f"Added example {example_id} to knowledge base")
    
    def search_similar(
        self,
        query_text: str,
        n_results: int = 5,
        filter_difficulty: Optional[str] = None,
        filter_cell_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Ищет похожие примеры в базе знаний
        
        Args:
            query_text: Текст запроса
            n_results: Количество результатов
            filter_difficulty: Фильтр по сложности ("easy", "medium", "hard")
            filter_cell_type: Фильтр по типу ("explicit", "implicit")
        
        Returns:
            Список словарей с результатами поиска
        """
        query_embedding = self.embedder.encode(query_text).tolist()
        
        where_clause = {}
        if filter_difficulty:
            where_clause["difficulty"] = filter_difficulty
        if filter_cell_type:
            where_clause["cell_type"] = filter_cell_type
        
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
    
    def get_all_examples(self) -> List[Dict]:
        """Возвращает все примеры из базы знаний"""
        results = self.collection.get()
        
        return [
            {
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            }
            for i in range(len(results["ids"]))
        ]
    
    def delete_example(self, example_id: str):
        """Удаляет пример из базы знаний"""
        self.collection.delete(ids=[example_id])
        logger.info(f"Deleted example {example_id}")


def populate_from_gemini_analysis(
    kb_path: str = "./mast_cells_kb",
    analysis_file: str = "mast_cells_coordinates_analysis_result.json",
    mast_dir: str = "MAST_GEMINI"
):
    """
    Пополняет базу знаний на основе результатов анализа Gemini
    
    Args:
        kb_path: Путь к базе знаний
        analysis_file: Путь к файлу с результатами анализа
        mast_dir: Директория с изображениями мастоцитов
    """
    kb = MastCellsKnowledgeBase(db_path=kb_path)
    
    # Читаем результаты анализа
    analysis_path = Path(analysis_file)
    if not analysis_path.exists():
        logger.error(f"Analysis file not found: {analysis_file}")
        return
    
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis_results = json.load(f)
    
    logger.info(f"Processing {len(analysis_results)} analysis results...")
    
    for idx, result in enumerate(analysis_results, 1):
        image_no = result["image_no"]
        image_yes = result["image_yes"]
        
        # Извлекаем инсайты из этапа 3
        step3_text = result.get("step3_recommendations", "")
        
        # Парсим координаты из этапа 1
        step1_text = result.get("step1_coordinates", "")
        
        # Извлекаем ключевые инсайты (упрощенная версия)
        gemini_insights = [
            "Rule of 'Dirty Halo': creates space with muddy pink substance",
            "Nuclear criterion: ovoid, 'plump' nucleus",
            "Law of neighborhood: rarely solitary",
            "Effect of halo: space around cell"
        ]
        
        # Добавляем в базу знаний
        kb.add_example(
            example_id=f"pair_{image_no.replace('.png', '')}",
            image_path=f"{mast_dir}/{image_yes}",
            ihc_image_path=None,  # Можно добавить если есть парные ИГХ
            morphological_features={
                "nucleus": "central, round/ovoid, hyperchromatic",
                "cytoplasm": "eosinophilic, pink, granular",
                "shape": "round or ovoid, 'fried egg' pattern",
                "location": "stroma, between crypts"
            },
            gemini_insights=gemini_insights,
            difficulty="medium",
            cell_type="explicit"  # Можно определить из анализа
        )
    
    logger.info("Knowledge base populated successfully!")
    
    # Показываем статистику
    all_examples = kb.get_all_examples()
    logger.info(f"Total examples in KB: {len(all_examples)}")


def add_explicit_examples(kb_path: str = "./mast_cells_kb"):
    """Добавляет примеры явных мастоцитов на основе рекомендаций Gemini"""
    
    kb = MastCellsKnowledgeBase(db_path=kb_path)
    
    # Примеры явных мастоцитов
    explicit_examples = [
        {
            "example_id": "explicit_001",
            "image_path": "MAST_GEMINI/01_yes.png",
            "morphological_features": {
                "nucleus": "central, round, hyperchromatic, dark purple",
                "cytoplasm": "eosinophilic, granular, abundant, pink halo",
                "shape": "round, 'fried egg' pattern",
                "location": "stroma, between crypts, isolated"
            },
            "gemini_insights": [
                "Rule of 'Dirty Halo': creates space with muddy pink substance",
                "Classic 'fried egg' pattern: dark yolk (nucleus) + pink white (cytoplasm)",
                "High contrast with background"
            ],
            "difficulty": "easy",
            "cell_type": "explicit",
            "confidence": "high"
        },
        {
            "example_id": "explicit_002",
            "image_path": "MAST_GEMINI/02_yes.png",
            "morphological_features": {
                "nucleus": "central, round, hyperchromatic",
                "cytoplasm": "eosinophilic, granular, well-defined",
                "shape": "round, isolated",
                "location": "stroma, lower part"
            },
            "gemini_insights": [
                "Clear boundaries, good contrast",
                "Abundant granular cytoplasm"
            ],
            "difficulty": "easy",
            "cell_type": "explicit",
            "confidence": "high"
        }
    ]
    
    for example in explicit_examples:
        kb.add_example(**example)
    
    logger.info(f"Added {len(explicit_examples)} explicit examples")


def add_implicit_examples(kb_path: str = "./mast_cells_kb"):
    """Добавляет примеры неявных мастоцитов на основе рекомендаций Gemini"""
    
    kb = MastCellsKnowledgeBase(db_path=kb_path)
    
    # Примеры неявных мастоцитов
    implicit_examples = [
        {
            "example_id": "implicit_001",
            "image_path": "MAST_GEMINI/02_yes.png",
            "morphological_features": {
                "nucleus": "central, ovoid, hyperchromatic",
                "cytoplasm": "pale pink, blurred boundaries, minimal",
                "shape": "ovoid, slightly elongated",
                "location": "stroma, near vessels"
            },
            "gemini_insights": [
                "Rule of 'Dirty Halo': creates space with muddy pink substance",
                "Nuclear criterion: ovoid, 'plump' nucleus",
                "Low contrast, blends with background",
                "Effect of halo: space around cell"
            ],
            "difficulty": "hard",
            "cell_type": "implicit",
            "confidence": "medium"
        },
        {
            "example_id": "implicit_002",
            "image_path": "MAST_GEMINI/03_yes.png",
            "morphological_features": {
                "nucleus": "central, round, hyperchromatic",
                "cytoplasm": "very pale, barely visible, no clear boundaries",
                "shape": "round, but boundaries unclear",
                "location": "stroma, center, near inflammatory infiltrate"
            },
            "gemini_insights": [
                "Mimics lymphocyte/plasmocyte",
                "Absence of visible cytoplasm on H&E",
                "Low contrast, merges with stroma",
                "Nuclear characteristics: lighter chromatin than lymphocytes"
            ],
            "difficulty": "hard",
            "cell_type": "implicit",
            "confidence": "low"
        }
    ]
    
    for example in implicit_examples:
        kb.add_example(**example)
    
    logger.info(f"Added {len(implicit_examples)} implicit examples")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Mast Cells Knowledge Base")
    parser.add_argument(
        "--action",
        choices=["populate", "add_explicit", "add_implicit", "search", "list"],
        default="populate",
        help="Action to perform"
    )
    parser.add_argument(
        "--kb_path",
        default="./mast_cells_kb",
        help="Path to knowledge base directory"
    )
    parser.add_argument(
        "--query",
        help="Search query (for search action)"
    )
    parser.add_argument(
        "--n_results",
        type=int,
        default=5,
        help="Number of search results"
    )
    
    args = parser.parse_args()
    
    if args.action == "populate":
        populate_from_gemini_analysis(kb_path=args.kb_path)
        add_explicit_examples(kb_path=args.kb_path)
        add_implicit_examples(kb_path=args.kb_path)
    
    elif args.action == "add_explicit":
        add_explicit_examples(kb_path=args.kb_path)
    
    elif args.action == "add_implicit":
        add_implicit_examples(kb_path=args.kb_path)
    
    elif args.action == "search":
        if not args.query:
            logger.error("--query is required for search action")
        else:
            kb = MastCellsKnowledgeBase(db_path=args.kb_path)
            results = kb.search_similar(args.query, n_results=args.n_results)
            print("\nSearch results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. ID: {result['id']}")
                print(f"   Distance: {result['distance']:.4f}")
                print(f"   Text: {result['text']}")
                print(f"   Metadata: {result['metadata']}")
    
    elif args.action == "list":
        kb = MastCellsKnowledgeBase(db_path=args.kb_path)
        examples = kb.get_all_examples()
        print(f"\nTotal examples: {len(examples)}")
        for example in examples:
            print(f"\nID: {example['id']}")
            print(f"Text: {example['text']}")
            print(f"Metadata: {example['metadata']}")

