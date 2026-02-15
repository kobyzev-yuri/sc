#!/usr/bin/env python3
"""
Анализ мастоцитов с использованием Knowledge Base (RAG) для улучшения результатов Gemini.

Использует базу знаний для предоставления контекста и примеров Gemini.
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from train_knowledge_base import MastCellsKnowledgeBase
from analyze_mast_cells_coordinates_gemini import GeminiVisionService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedMastCellsAnalyzer:
    """Анализатор мастоцитов с использованием Knowledge Base"""
    
    def __init__(self, kb_path: str = "./mast_cells_kb"):
        """
        Инициализирует анализатор с Knowledge Base
        
        Args:
            kb_path: Путь к базе знаний
        """
        self.gemini = GeminiVisionService()
        self.kb = MastCellsKnowledgeBase(db_path=kb_path)
        logger.info("Enhanced analyzer initialized with Knowledge Base")
    
    async def analyze_with_context(
        self,
        image_path: Path,
        query: str = "Find all mast cells in this image, including implicit ones",
        n_examples: int = 5,
        filter_difficulty: Optional[str] = None,
        filter_cell_type: Optional[str] = None
    ) -> str:
        """
        Анализирует изображение с использованием Knowledge Base для контекста
        
        Args:
            image_path: Путь к изображению для анализа
            query: Запрос для анализа
            n_examples: Количество примеров из KB для контекста
            filter_difficulty: Фильтр по сложности ("easy", "medium", "hard")
            filter_cell_type: Фильтр по типу ("explicit", "implicit")
        
        Returns:
            Текст ответа от Gemini
        """
        # 1. Ищем похожие примеры в базе знаний
        logger.info(f"Searching knowledge base for similar examples...")
        similar_examples = self.kb.search_similar(
            query_text=query,
            n_results=n_examples,
            filter_difficulty=filter_difficulty,
            filter_cell_type=filter_cell_type
        )
        
        logger.info(f"Found {len(similar_examples)} similar examples in KB")
        
        # 2. Формируем контекстный промпт
        context_parts = [
            "Based on previous successful analyses and a knowledge base of mast cell detection cases, here are key patterns and examples:",
            "",
            "=== KEY MORPHOLOGICAL PATTERNS ===",
            "",
            "EXPLICIT MAST CELLS:",
            "- Central, round/ovoid nucleus (dark purple, hyperchromatic)",
            "- Eosinophilic (pink) granular cytoplasm forming a halo around nucleus",
            "- 'Fried egg' pattern: dark 'yolk' (nucleus) + pink 'white' (cytoplasm)",
            "- Clear boundaries, good contrast with background",
            "- Located in stroma (connective tissue) between crypts",
            "",
            "IMPLICIT MAST CELLS:",
            "- Central nucleus (key differentiator from plasmocytes)",
            "- Pale pink cytoplasm with blurred boundaries",
            "- 'Dirty halo' effect: creates space with muddy pink substance",
            "- Ovoid, 'plump' nucleus (not perfectly round like lymphocytes, not elongated like fibroblasts)",
            "- Low contrast, blends with background",
            "- Often found near explicit mast cells (neighborhood effect)",
            "",
            "=== DISTINGUISHING FEATURES ===",
            "",
            "vs LYMPHOCYTES:",
            "- Mast cells have visible cytoplasm (lymphocytes have almost none)",
            "- Mast cell nucleus is ovoid/plump (lymphocyte nucleus is perfectly round)",
            "",
            "vs PLASMOCYTES:",
            "- Mast cells have CENTRAL nucleus (plasmocytes have ECCENTRIC nucleus)",
            "- Mast cell cytoplasm is granular (plasmocyte has 'cartwheel' chromatin pattern)",
            "",
            "vs FIBROBLASTS:",
            "- Mast cells have round/ovoid nucleus (fibroblasts have elongated, spindle-shaped nucleus)",
            "- Mast cells have no long 'tails' or processes",
            "",
            "=== SUCCESSFUL DETECTION EXAMPLES ===",
            ""
        ]
        
        for i, example in enumerate(similar_examples, 1):
            context_parts.append(f"Example {i} (similarity: {1 - example['distance']:.2%}):")
            context_parts.append(f"  {example['text']}")
            if example['metadata'].get('cell_type'):
                context_parts.append(f"  Type: {example['metadata']['cell_type']}")
            if example['metadata'].get('confidence'):
                context_parts.append(f"  Confidence: {example['metadata']['confidence']}")
            context_parts.append("")
        
        context_parts.append("=== ANALYSIS TASK ===")
        context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # 3. Создаем расширенный промпт
        enhanced_prompt = f"""{context}

{query}

INSTRUCTIONS:
1. Use the patterns and examples above to guide your analysis
2. Pay special attention to:
   - Central nucleus position (key differentiator from plasmocytes)
   - Presence of pink/rosy cytoplasm halo (differentiator from lymphocytes)
   - Round/ovoid shape (differentiator from fibroblasts)
   - 'Dirty halo' effect around cell (indicator of implicit mast cells)
   - Neighborhood effect: if you find one explicit mast cell, look for implicit ones nearby (within 50-100 microns)

3. For each found mast cell, provide:
   - Coordinates (x, y) of nucleus center
   - Type: "explicit" or "implicit"
   - Confidence: "high", "medium", or "low"
   - Brief description of morphological features

4. Be especially careful with implicit mast cells - they may have:
   - Very pale cytoplasm that's barely visible
   - Blurred boundaries that merge with stroma
   - Low contrast with background
   - But still have central, ovoid nucleus

Format your response as a structured list with coordinates and descriptions.
"""
        
        # 4. Отправляем запрос к Gemini
        logger.info("Sending request to Gemini with enhanced context...")
        result = await self.gemini.analyze_images(
            image_paths=[image_path],
            prompt=enhanced_prompt,
            system_prompt="""You are an expert pathologist with access to a knowledge base of successful mast cell detection cases.
Use the provided context and examples to improve your detection accuracy, especially for implicit mast cells.
Be thorough and systematic in your analysis.""",
            preserve_resolution=True
        )
        
        return result
    
    async def analyze_batch_with_kb(
        self,
        image_paths: List[Path],
        query: str = "Find all mast cells in this image",
        n_examples: int = 5
    ) -> List[dict]:
        """
        Анализирует несколько изображений с использованием KB
        
        Args:
            image_paths: Список путей к изображениям
            query: Запрос для анализа
            n_examples: Количество примеров из KB
        
        Returns:
            Список результатов анализа
        """
        results = []
        
        for img_path in image_paths:
            logger.info(f"Analyzing {img_path.name}...")
            
            try:
                result_text = await self.analyze_with_context(
                    image_path=img_path,
                    query=query,
                    n_examples=n_examples
                )
                
                results.append({
                    "image": img_path.name,
                    "status": "success",
                    "result": result_text
                })
                
                logger.info(f"✅ Completed analysis of {img_path.name}")
            
            except Exception as e:
                logger.error(f"❌ Error analyzing {img_path.name}: {e}")
                results.append({
                    "image": img_path.name,
                    "status": "error",
                    "result": f"Error: {str(e)}"
                })
        
        return results
    
    async def close(self):
        """Закрывает соединения"""
        await self.gemini.close()


async def main():
    """Пример использования"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze mast cells with Knowledge Base")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to analyze"
    )
    parser.add_argument(
        "--kb_path",
        default="./mast_cells_kb",
        help="Path to knowledge base directory"
    )
    parser.add_argument(
        "--query",
        default="Find all mast cells in this image, including implicit ones",
        help="Analysis query"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=5,
        help="Number of examples from KB to use"
    )
    parser.add_argument(
        "--filter_difficulty",
        choices=["easy", "medium", "hard"],
        help="Filter examples by difficulty"
    )
    parser.add_argument(
        "--filter_cell_type",
        choices=["explicit", "implicit"],
        help="Filter examples by cell type"
    )
    
    args = parser.parse_args()
    
    analyzer = EnhancedMastCellsAnalyzer(kb_path=args.kb_path)
    
    try:
        if args.image:
            # Анализ одного изображения
            image_path = Path(args.image)
            if not image_path.exists():
                logger.error(f"Image not found: {args.image}")
                return
            
            result = await analyzer.analyze_with_context(
                image_path=image_path,
                query=args.query,
                n_examples=args.n_examples,
                filter_difficulty=args.filter_difficulty,
                filter_cell_type=args.filter_cell_type
            )
            
            print("\n" + "="*80)
            print("ANALYSIS RESULT:")
            print("="*80)
            print(result)
            print("="*80)
        
        else:
            # Анализ всех _no изображений
            mast_dir = Path("MAST_GEMINI")
            images_no = sorted(mast_dir.glob("*_no.png"))
            
            if not images_no:
                logger.error("No _no.png images found in MAST_GEMINI directory")
                return
            
            logger.info(f"Found {len(images_no)} images to analyze")
            
            results = await analyzer.analyze_batch_with_kb(
                image_paths=images_no,
                query=args.query,
                n_examples=args.n_examples
            )
            
            # Сохраняем результаты
            output_file = Path("mast_cells_analysis_with_kb_result.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write("ANALYSIS RESULTS WITH KNOWLEDGE BASE\n")
                f.write("="*80 + "\n\n")
                
                for result in results:
                    f.write(f"Image: {result['image']}\n")
                    f.write(f"Status: {result['status']}\n")
                    f.write("-"*80 + "\n")
                    f.write(result['result'])
                    f.write("\n" + "="*80 + "\n\n")
            
            logger.info(f"Results saved to {output_file}")
    
    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(main())

