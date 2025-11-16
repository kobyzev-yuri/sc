"""
Система отслеживания экспериментов для поиска лучших моделей и параметров.

Ведет историю всех экспериментов с метриками, параметрами и метаданными.
Позволяет сравнивать эксперименты и находить лучшие результаты.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


class ExperimentTracker:
    """
    Трекер экспериментов для отслеживания лучших моделей и параметров.
    
    Ведет централизованную базу данных всех экспериментов с:
    - Метриками качества (score, separation, mod_norm, explained_variance)
    - Параметрами модели (признаки, настройки)
    - Метаданными (train set, версия агрегации, timestamp)
    - Ссылками на сохраненные модели и результаты
    """
    
    def __init__(self, experiments_dir: Path = Path("experiments")):
        """
        Args:
            experiments_dir: Базовая директория для экспериментов
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Файл с централизованной базой экспериментов
        self.registry_file = self.experiments_dir / "experiments_registry.json"
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Загружает реестр экспериментов из файла"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    self.registry = json.load(f)
            except Exception:
                self.registry = {
                    "experiments": [],
                    "best_experiments": {},
                    "version": "1.0"
                }
        else:
            self.registry = {
                "experiments": [],
                "best_experiments": {},
                "version": "1.0"
            }
    
    def _save_registry(self) -> None:
        """Сохраняет реестр экспериментов в файл"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def register_experiment(
        self,
        experiment_name: str,
        experiment_dir: Path,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Регистрирует новый эксперимент в системе отслеживания.
        
        Args:
            experiment_name: Имя эксперимента
            experiment_dir: Директория эксперимента
            metrics: Словарь с метриками качества
            parameters: Словарь с параметрами модели (признаки, настройки)
            metadata: Дополнительные метаданные (train set, версия агрегации и т.д.)
            
        Returns:
            ID эксперимента в реестре
        """
        experiment_dir = Path(experiment_dir)
        
        # Создаем уникальный ID эксперимента
        exp_id = hashlib.md5(
            f"{experiment_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Подготавливаем запись эксперимента
        experiment_record = {
            "id": exp_id,
            "name": experiment_name,
            "directory": str(experiment_dir.relative_to(self.experiments_dir.parent)),
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "score": float(metrics.get("score", 0)),
                "separation": float(metrics.get("separation", 0)),
                "mean_pc1_norm_mod": float(metrics.get("mean_pc1_norm_mod", 0)),
                "explained_variance": float(metrics.get("explained_variance", 0)),
                "mean_pc1_mod": float(metrics.get("mean_pc1_mod", 0)),
                "mean_pc1_normal": float(metrics.get("mean_pc1_normal", 0)),
            },
            "parameters": {
                "selected_features": parameters.get("selected_features", []),
                "n_features": len(parameters.get("selected_features", [])),
                "method": parameters.get("method", "unknown"),
                "use_relative_features": parameters.get("use_relative_features", True),
                **{k: v for k, v in parameters.items() 
                   if k not in ["selected_features", "method", "use_relative_features"]}
            },
            "metadata": metadata or {},
        }
        
        # Добавляем информацию о данных, если не указана
        if metadata:
            if "train_set" not in experiment_record["metadata"] and "train_set" in metadata:
                experiment_record["metadata"]["train_set"] = metadata["train_set"]
            if "aggregation_version" not in experiment_record["metadata"] and "aggregation_version" in metadata:
                experiment_record["metadata"]["aggregation_version"] = metadata["aggregation_version"]
        
        if "train_set" not in experiment_record["metadata"]:
            experiment_record["metadata"]["train_set"] = "results/predictions"  # По умолчанию
        if "aggregation_version" not in experiment_record["metadata"]:
            experiment_record["metadata"]["aggregation_version"] = "current"  # По умолчанию
        
        # Добавляем в реестр
        self.registry["experiments"].append(experiment_record)
        
        # Обновляем лучшие эксперименты
        self._update_best_experiments(experiment_record)
        
        # Сохраняем реестр
        self._save_registry()
        
        return exp_id
    
    def _update_best_experiments(self, experiment: Dict) -> None:
        """Обновляет информацию о лучших экспериментах"""
        metrics = experiment["metrics"]
        
        # Лучший по Score
        if "best_score" not in self.registry["best_experiments"]:
            self.registry["best_experiments"]["best_score"] = experiment
        elif metrics["score"] > self.registry["best_experiments"]["best_score"]["metrics"]["score"]:
            self.registry["best_experiments"]["best_score"] = experiment
        
        # Лучший по Separation
        if "best_separation" not in self.registry["best_experiments"]:
            self.registry["best_experiments"]["best_separation"] = experiment
        elif metrics["separation"] > self.registry["best_experiments"]["best_separation"]["metrics"]["separation"]:
            self.registry["best_experiments"]["best_separation"] = experiment
        
        # Лучший по Mod позиции
        if "best_mod_position" not in self.registry["best_experiments"]:
            self.registry["best_experiments"]["best_mod_position"] = experiment
        elif metrics["mean_pc1_norm_mod"] > self.registry["best_experiments"]["best_mod_position"]["metrics"]["mean_pc1_norm_mod"]:
            self.registry["best_experiments"]["best_mod_position"] = experiment
        
        # Лучший по объясненной дисперсии
        if "best_explained_variance" not in self.registry["best_experiments"]:
            self.registry["best_experiments"]["best_explained_variance"] = experiment
        elif metrics["explained_variance"] > self.registry["best_experiments"]["best_explained_variance"]["metrics"]["explained_variance"]:
            self.registry["best_experiments"]["best_explained_variance"] = experiment
    
    def get_best_experiments(self) -> Dict[str, Dict]:
        """Возвращает словарь с лучшими экспериментами по разным метрикам"""
        return self.registry.get("best_experiments", {})
    
    def list_experiments(
        self,
        sort_by: str = "score",
        limit: Optional[int] = None,
        filter_by: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Возвращает список экспериментов в виде DataFrame.
        
        Args:
            sort_by: Поле для сортировки (score, separation, mean_pc1_norm_mod, timestamp)
            limit: Максимальное число экспериментов для возврата
            filter_by: Словарь с фильтрами {field: value}
            
        Returns:
            DataFrame с экспериментами
        """
        experiments = self.registry.get("experiments", [])
        
        # Фильтрация
        if filter_by:
            filtered = []
            for exp in experiments:
                match = True
                for field, value in filter_by.items():
                    if field in exp.get("parameters", {}):
                        if exp["parameters"][field] != value:
                            match = False
                            break
                    elif field in exp.get("metadata", {}):
                        if exp["metadata"][field] != value:
                            match = False
                            break
                    else:
                        match = False
                        break
                if match:
                    filtered.append(exp)
            experiments = filtered
        
        # Преобразуем в DataFrame
        if not experiments:
            return pd.DataFrame()
        
        rows = []
        for exp in experiments:
            row = {
                "id": exp["id"],
                "name": exp["name"],
                "timestamp": exp["timestamp"],
                "score": exp["metrics"]["score"],
                "separation": exp["metrics"]["separation"],
                "mod_norm": exp["metrics"]["mean_pc1_norm_mod"],
                "explained_variance": exp["metrics"]["explained_variance"],
                "method": exp["parameters"]["method"],
                "n_features": exp["parameters"]["n_features"],
                "train_set": exp["metadata"].get("train_set", "unknown"),
                "aggregation_version": exp["metadata"].get("aggregation_version", "unknown"),
                "directory": exp["directory"],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Сортировка
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        
        # Ограничение
        if limit:
            df = df.head(limit)
        
        return df
    
    def compare_experiments(self, exp_ids: List[str]) -> pd.DataFrame:
        """
        Сравнивает несколько экспериментов.
        
        Args:
            exp_ids: Список ID экспериментов для сравнения
            
        Returns:
            DataFrame со сравнением
        """
        experiments = []
        for exp_id in exp_ids:
            exp = self._find_experiment_by_id(exp_id)
            if exp:
                experiments.append(exp)
        
        if not experiments:
            return pd.DataFrame()
        
        # Формируем таблицу сравнения
        comparison_rows = []
        for exp in experiments:
            row = {
                "id": exp["id"],
                "name": exp["name"],
                "method": exp["parameters"]["method"],
                "n_features": exp["parameters"]["n_features"],
                "score": exp["metrics"]["score"],
                "separation": exp["metrics"]["separation"],
                "mod_norm": exp["metrics"]["mean_pc1_norm_mod"],
                "explained_variance": exp["metrics"]["explained_variance"],
                "train_set": exp["metadata"].get("train_set", "unknown"),
                "timestamp": exp["timestamp"],
            }
            comparison_rows.append(row)
        
        return pd.DataFrame(comparison_rows)
    
    def _find_experiment_by_id(self, exp_id: str) -> Optional[Dict]:
        """Находит эксперимент по ID"""
        for exp in self.registry.get("experiments", []):
            if exp["id"] == exp_id:
                return exp
        return None
    
    def get_experiment_details(self, exp_id: str) -> Optional[Dict]:
        """Возвращает детальную информацию об эксперименте"""
        return self._find_experiment_by_id(exp_id)
    
    def export_summary_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Экспортирует сводный отчет по всем экспериментам.
        
        Args:
            output_path: Путь для сохранения отчета (None = автоматически)
            
        Returns:
            Путь к сохраненному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.experiments_dir / f"experiments_summary_{timestamp}.md"
        
        output_path = Path(output_path)
        
        # Получаем лучшие эксперименты
        best_exps = self.get_best_experiments()
        
        # Получаем топ-10 по Score
        top_experiments = self.list_experiments(sort_by="score", limit=10)
        
        # Формируем отчет
        report_lines = [
            "# Сводный отчет по экспериментам",
            "",
            f"**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Всего экспериментов:** {len(self.registry.get('experiments', []))}",
            "",
            "---",
            "",
            "## Лучшие эксперименты по метрикам",
            "",
        ]
        
        # Лучший по Score
        if "best_score" in best_exps:
            exp = best_exps["best_score"]
            report_lines.extend([
                "### 🏆 Лучший по Score (комплексная оценка)",
                "",
                f"- **Эксперимент:** {exp['name']}",
                f"- **Метод:** {exp['parameters']['method']}",
                f"- **Score:** {exp['metrics']['score']:.4f}",
                f"- **Separation:** {exp['metrics']['separation']:.4f}",
                f"- **Mod (норм. PC1):** {exp['metrics']['mean_pc1_norm_mod']:.4f}",
                f"- **Объясненная дисперсия:** {exp['metrics']['explained_variance']:.4f}",
                f"- **Признаков:** {exp['parameters']['n_features']}",
                f"- **Train set:** {exp['metadata'].get('train_set', 'unknown')}",
                f"- **Директория:** {exp['directory']}",
                "",
            ])
        
        # Лучший по Separation
        if "best_separation" in best_exps:
            exp = best_exps["best_separation"]
            report_lines.extend([
                "### 🎯 Лучший по Separation",
                "",
                f"- **Эксперимент:** {exp['name']}",
                f"- **Метод:** {exp['parameters']['method']}",
                f"- **Separation:** {exp['metrics']['separation']:.4f}",
                f"- **Score:** {exp['metrics']['score']:.4f}",
                "",
            ])
        
        # Топ-10 экспериментов
        if len(top_experiments) > 0:
            report_lines.extend([
                "---",
                "",
                "## Топ-10 экспериментов по Score",
                "",
                top_experiments.to_markdown(index=False),
                "",
            ])
        
        # Сохраняем отчет
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Сводный отчет сохранен: {output_path}")
        
        return output_path


def register_experiment_from_directory(
    experiment_dir: Path,
    tracker: Optional[ExperimentTracker] = None,
    train_set: Optional[str] = None,
    aggregation_version: Optional[str] = None,
) -> str:
    """
    Регистрирует эксперимент из директории эксперимента.
    
    Args:
        experiment_dir: Директория эксперимента
        tracker: Экземпляр ExperimentTracker (None = создается новый)
        train_set: Путь к train set (None = определяется автоматически)
        aggregation_version: Версия агрегации (None = "current")
        
    Returns:
        ID зарегистрированного эксперимента
    """
    experiment_dir = Path(experiment_dir)
    
    if tracker is None:
        tracker = ExperimentTracker()
    
    # Загружаем конфигурацию признаков
    json_files = list(experiment_dir.glob("best_features_*.json"))
    if not json_files:
        raise ValueError(f"Не найдено best_features_*.json в {experiment_dir}")
    
    best_file = sorted(json_files)[-1]
    with open(best_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Загружаем метаданные
    metadata_file = experiment_dir / "metadata.json"
    loaded_metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
    
    # Подготавливаем параметры
    parameters = {
        "selected_features": config.get("selected_features", []),
        "method": config.get("method", "unknown"),
        "use_relative_features": config.get("use_relative_features", True),
    }
    
    # Подготавливаем метрики
    metrics = config.get("metrics", {})
    
    # Объединяем метаданные: сначала из файла, потом переданные параметры
    metadata = loaded_metadata.copy()
    if train_set:
        metadata["train_set"] = train_set
    elif "train_set" not in metadata:
        metadata["train_set"] = "results/predictions"  # По умолчанию
    
    if aggregation_version:
        metadata["aggregation_version"] = aggregation_version
    elif "aggregation_version" not in metadata:
        metadata["aggregation_version"] = "current"  # По умолчанию
    
    # Регистрируем эксперимент
    exp_id = tracker.register_experiment(
        experiment_name=experiment_dir.name,
        experiment_dir=experiment_dir,
        metrics=metrics,
        parameters=parameters,
        metadata=metadata,
    )
    
    return exp_id

