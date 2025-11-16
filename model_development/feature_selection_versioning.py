"""
Система версионирования и управления результатами подбора признаков.

Позволяет сохранять результаты экспериментов без перезаписи и выбирать,
какой результат экспортировать в dashboard.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import shutil


class FeatureSelectionVersionManager:
    """
    Менеджер версий результатов подбора признаков.
    """
    
    def __init__(self, experiments_base_dir: Path = Path("experiments/feature_selection")):
        """
        Args:
            experiments_base_dir: Базовая директория для экспериментов
        """
        self.experiments_base_dir = Path(experiments_base_dir)
        self.experiments_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Файл с метаданными всех экспериментов
        self.metadata_file = self.experiments_base_dir / "experiments_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Загружает метаданные экспериментов."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self):
        """Сохраняет метаданные экспериментов."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def create_experiment(
        self,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """
        Создает новую директорию для эксперимента.
        
        Args:
            experiment_name: Имя эксперимента (если None, генерируется автоматически)
            description: Описание эксперимента
            tags: Теги для поиска (например, ['forward_selection', 'no_paneth'])
            
        Returns:
            Путь к директории эксперимента
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        experiment_dir = self.experiments_base_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем метаданные эксперимента
        experiment_metadata = {
            'name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'description': description or '',
            'tags': tags or [],
            'status': 'in_progress',
        }
        
        self.metadata[experiment_name] = experiment_metadata
        self._save_metadata()
        
        return experiment_dir
    
    def finalize_experiment(
        self,
        experiment_name: str,
        best_method: str,
        metrics: Dict,
        selected_features: List[str],
    ):
        """
        Финализирует эксперимент, сохраняя результаты.
        
        Args:
            experiment_name: Имя эксперимента
            best_method: Лучший метод
            metrics: Метрики лучшего метода
            selected_features: Отобранные признаки
        """
        if experiment_name not in self.metadata:
            raise ValueError(f"Эксперимент {experiment_name} не найден")
        
        experiment_dir = self.experiments_base_dir / experiment_name
        
        # Обновляем метаданные
        self.metadata[experiment_name].update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'best_method': best_method,
            'metrics': {
                'score': float(metrics.get('score', 0)),
                'separation': float(metrics.get('separation', 0)),
                'mean_pc1_norm_mod': float(metrics.get('mean_pc1_norm_mod', 0)),
                'explained_variance': float(metrics.get('explained_variance', 0)),
            },
            'n_features': len(selected_features),
        })
        
        self._save_metadata()
    
    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Возвращает список экспериментов.
        
        Args:
            status: Фильтр по статусу ('completed', 'in_progress')
            tags: Фильтр по тегам
            
        Returns:
            DataFrame с информацией об экспериментах
        """
        experiments = []
        
        for exp_name, exp_data in self.metadata.items():
            # Фильтрация по статусу
            if status and exp_data.get('status') != status:
                continue
            
            # Фильтрация по тегам
            if tags:
                exp_tags = exp_data.get('tags', [])
                if not any(tag in exp_tags for tag in tags):
                    continue
            
            experiments.append({
                'name': exp_name,
                'created_at': exp_data.get('created_at', ''),
                'status': exp_data.get('status', 'unknown'),
                'best_method': exp_data.get('best_method', ''),
                'score': exp_data.get('metrics', {}).get('score', 0),
                'separation': exp_data.get('metrics', {}).get('separation', 0),
                'mod_norm': exp_data.get('metrics', {}).get('mean_pc1_norm_mod', 0),
                'n_features': exp_data.get('n_features', 0),
                'tags': ', '.join(exp_data.get('tags', [])),
                'description': exp_data.get('description', ''),
            })
        
        if not experiments:
            return pd.DataFrame()
        
        df = pd.DataFrame(experiments)
        df = df.sort_values('created_at', ascending=False)
        return df
    
    def get_best_experiment(self, metric: str = 'score') -> Optional[str]:
        """
        Возвращает имя лучшего эксперимента по указанной метрике.
        
        Args:
            metric: Метрика для сравнения ('score', 'separation', 'mod_norm')
            
        Returns:
            Имя лучшего эксперимента или None
        """
        completed_experiments = self.list_experiments(status='completed')
        
        if len(completed_experiments) == 0:
            return None
        
        if metric == 'score':
            best = completed_experiments.loc[completed_experiments['score'].idxmax()]
        elif metric == 'separation':
            best = completed_experiments.loc[completed_experiments['separation'].idxmax()]
        elif metric == 'mod_norm':
            best = completed_experiments.loc[completed_experiments['mod_norm'].idxmax()]
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")
        
        return best['name']
    
    def export_to_dashboard(
        self,
        experiment_name: str,
        backup_current: bool = True,
    ) -> Path:
        """
        Экспортирует результаты эксперимента в dashboard.
        
        Args:
            experiment_name: Имя эксперимента для экспорта (или путь к директории)
            backup_current: Создать резервную копию текущей конфигурации dashboard
            
        Returns:
            Путь к конфигурационному файлу dashboard
        """
        # Проверяем, является ли это путем к директории
        experiment_path = Path(experiment_name)
        if experiment_path.is_absolute() and experiment_path.exists() and experiment_path.is_dir():
            experiment_dir = experiment_path
        elif experiment_path.exists() and experiment_path.is_dir():
            # Относительный путь, который существует
            experiment_dir = experiment_path
        else:
            # Ищем в базовой директории
            experiment_dir = self.experiments_base_dir / experiment_name
            if not experiment_dir.exists():
                # Пробуем найти в experiments/ (на уровень выше)
                parent_experiments_dir = self.experiments_base_dir.parent
                experiment_dir = parent_experiments_dir / experiment_name
                if not experiment_dir.exists():
                    raise ValueError(f"Эксперимент {experiment_name} не найден. Проверьте путь.")
        
        # Ищем JSON файл с лучшими признаками
        json_files = list(experiment_dir.glob("best_features_*.json"))
        if not json_files:
            raise ValueError(f"Не найден файл best_features_*.json в {experiment_dir}")
        
        # Берем последний файл
        best_features_file = sorted(json_files)[-1]
        
        # Загружаем конфигурацию
        with open(best_features_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Путь к конфигурации dashboard
        dashboard_config_path = Path(__file__).parent / "feature_selection_config_relative.json"
        
        # Создаем резервную копию текущей конфигурации
        if backup_current and dashboard_config_path.exists():
            backup_path = dashboard_config_path.parent / f"feature_selection_config_relative_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(dashboard_config_path, backup_path)
            print(f"✓ Резервная копия создана: {backup_path}")
        
        # Формируем конфигурацию для dashboard
        dashboard_config = {
            "selected_features": config['selected_features'],
            "description": (
                f"Экспортировано из эксперимента '{experiment_name}'. "
                f"Метод: {config.get('method', 'unknown')}. "
                f"Score: {config.get('metrics', {}).get('score', 0):.4f}, "
                f"Separation: {config.get('metrics', {}).get('separation', 0):.4f}, "
                f"Mod (норм. PC1): {config.get('metrics', {}).get('mean_pc1_norm_mod', 0):.4f}"
            ),
            "last_updated": datetime.now().isoformat(),
            "method": config.get('method', 'unknown'),
            "metrics": config.get('metrics', {}),
            "n_features": len(config['selected_features']),
            "source_experiment": experiment_name,
        }
        
        # Сохраняем в dashboard
        with open(dashboard_config_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Конфигурация экспортирована в dashboard: {dashboard_config_path}")
        print(f"  Источник: {experiment_name}")
        print(f"  Метод: {config.get('method', 'unknown')}")
        print(f"  Score: {config.get('metrics', {}).get('score', 0):.4f}")
        
        return dashboard_config_path
    
    def compare_experiments(
        self,
        experiment_names: List[str],
    ) -> pd.DataFrame:
        """
        Сравнивает несколько экспериментов.
        
        Args:
            experiment_names: Список имен экспериментов для сравнения
            
        Returns:
            DataFrame с сравнением
        """
        comparison = []
        
        for exp_name in experiment_names:
            if exp_name not in self.metadata:
                continue
            
            exp_data = self.metadata[exp_name]
            comparison.append({
                'experiment': exp_name,
                'method': exp_data.get('best_method', ''),
                'score': exp_data.get('metrics', {}).get('score', 0),
                'separation': exp_data.get('metrics', {}).get('separation', 0),
                'mod_norm': exp_data.get('metrics', {}).get('mean_pc1_norm_mod', 0),
                'explained_variance': exp_data.get('metrics', {}).get('explained_variance', 0),
                'n_features': exp_data.get('n_features', 0),
                'created_at': exp_data.get('created_at', ''),
                'tags': ', '.join(exp_data.get('tags', [])),
            })
        
        return pd.DataFrame(comparison)


def list_all_experiments(experiments_dir: Path = Path("experiments/feature_selection")) -> pd.DataFrame:
    """
    Утилита для вывода списка всех экспериментов.
    
    Args:
        experiments_dir: Директория с экспериментами
        
    Returns:
        DataFrame с экспериментами
    """
    manager = FeatureSelectionVersionManager(experiments_dir)
    return manager.list_experiments()


def export_experiment_to_dashboard(
    experiment_name: str,
    experiments_dir: Path = Path("experiments/feature_selection"),
    backup_current: bool = True,
) -> Path:
    """
    Утилита для экспорта эксперимента в dashboard.
    
    Args:
        experiment_name: Имя эксперимента
        experiments_dir: Директория с экспериментами
        backup_current: Создать резервную копию текущей конфигурации
        
    Returns:
        Путь к конфигурационному файлу dashboard
    """
    manager = FeatureSelectionVersionManager(experiments_dir)
    return manager.export_to_dashboard(experiment_name, backup_current=backup_current)

