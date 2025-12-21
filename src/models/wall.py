from typing import List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Wall:
    """Класс для представления стены."""
    
    id: str
    points: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    confidence: float = 0.0
    bbox: Tuple[float, float, float, float] = None  # (x1, y1, x2, y2) for visualisation
    
    def to_dict(self) -> dict:
        """Преобразовать стену в словарь для JSON."""
        return {
            "id": self.id,
            "points": self.points
        }
    
    def __repr__(self) -> str:
        return f"Wall(id={self.id}, points={len(self.points)} points, conf={self.confidence:.3f})"

