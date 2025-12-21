from typing import Any, Dict, List
from pathlib import Path
import sys
import json


src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.wall import Wall


def format_walls_to_json(
    walls: List[Wall],
    source_name: str = "image"
) -> Dict:
    return {
        "meta": {
            "source": source_name
        },
        "walls": [wall.to_dict() for wall in walls]
    }


def export_to_json(
    output: Dict,
    output_path: Path
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

