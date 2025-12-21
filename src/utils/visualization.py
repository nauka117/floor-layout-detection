from typing import Any, List, Optional
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from ultralytics.engine.results import Results


src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.wall import Wall


def draw_walls_on_image(
    image: Image.Image,
    walls: List[Wall],
    line_width: int = 3,
    color: str = "red"
) -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for wall in walls:
        if len(wall.points) >= 2:
            for i in range(len(wall.points) - 1):
                start = tuple(wall.points[i])
                end = tuple(wall.points[i + 1])
                draw.line([start, end], fill=color, width=line_width)
    
    return img_copy


def visualize_yolo_results(
    result,
    walls: Optional[List[Wall]] = None,
    highlight_walls: bool = True
) -> np.ndarray:
    if isinstance(result, list) and len(result) > 0:
        result_obj = result[0]
    elif isinstance(result, Results):
        result_obj = result
    else:
        raise ValueError(f"Unexpected result type: {type(result)}")
    
    annotated = result_obj.plot()[:, :, ::-1]  # BGR -> RGB
    
    if highlight_walls and walls:
        img = Image.fromarray(annotated)
        img = draw_walls_on_image(img, walls, line_width=4, color="blue")
        annotated = np.array(img)
    
    return annotated


def save_visualization(
    image: np.ndarray,
    output_path: Path,
    format: str = "PNG"
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(image)
    img.save(output_path, format=format)


def create_annotated_image(
    image_source: Any,
    result,
    walls: List[Wall],
    output_path: Optional[Path] = None
) -> Image.Image:
    annotated_array = visualize_yolo_results(result, walls, highlight_walls=True)
    annotated_image = Image.fromarray(annotated_array)
    
    if output_path:
        save_visualization(annotated_array, output_path)
    
    return annotated_image

