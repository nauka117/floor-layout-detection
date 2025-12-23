from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple, Optional
import sys

import numpy as np
from PIL import Image
from ultralytics import YOLO

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.wall import Wall


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "weights/yolo-v8/pretrained.pt"

WALL_CLASS_ID = 7


def _bbox_to_line_points(bbox: np.ndarray) -> List[List[float]]:
    x1, y1, x2, y2 = bbox.tolist()
    
    width = x2 - x1
    height = y2 - y1
    
    if width > height:
        center_y = (y1 + y2) / 2
        return [[x1, center_y], [x2, center_y]]
    else:
        center_x = (x1 + x2) / 2
        return [[center_x, y1], [center_x, y2]]


def extract_walls_from_detections(result, image_source: Any) -> List[Wall]:
    if len(result) == 0:
        return []
    
    r = result[0]
    walls = []
    
    if not hasattr(r, 'boxes') or r.boxes is None:
        return walls
    
    classes = r.boxes.cls.cpu().numpy()
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confidences = r.boxes.conf.cpu().numpy()
    
    wall_indices = np.where(classes == WALL_CLASS_ID)[0]
    
    for idx, wall_idx in enumerate(wall_indices):
        bbox = boxes_xyxy[wall_idx]
        confidence = float(confidences[wall_idx])
        
        points = _bbox_to_line_points(bbox)
        
        wall = Wall(
            id=f"w{idx + 1}",
            points=points,
            confidence=confidence,
            bbox=tuple(bbox.tolist())
        )
        walls.append(wall)
    
    return walls


def detect_layout(
    image_source: Any,
    model_path: Path = DEFAULT_MODEL_PATH,
    model: Optional[YOLO] = None,
) -> Tuple[List[Wall], str]:
    yolo_model = model or YOLO(str(model_path))
    
    if isinstance(image_source, (str, Path)):
        image_path = Path(image_source)
        image = Image.open(image_path)
        source_name = image_path.name
    else:
        image = Image.open(image_source)
        source_name = getattr(image_source, 'name', 'uploaded_image')
    
    result = yolo_model.predict(image)
    
    walls = extract_walls_from_detections(result, image_source)
    
    return walls, source_name


def get_detection_results(
    image_source: Any,
    model_path: Path = DEFAULT_MODEL_PATH,
    model: Optional[YOLO] = None,
):
    yolo_model = model or YOLO(str(model_path))
    image = Image.open(image_source)
    result = yolo_model.predict(image)
    walls = extract_walls_from_detections(result, image_source)
    
    return result, walls, image

