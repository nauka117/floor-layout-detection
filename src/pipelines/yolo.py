from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models/yolo-v8/pretrained.pt"


def run_inference(image_source: Any, model_path: Path = DEFAULT_MODEL_PATH):
    """Run YOLOv8 inference and return an annotated numpy array (RGB)."""
    model = YOLO(str(model_path))
    image = Image.open(image_source)
    result = model.predict(image)
    
    return result[0].plot()[:, :, ::-1]

