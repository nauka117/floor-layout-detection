
import argparse
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

from ui import _is_running_with_streamlit


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models/yolo-v8/pretrained.pt"


def run_inference(image_source: Any, model_path: Path = DEFAULT_MODEL_PATH):
    """Run YOLOv8 inference and return an annotated numpy array (RGB)."""
    model = YOLO(str(model_path))
    image = Image.open(image_source)
    result = model.predict(image)
    
    return result[0].plot()[:, :, ::-1]





def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference on an image from the console."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image file (png or jpeg).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the YOLO model weights (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the annotated image. Shows the image if omitted.",
    )

    args = parser.parse_args()
    res_plotted = run_inference(args.image, args.model)
    annotated_image = Image.fromarray(res_plotted)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        annotated_image.save(args.output)
        print(f"Saved annotated image to {args.output}")
    else:
        annotated_image.show()





if __name__ == "__main__":
    if _is_running_with_streamlit():
        from ui.streamlit import run_streamlit_app
        run_streamlit_app()
    else:
        main()