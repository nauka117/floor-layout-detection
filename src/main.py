
import argparse
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models/yolo-v8/pretrained.pt"


def run_inference(image_source: Any, model_path: Path = DEFAULT_MODEL_PATH):
    """Run YOLOv8 inference and return an annotated numpy array (RGB)."""
    model = YOLO(str(model_path))
    image = Image.open(image_source)
    result = model.predict(image)
    
    return result[0].plot()[:, :, ::-1]


def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title(":camera: Computer vision app")

    with st.sidebar:
        st.title("Upload a picture")
        image_bytes = st.file_uploader(
            "Upload image file", type=[".png", ".jpeg"], accept_multiple_files=False
        )

    st.write("## Uploaded picture")
    if image_bytes:
        st.write("🎉 Here's what you uploaded!")
        st.image(image_bytes, width=200)
    else:
        st.warning("👈 Please upload an image first...")
        st.stop()

    st.write("## YOLOv8 Object Detection")
    res_plotted = run_inference(image_bytes)
    st.image(res_plotted, caption="Detected objects", use_column_width=True)


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


def _is_running_with_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _is_running_with_streamlit():
        run_streamlit_app()
    else:
        main()