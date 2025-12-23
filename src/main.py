import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from pipelines.yolo import detect_layout, get_detection_results, DEFAULT_MODEL_PATH
from utils.visualization import create_annotated_image
from utils.json_export import format_walls_to_json, export_to_json
from utils.output_paths import prepare_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Detect floor layout from an image and extract walls as JSON."
    )
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="Path(s) to input image file(s) (png or jpeg).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the YOLO model weights (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save JSON output. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Optional path to save the annotated image.",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Show the annotated image (requires display).",
    )

    args = parser.parse_args()

    # Load model once for batch processing
    model = YOLO(str(args.model))

    is_single = len(args.images) == 1
    for image_path in args.images:
        print(f"\nProcessing image: {image_path}")
        walls, source_name = detect_layout(image_path, args.model, model=model)
        result_json = format_walls_to_json(walls, source_name)
        
        json_path = prepare_output_path(
            source_name,
            args.output_json,
            is_single=is_single,
            default_suffix=".json",
        )

        json_str = json.dumps(result_json, indent=2, ensure_ascii=False)
        if json_path:
            export_to_json(result_json, json_path)
            print(f"JSON saved to {json_path}")
        else:
            print("\n=== Detection Results (JSON) ===")
            print(json_str)
        
        need_image = args.output_image or args.show_image
        if need_image:
            result, walls, image = get_detection_results(image_path, args.model, model=model)
            image_path_to_save = prepare_output_path(
                source_name,
                args.output_image,
                is_single=is_single,
                default_suffix=".png",
            )
            annotated_image = create_annotated_image(
                image_path,
                result,
                walls,
                output_path=image_path_to_save
            )
            
            if image_path_to_save:
                print(f"Annotated image saved to {image_path_to_save}")
            
            if args.show_image:
                annotated_image.show()





if __name__ == "__main__":
    main()