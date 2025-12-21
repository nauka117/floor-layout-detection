import argparse
import json
from pathlib import Path

from pipelines.yolo import detect_layout, get_detection_results, DEFAULT_MODEL_PATH
from utils.visualization import create_annotated_image
from utils.json_export import format_walls_to_json, export_to_json


def main():
    parser = argparse.ArgumentParser(
        description="Detect floor layout from an image and extract walls as JSON."
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

    print(f"Processing image: {args.image}")
    walls, source_name = detect_layout(args.image, args.model)
    
    
    result_json = format_walls_to_json(walls, source_name)
    
    
    json_str = json.dumps(result_json, indent=2, ensure_ascii=False)
    if args.output_json:
        export_to_json(result_json, args.output_json)
        print(f"JSON saved to {args.output_json}")
    else:
        print("\n=== Detection Results (JSON) ===")
        print(json_str)
    
    
    if args.output_image or args.show_image:
        result, walls, image = get_detection_results(args.image, args.model)
        annotated_image = create_annotated_image(
            args.image,
            result,
            walls,
            output_path=args.output_image
        )
        
        if args.output_image:
            print(f"Annotated image saved to {args.output_image}")
        
        if args.show_image:
            annotated_image.show()





if __name__ == "__main__":
    main()