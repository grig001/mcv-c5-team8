import json
import numpy as np
from pycocotools import mask as mask_utils

def polygon_to_rle(segmentation, image_height, image_width):
    """Convert polygon segmentation to RLE format using pycocotools."""
    rle = mask_utils.frPyObjects(segmentation, image_height, image_width)
    if isinstance(rle, list):
        rle = rle[0]  # Handle multiple RLE segments
    rle["counts"] = rle["counts"].decode("utf-8")  # Convert bytes to string
    return rle

def convert_segmentation_to_rle(json_path, output_path):
    """Convert polygon segmentations in a COCO JSON file to RLE format."""
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    for annotation in coco_data["annotations"]:
        segmentation = annotation["segmentation"]

        if isinstance(segmentation, list) and all(isinstance(i, list) for i in segmentation):
            # Convert polygon to RLE
            image_id = annotation["image_id"]
            image_info = next(img for img in coco_data["images"] if img["id"] == image_id)
            height, width = image_info["height"], image_info["width"]
            annotation["segmentation"] = polygon_to_rle(segmentation, height, width)

    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=4)

# Run the conversion
convert_segmentation_to_rle("Domain_shift_data/COCO_Football Pixel.json", "COCO_Football_rle.json")
