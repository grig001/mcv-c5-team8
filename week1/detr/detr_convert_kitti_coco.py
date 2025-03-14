import os
import json
from PIL import Image

GT_FOLDER = "./gt_bboxes_detr"
IMAGE_FOLDER = "/home/c5mcv08/mcv/datasets/C5/KITTI-MOTS/training/image_02"
COCO_ANNOTATIONS = "./coco_annotations.json"

CLASS_MAP = {"Pedestrian": 1, "Car": 2}


def convert_to_coco(gt_folder, image_folder, output_json):
    images = []
    annotations = []
    categories = [
        {"id": 1, "name": "Pedestrian", "supercategory": "person"},
        {"id": 2, "name": "Car", "supercategory": "vehicle"}
    ]
    ann_id = 1
    img_id_map = {}

    for gt_file in sorted(os.listdir(gt_folder)):
        seq_id = gt_file.split("_")[-1].split(".")[0]
        with open(os.path.join(gt_folder, gt_file), "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            img_filename, label, score, x1, y1, x2, y2 = parts
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            width, height = x2 - x1, y2 - y1

            if label not in CLASS_MAP:
                continue

            img_path = os.path.join(image_folder, seq_id, img_filename)

            if img_filename not in img_id_map:
                img_id = len(images) + 1
                img_id_map[img_filename] = img_id
                img = Image.open(img_path)
                images.append({
                    "id": img_id,
                    "file_name": os.path.join(seq_id, img_filename),
                    "width": img.width,
                    "height": img.height
                })
            else:
                img_id = img_id_map[img_filename]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_MAP[label],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            ann_id += 1

    coco_format = {"images": images, "annotations": annotations, "categories": categories}
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"Saved COCO annotations to {output_json}")


convert_to_coco(GT_FOLDER, IMAGE_FOLDER, COCO_ANNOTATIONS)
