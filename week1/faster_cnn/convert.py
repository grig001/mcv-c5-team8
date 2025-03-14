import json
import os
import cv2

image_path = "dataset/image_02/"
input_folder = "gt_bboxes"
output_folder = "json_annotations"
os.makedirs(output_folder, exist_ok=True)

for file_idx in range(21):
    gt_file = os.path.join(input_folder, f"gt_bboxes_{file_idx:04d}.txt")
    output_json = os.path.join(output_folder, f"gt_bboxes_{file_idx:04d}.json")

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}  
    image_id_map = []  
    annotation_id = 1  

    if not os.path.exists(gt_file):
        print(f"File {gt_file} not found, skipping...")
        continue

    with open(gt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(int, line.strip().split()))
        frame_id, obj_id, class_id, x_min, y_min, x_max, y_max = parts

        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        if frame_id not in image_id_map:
            image_id_map.append(frame_id)


            folder_index = str(file_idx).zfill(4)
            img = cv2.imread(image_path + folder_index + '/' + f"{frame_id:06d}.png")  # Load the image
            height, width, channels = img.shape

            coco_data["images"].append({
                "id": int(1e5 * file_idx + frame_id),
                "file_name": f"{frame_id:06d}.png",
                "height": height,  
                "width": width
            })

        if class_id not in category_map:
            category_map[class_id] = len(category_map) + 1
            coco_data["categories"].append({
                "id": category_map[class_id] - 1,
                "name": f"class_{class_id}",
                "supercategory": "object"
            })

        if category_map[class_id] == 3:
            category_map[class_id] -= 1
        

        coco_data["annotations"].append({
            "id": int(1e5 * file_idx + annotation_id),
            "image_id": int(1e5 * file_idx + frame_id),
            "category_id": category_map[class_id] - 1,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        annotation_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Saved {output_json}")

