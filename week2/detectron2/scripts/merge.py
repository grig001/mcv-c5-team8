import json
import os

output_train = "w2/merged_annotations/fine_tune_train.json"
output_val = "w2/merged_annotations/fine_tune_val.json"

# Function to check and create empty JSON file if missing
def ensure_json_exists(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directories if needed
        with open(file_path, "w") as f:
            json.dump({"images": [], "annotations": [], "categories": []}, f, indent=4)
        print(f"Created new empty JSON file: {file_path}")
    else:
        print(f"File already exists: {file_path}")

# Check both files
ensure_json_exists(output_train)
ensure_json_exists(output_val)


train_coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

val_coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}


val = [2, 6, 7, 8, 10, 13, 14, 16, 18]

for i in range(21):

    folder_index = str(i).zfill(4)

    image_path = f"dataset/image_02/{folder_index}/"
    annotations = f"w2/json_annotations/gt_bboxes_{folder_index}.json"

    with open(annotations, 'r') as infile:
        coco_data = json.load(infile)

    for image in coco_data['images']:
        
        image['file_name'] = folder_index + '_' + image['file_name']

        if i in val:
            val_coco_data['images'].append(image)
        else:
            train_coco_data['images'].append(image)

    for annotation in coco_data['annotations']:

        if i in val:
            val_coco_data['annotations'].append(annotation)
        else:
            train_coco_data['annotations'].append(annotation)


for category in coco_data['categories']:    
    train_coco_data['categories'].append(category)
    val_coco_data['categories'].append(category)

print(len(val_coco_data['images']))
print(len(train_coco_data['images']))


with open(output_train, 'w') as outfile:
    json.dump(train_coco_data, outfile, indent=4)

with open(output_val, 'w') as outfile:
    json.dump(val_coco_data, outfile, indent=4)


