import json

output_train = "dataset/merged_annotations/train_fine_tune.json"
output_val = "dataset/merged_annotations/val_fine_tune.json"

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
    annotations = f"json_annotations/gt_bboxes_{folder_index}.json"

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


