import os
import cv2
import matplotlib.pyplot as plt

base_dir = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C5_project/KITTI-MOTS/yolo_format/dataset"
image_dir = os.path.join(base_dir, "images", "val", "0013")  # Change to "val" if needed
label_dir = os.path.join(base_dir, "labels", "val", "0013")  # Change to "val" if needed

def visualize_image_with_labels(image_name, image_dir, label_dir):
    # Read the image
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    label_name = image_name.replace('.png', '.txt')
    label_path = os.path.join(label_dir, label_name)

    if not os.path.exists(label_path):
        print(f"No label file for {image_name}")
        return

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Each line has: class_id center_x center_y width height
        parts = line.strip().split()
        class_id = int(parts[0])
        center_x, center_y, width, height = map(float, parts[1:])

        img_height, img_width, _ = image.shape
        x_center = int(center_x * img_width)
        y_center = int(center_y * img_height)
        w = int(width * img_width)
        h = int(height * img_height)

        x1 = x_center - w // 2
        y1 = y_center - h // 2
        x2 = x_center + w // 2
        y2 = y_center + h // 2

        color = (255, 0, 0)  # Red for pedestrian class (you can change colors based on class)
        thickness = 2
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = str(class_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(image, label, (x1, y1-10), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.show()

image_name = "000001.png"  # Change this to any image from your dataset
visualize_image_with_labels(image_name, image_dir, label_dir)
