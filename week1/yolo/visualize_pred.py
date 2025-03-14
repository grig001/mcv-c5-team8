import cv2
import os

# This script shows images from sequence 0000 with bboxes from predictions and gt for debugging purposes.

image_folder = "../KITTI-MOTS/training/image_02/0000"  
gt_bboxes_file = "gt_bboxes/gt_bboxes_0000.txt"  
pred_bboxes_file = "output_yolov8_kitti_mots_txt/predictions_0000.txt" 

GT_COLOR = (0, 255, 0)  
PRED_COLOR = (0, 0, 255) 

def load_bboxes_from_txt(txt_file):
    """Loads bounding boxes from a given text file."""
    bboxes = {}
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) < 6:
                continue  

            frame_id = parts[0].split(".")[0] 
            class_id = parts[1] if len(parts) > 6 else "Unknown"  
            conf = float(parts[2]) if len(parts) > 6 else None  
            x_min, y_min, x_max, y_max = map(int, parts[-4:])  

            if frame_id not in bboxes:
                bboxes[frame_id] = []
            
            bboxes[frame_id].append((class_id, conf, (x_min, y_min, x_max, y_max)))
    
    return bboxes

def draw_bboxes(image, bboxes, color, label_prefix=""):
    """Draws bounding boxes on an image."""
    for class_id, conf, (x_min, y_min, x_max, y_max) in bboxes:
        label = f"{label_prefix}{class_id}"
        if conf is not None:
            label += f" {conf:.2f}" 

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def visualize_image(image_id):
    """Loads an image and overlays ground truth & predicted bounding boxes."""
    img_path = os.path.join(image_folder, f"{image_id}.png")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    image = cv2.imread(img_path)

    gt_bboxes = load_bboxes_from_txt(gt_bboxes_file)
    pred_bboxes = load_bboxes_from_txt(pred_bboxes_file)

    if image_id in gt_bboxes:
        draw_bboxes(image, gt_bboxes[image_id], GT_COLOR, "GT: ")
    if image_id in pred_bboxes:
        draw_bboxes(image, pred_bboxes[image_id], PRED_COLOR, "Pred: ")

    cv2.imshow(f"Bounding Boxes - {image_id}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_id = input("Enter image ID: ")  # Example: "000000"
visualize_image(image_id)
