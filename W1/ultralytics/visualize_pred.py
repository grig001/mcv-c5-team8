import cv2
import os

# Paths (Set these correctly)
image_folder = "../KITTI-MOTS/training/image_02/0013"  # Folder containing KITTI images
gt_bboxes_file = "gt_bboxes/gt_bboxes_0013.txt"  # Adjust for your sequence
pred_bboxes_file = "output_yolov8_kitti_mots_txt/predictions_0013.txt"  # Your YOLO predictions

# Define colors
GT_COLOR = (0, 255, 0)  # Green for ground truth
PRED_COLOR = (0, 0, 255)  # Red for predictions

def load_bboxes_from_txt(txt_file):
    """Loads bounding boxes from a given text file."""
    bboxes = {}
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) < 6:
                continue  # Skip invalid lines

            frame_id = parts[0].split(".")[0]  # Extract frame number
            class_id = parts[1] if len(parts) > 6 else "Unknown"  # Class name or ID
            conf = float(parts[2]) if len(parts) > 6 else None  # Confidence (only for YOLO)
            x_min, y_min, x_max, y_max = map(int, parts[-4:])  # Bounding box

            if frame_id not in bboxes:
                bboxes[frame_id] = []
            
            bboxes[frame_id].append((class_id, conf, (x_min, y_min, x_max, y_max)))
    
    return bboxes

def draw_bboxes(image, bboxes, color, label_prefix=""):
    """Draws bounding boxes on an image."""
    for class_id, conf, (x_min, y_min, x_max, y_max) in bboxes:
        label = f"{label_prefix}{class_id}"
        if conf is not None:
            label += f" {conf:.2f}"  # Add confidence if available

        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def visualize_image(image_id):
    """Loads an image and overlays ground truth & predicted bounding boxes."""
    img_path = os.path.join(image_folder, f"{image_id}.png")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    # Load image
    image = cv2.imread(img_path)

    # Load bounding boxes
    gt_bboxes = load_bboxes_from_txt(gt_bboxes_file)
    pred_bboxes = load_bboxes_from_txt(pred_bboxes_file)

    # Draw bounding boxes
    if image_id in gt_bboxes:
        draw_bboxes(image, gt_bboxes[image_id], GT_COLOR, "GT: ")
    if image_id in pred_bboxes:
        draw_bboxes(image, pred_bboxes[image_id], PRED_COLOR, "Pred: ")

    # Show the image
    cv2.imshow(f"Bounding Boxes - {image_id}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_id = input("Enter image ID: ")  # Example: "000000"
visualize_image(image_id)
