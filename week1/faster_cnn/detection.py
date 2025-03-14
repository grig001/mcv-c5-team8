import os
import json
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2

val = [2, 6, 7, 8, 10, 13, 14, 16, 18]

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg.MODEL.WEIGHTS = "models/faster_2/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda:0"
    return cfg

def run_inference(cfg, image_path):

    predictor = DefaultPredictor(cfg)
    
    image = cv2.imread(image_path)
    
    outputs = predictor(image)
    
    instances = outputs["instances"]
    pred_boxes = instances.pred_boxes.tensor
    
    pred_boxes_xywh = []
    for box in pred_boxes:
        x1, y1, x2, y2 = box.tolist()  
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        pred_boxes_xywh.append([x, y, w, h])
    
    outputs["instances"].pred_boxes = torch.tensor(pred_boxes_xywh)
    return outputs



def format_detections(outputs, image_id):
    detections = []
    
    boxes = outputs["instances"].pred_boxes.cpu().numpy()  
    scores = outputs["instances"].scores.cpu().numpy()  
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()

    for i in range(len(boxes)):
        detection = {
            "image_id": image_id,
            "category_id": int(pred_classes[i]), 
            "bbox": boxes[i].tolist(),  
            "score": float(scores[i])  
        }
        detections.append(detection)
    
    return detections

def detect_and_save(cfg, image_dir, output_json):

    valid_folders = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018'] 
    folder_paths = []
    for folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder)
        if os.path.isdir(folder_path):
            if folder in valid_folders:  
                folder_paths.append(folder_path)

    print(folder_paths)
    all_detections = []

    for folder_path in folder_paths:
        print(folder_path)

        folder_id = int(os.path.basename(folder_path))

        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        
        for image_path in image_files:

                image_id = int(image_path[0:6])

                outputs = run_inference(cfg, folder_path + '/' + image_path)
                
                detections = format_detections(outputs, int(1e5 * folder_id + image_id))
                all_detections.extend(detections)


    print(len(all_detections))
    with open(output_json, "w") as f:
        json.dump(all_detections, f, indent=4)

if __name__ == "__main__":
    cfg = setup_cfg()
    
    image_dir = "dataset/image_02/"
    
    output_json = "det_faster_rcnn_fine_tune_faster_3.json"
    
    detect_and_save(cfg, image_dir, output_json)
