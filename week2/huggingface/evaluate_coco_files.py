import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(eval_json_path: str, predictions_json_path: str):

    # Load ground truth annotations, predicted results and initialize COCO evaluation
    coco_gt = COCO(eval_json_path)
    coco_dt = coco_gt.loadRes(predictions_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate mask2former segmentation results.")

    # Default paths
    parser.add_argument("--eval", type=str, default="datasets/eval_gt.json",
                        help="Path to the ground truth annotations in COCO format.")
    parser.add_argument("--predictions", type=str, default="predictions/predictions_pretrained.json",
                        help="Path to the predicted results in COCO format.")

    args = parser.parse_args()

    evaluate_coco(args.eval, args.predictions)
