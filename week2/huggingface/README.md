## Folder Structure

```bash
huggingface/
├── datasets
│   ├── eval_gt.json
│   ├── train_gt_all.json
│   ├── train_gt.json
│   └── val_gt.json
├── evaluate_coco_files.py
├── fine_tune.py
├── inference_pretrained.py
├── job_mask2former
├── job_output
│   └── job_example
├── predictions
│   └── predictions_pretrained.json
└── README.md
```

## Usage

### Run inference on a pretrained model and generate evaluation metrics
```bash
python inference_pretrained.py
```
This function uses datasets that are directly linked to the first created KITTI-MOTS split folder.


### Fine-tune the Mask2Former model on the KITTI-MOTS dataset
```bash
python fine_tune.py
```

This function uses datasets that are directly linked to the first created KITTI-MOTS split folder.

### Evaluate predictions using COCO evaluation
```bash
python evaluate_coco_files.py --eval datasets/eval_gt.json --predictions predictions/predictions_pretrained.json
```

This script compares a set of predicted segmentation results against ground truth annotations using the COCO evaluation framework.

### Running jobs via SLURM
Instead of running the scripts directly, you can submit them as SLURM jobs by adjusting the `job_mask2former` file, making it executable, and submitting it:

```bash
chmod +x job_mask2former
sbatch job_mask2former
```

This allows execution on a compute cluster using SLURM job scheduling.


## Dependencies
Ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- OpenCV
- pycocotools
- json
