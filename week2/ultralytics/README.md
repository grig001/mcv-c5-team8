## Folder Structure

```bash
ultralytics/
├── datasets
│   ├── yolo_fine_tune.yaml
│   └── yolo_pretrained.yaml
├── evaluate_fine_tuned.py
├── fine_tune_augm.py
├── fine_tune_optuna.py
├── fine_tune.py
├── inference_pretrained.py
├── job_output
│   └── job_example
├── job_yolo
├── prepare_yolo_data.py
└── README.md
```

## Usage

### Preparing the YOLO Dataset
The `datasets` folder contains the dataset configuration files required for training and evaluation. Before running any training or inference scripts, ensure the dataset is properly formatted and organized.

Use the `prepare_yolo_data.py` script to structure the dataset for YOLO training:
```bash
python prepare_yolo_data.py
```

### Running inference on a pretrained model
```bash
python inference_pretrained.py
```

This script evaluates the YOLOv8 segmentation model using the test dataset defined in `datasets/yolo_pretrained.yaml` and saves the results in COCO evaluation format.

### Fine-tuning YOLOv8 on a custom dataset
```bash
python fine_tune.py
```

This script trains the YOLOv8 segmentation model using the dataset defined in `datasets/yolo_fine_tune.yaml`.

### Hyperparameter optimization with Optuna
```bash
python fine_tune_optuna.py
```

This script optimizes the YOLOv8 training hyperparameters using Optuna and selects the best-performing model.

### Fine-tuning with data augmentation
```bash
python fine_tune_augm.py
```

This script fine-tunes the YOLOv8 segmentation model with additional data augmentation techniques.

### Evaluating a fine-tuned model
```bash
python evaluate_fine_tuned.py
```

This script evaluates a fine-tuned YOLOv8 segmentation model using the test dataset defined in `datasets/yolo_fine_tune.yaml`.

### Running jobs via SLURM
Instead of running the scripts manually, you can submit them as SLURM jobs by adjusting the `job_yolo` file, making it executable, and submitting it:
```bash
chmod +x job_yolo
sbatch job_yolo
```

This allows execution on a compute cluster using SLURM job scheduling.

## Dependencies
Ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- OpenCV
- pycocotools
- ultralytics
- optuna
- json
