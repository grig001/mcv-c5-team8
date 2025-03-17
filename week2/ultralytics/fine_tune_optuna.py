import optuna
from ultralytics import YOLO


def objective(trial):
    """Optimize YOLOv8 hyperparameters using Optuna."""

    # Hyperparameter Search
    epochs = trial.suggest_int("epochs", 5, 20)
    batch = trial.suggest_categorical("batch", [8, 16, 32])
    imgsz = trial.suggest_categorical("imgsz", [416, 512, 640])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # Load YOLO Model
    model = YOLO("models/yolov8n-seg.pt")

    # Train YOLO Model
    results = model.train(
        data="datasets/yolo_fine_tune.yaml",
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device="cuda",
        lr0=learning_rate,
        name=f"optuna_trial_{trial.number}"
    )

    print(f"\nAvailable keys in results_dict for Trial {trial.number}: {list(results.results_dict.keys())}")
    trial.set_user_attr("results_dict", results.results_dict)
    best_map = results.results_dict.get("metrics/mAP50-95(B)", None)

    if best_map is None:
        raise ValueError(f"No valid mAP found in results for trial {trial.number}. Check logs.")

    print(f"Using mAP Key: metrics/mAP50-95(B) -> Value: {best_map}")

    return best_map


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=10, catch=(ValueError, KeyError))

print("\nBest Hyperparameters Found:")
print(study.best_params)

best_params = study.best_params
final_model = YOLO("models/yolov8n-seg.pt")

final_model.train(
    data="datasets/yolo_fine_tune.yaml",
    epochs=best_params["epochs"],
    batch=best_params["batch"],
    imgsz=best_params["imgsz"],
    device="cuda",
    lr0=best_params["learning_rate"],
    name="final_trained_model"
)

print("\nYOLO Model successfully trained with optimized hyperparameters!")
