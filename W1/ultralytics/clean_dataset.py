import os

# Path to the main directory containing the 21 sequence folders
main_dir = "../KITTI-MOTS/yolo_format/dataset/images"

# Loop over each sequence folder
for i in range(21):
    seq_folder = os.path.join(main_dir, f"{i:04d}")  # Format folder name (0000, 0001, ..., 0020)
    gt_file = os.path.join(main_dir, f"gt_bboxes_{i:04d}.txt")  # Path to ground truth file

    if not os.path.exists(gt_file):
        print(f"Skipping {seq_folder}, {gt_file} not found.")
        continue

    # Read the gt file and collect the image names that have GT information
    gt_images = set()
    with open(gt_file, "r") as f:
        for line in f:
            gt_images.add(line.split()[0])  # Extract the filename (first column)

    # Loop through all images in the folder
    for img_file in os.listdir(seq_folder):
        if img_file.endswith(".png") and img_file not in gt_images:
            img_path = os.path.join(seq_folder, img_file)
            os.remove(img_path)  # Delete the image
            print(f"Deleted: {img_path}")

print("Cleanup complete!")
