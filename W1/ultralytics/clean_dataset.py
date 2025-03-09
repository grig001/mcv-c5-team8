import os


main_dir = "../KITTI-MOTS/yolo_format/dataset/images"


for i in range(21):
    seq_folder = os.path.join(main_dir, f"{i:04d}") 
    gt_file = os.path.join(main_dir, f"gt_bboxes_{i:04d}.txt")  

    if not os.path.exists(gt_file):
        print(f"Skipping {seq_folder}, {gt_file} not found.")
        continue
  
    gt_images = set()
    with open(gt_file, "r") as f:
        for line in f:
            gt_images.add(line.split()[0])  # Extract the filename (first column)

    for img_file in os.listdir(seq_folder):
        if img_file.endswith(".png") and img_file not in gt_images:
            img_path = os.path.join(seq_folder, img_file)
            os.remove(img_path)  # Delete the image
            print(f"Deleted: {img_path}")

print("Cleanup complete!")
