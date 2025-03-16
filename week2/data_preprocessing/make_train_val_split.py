#!/usr/bin/env python3
import os
import shutil
import argparse


def create_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")


def copy_files(src, dest):
    if os.path.exists(src):
        shutil.copytree(src, dest, dirs_exist_ok=True)
        print(f"Copied from {src} → {dest}")
    else:
        print(f"Warning: Source path not found: {src}")


def main(kitti_path):
    """Setup KITTI-MOTS dataset structure."""
    home_dir = os.path.expanduser("~")
    dest_base = os.path.join(home_dir, "team8_split_KITTI-MOTS")

    # Create directory structure
    create_directory(dest_base)
    create_directory(os.path.join(dest_base, "train"))
    create_directory(os.path.join(dest_base, "eval"))
    create_directory(os.path.join(dest_base, "instances_txt"))
    create_directory(os.path.join(dest_base, "instances_txt", "train"))
    create_directory(os.path.join(dest_base, "instances_txt", "eval"))

    # Copy instances_txt for train and eval
    copy_files(os.path.join(kitti_path, "instances_txt"), os.path.join(dest_base, "instances_txt", "train"))
    copy_files(os.path.join(kitti_path, "instances_txt"), os.path.join(dest_base, "instances_txt", "eval"))

    # Copy training images (0000-0017)
    for i in range(18):
        seq = f"{i:04d}"
        copy_files(os.path.join(kitti_path, "training/image_02", seq), os.path.join(dest_base, "train", seq))

    # Copy evaluation images (0018-0020)
    for i in range(18, 21):
        seq = f"{i:04d}"
        copy_files(os.path.join(kitti_path, "training/image_02", seq), os.path.join(dest_base, "eval", seq))

    print("\nKITTI-MOTS dataset setup completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup KITTI-MOTS dataset for YOLO segmentation.")
    parser.add_argument("--kitti_path", type=str, help="Path to KITTI-MOTS dataset.",
                        default="/ghome/c5mcv08/mcv/datasets/C5/KITTI-MOTS/")
    args = parser.parse_args()
    main(args.kitti_path)
