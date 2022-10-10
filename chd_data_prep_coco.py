import sys
from glob import glob
from random import shuffle
import shutil
import os
import json
import argparse
from pprint import pprint

TRAINING_IMAGES = 500
TESTING_IMAGES = 100
VAL_IMAGES = 100

# orig categories names as key + new id as val
NEW_CLASSES = {'person': 0}

DATA_DIR = "C:\\Users\\davet\\OneDrive\\Desktop\\coco\\coco_orig"
"""
- annotations
- images
  - test2017
  - train2017
  - val2017  
- labels
  - test2017
  - train2017
  - val2017  
"""

RESULTS_DIR = "C:\\Users\\davet\\OneDrive\\Desktop\\coco\\coco_chd"
"""
- test
    - images
    - labels
- train
    - images
    - labels
- val
    - images
    - labels
"""
TOTAL_IMAGES = TRAINING_IMAGES + TESTING_IMAGES + VAL_IMAGES


def clear_results_dir(clear_dir):
    print(f"Deleting images, annotations and cache from results directory: {clear_dir}")
    images = glob(f"{clear_dir}\\*\\images\\*.jpg")
    for image_path in images:
        os.remove(image_path)
    labels = glob(f"{clear_dir}\\*\\labels\\*.txt")
    for label_path in labels:
        os.remove(label_path)
    caches = glob(f"{clear_dir}\\*\\*.cache")
    for cache_path in caches:
        os.remove(cache_path)


def parse_annotations(image):
    yolo_lines = []
    img_height = image['height']
    img_width = image['width']
    for annotation in [x for x in annotations if x["image_id"] == image["id"]]:
        object_type = next((c["name"] for c in categories if annotation['category_id'] == c["id"]), None)
        width = annotation['bbox'][2] / img_width
        height = annotation['bbox'][3] / img_height
        xcenter = (annotation['bbox'][0] / img_width) + (width/2)
        ycenter = (annotation['bbox'][1] / img_height) + (height/2)
        bndbox = f"{xcenter} {ycenter} {width} {height}"
        if object_type in NEW_CLASSES.keys():
            new_line = f"{NEW_CLASSES[object_type]} {bndbox}"
            yolo_lines.append(new_line)
    if len(yolo_lines) == 0:
        return False, None
    return True, "\n".join(yolo_lines)


def add_to_results_dir(image_src, new_text, subdirectory):
    image_dst = f"{RESULTS_DIR}\\{subdirectory}\\images\\{os.path.basename(image_src)}"
    shutil.copyfile(image_src, image_dst)

    labels_dst = image_dst.replace("images", "labels").replace(".jpg", ".txt")
    with open(labels_dst, "w") as labels_file:
        labels_file.write(new_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_clean', action='store_true', help="Don't clean results directory")
    parser.add_argument('--print-categories', action='store_true', help='Print object catigories')
    opt = parser.parse_args()
    if not opt.no_clean:
        clear_results_dir(RESULTS_DIR)

    with open(f"{DATA_DIR}\\annotations\\instances_train2017.json", "r") as stream:
        annotations = json.load(stream)
        categories = annotations["categories"]
        if opt.print_categories:
            pprint(categories)
        images = annotations["images"]
        shuffle(images)
        annotations = annotations["annotations"]
        print(f"{len(images)} images found")

    success_count = 0
    for image in images:
        image_path = f"{DATA_DIR}\\images\\train2017\\{image['file_name']}"
        success, new_text = parse_annotations(image)
        if success:
            success_count += 1
            if success_count <= TRAINING_IMAGES:
                add_to_results_dir(image_path, new_text, "train")
            elif success_count <= (TRAINING_IMAGES + TESTING_IMAGES):
                add_to_results_dir(image_path, new_text, "test")
            else:
                add_to_results_dir(image_path, new_text, "val")
        if success_count >= TOTAL_IMAGES:
            break

"""
python train.py --workers 8 --device 0 --batch-size 6 --data data/chd_2022.yaml --img 640 640 --cfg cfg/training/ yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml --epochs 50
"""