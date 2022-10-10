import yaml
import shutil
import os
from glob import glob
from random import shuffle

TRAINING_IMAGES = 2000
VAL_IMAGES = 400

# USED @ END to Check vs. "validation" while training
TESTING_IMAGES = 400

CLASS_NAMES = ['person', 'crutches', 'walking_frame', 'wheelchair', 'push_wheelchair']

DATA_DIR = "C:\\Users\\davet\\OneDrive\\Desktop\\fmad\\fmad_orig"
"""
- images
- labels
"""

RESULTS_DIR = "C:\\Users\\davet\\OneDrive\\Desktop\\fmad\\fmad_chd"
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


def parse_annotations(file_path):
    yolo_lines = []
    with open(file_path, "r") as stream:
        try:
            results = yaml.safe_load(stream)
            object_list = results['annotation']['object']
            img_height = int(results['annotation']['size']['height'])
            img_width = int(results['annotation']['size']['width'])
        except:
            print("ERROR. Skipping")
            return False, None
        for obj in object_list:
            bndbox = obj
            xmin = float(bndbox['bndbox']['xmin'])
            xmax = float(bndbox['bndbox']['xmax'])
            ymin = float(bndbox['bndbox']['ymin'])
            ymax = float(bndbox['bndbox']['ymax'])
            width = (xmax - xmin)/img_width
            height = (ymax - ymin)/img_height
            xcenter = ((xmax + xmin)/2)/img_width
            ycenter = ((ymax + ymin)/2)/img_height
            bndbox = f"{xcenter} {ycenter} {width} {height}"
            new_line = f"{CLASS_NAMES.index(obj['name'])} {bndbox}"
            yolo_lines.append(new_line)
        return True, "\n".join(yolo_lines)


def add_to_results_dir(labels_src, new_text, subdirectory):
    labels_dst = f"{RESULTS_DIR}\\{subdirectory}\\labels\\{os.path.basename(labels_src)}".replace(".yml", ".txt")
    with open(labels_dst, "w") as labels_file:
        labels_file.write(new_text)

    image_src = labels_src.replace("labels", "images").replace(".yml", ".png")
    image_dst = f"{RESULTS_DIR}\\{subdirectory}\\images\\{os.path.basename(image_src)}"
    shutil.copyfile(image_src, image_dst)


def clear_results_dir(clear_dir):
    print(f"Deleting images, annotations and cache from results directory: {clear_dir}")
    images = glob(f"{clear_dir}\\*\\images\\*.png")
    for image_path in images:
        os.remove(image_path)
    labels = glob(f"{clear_dir}\\*\\labels\\*.txt")
    for label_path in labels:
        os.remove(label_path)
    caches = glob(f"{clear_dir}\\*\\*.cache")
    for cache_path in caches:
        os.remove(cache_path)


if __name__ == '__main__':
    clear_results_dir(RESULTS_DIR)
    all_label_files = glob(f"{DATA_DIR}\\labels\\*.yml")
    shuffle(all_label_files)
    success_count = 0
    for label_path in all_label_files:
        success, new_text = parse_annotations(label_path)
        if success:
            success_count += 1
            if success_count <= TRAINING_IMAGES:
                add_to_results_dir(label_path, new_text, "train")
            elif success_count <= (TRAINING_IMAGES + TESTING_IMAGES):
                add_to_results_dir(label_path, new_text, "test")
            else:
                add_to_results_dir(label_path, new_text, "val")
        if success_count >= TOTAL_IMAGES:
            break

"""
python train.py --workers 8 --device 0 --batch-size 6 --data data/chd_fmad_2022.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml --epochs 50 --rect
"""