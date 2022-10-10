from glob import glob
from random import shuffle
import shutil
import os

TRAINING_IMAGES = 100
TESTING_IMAGES = 20
VAL_IMAGES = 10
NEW_CLASSES = {'person': 0, 'bicycle': 1}
ORIG_CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush' ]

DATA_DIR = "C:\\Users\\davet\\OneDrive\\Desktop\\coco\\coco_orig"
"""
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


def parse_labels(file_path):
    filtered_lines = []
    with open(file_path, "r") as stream:
        lines = stream.readlines()
        for index, line in enumerate(lines):
            orig_class_index = int(line.split(" ", 1)[0])
            object_location = line.split(" ", 1)[1]
            object_type = ORIG_CLASS_NAMES[orig_class_index]
            if object_type in NEW_CLASSES.keys():
                new_line = f"{NEW_CLASSES[object_type]} {object_location}"
                filtered_lines.append(new_line)
    if len(filtered_lines) == 0:
        return False, None
    return True, "".join(filtered_lines)


def add_to_results_dir(labels_src, new_text, subdirectory):
    labels_dst = f"{RESULTS_DIR}\\{subdirectory}\\labels\\{os.path.basename(labels_src)}"
    with open(labels_dst, "w") as labels_file:
        labels_file.write(new_text)

    image_src = labels_src.replace("labels", "images").replace(".txt", ".jpg")
    image_dst = f"{RESULTS_DIR}\\{subdirectory}\\images\\{os.path.basename(image_src)}"
    shutil.copyfile(image_src, image_dst)


def clear_results_dir():
    images = glob(f"{RESULTS_DIR}\\*\\images\\*.jpg")
    print(f"{RESULTS_DIR}\\*\\images\\.jpg")
    for image_path in images:
        os.remove(image_path)
    labels = glob(f"{RESULTS_DIR}\\*\\labels\\*.txt")
    for label_path in labels:
        os.remove(label_path)


if __name__ == '__main__':
    clear_results_dir()
    all_label_files = glob(f"{DATA_DIR}\\labels\\*\\*.txt")
    shuffle(all_label_files)
    success_count = 0
    for label_path in all_label_files:
        success, new_text = parse_labels(label_path)
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
python train.py --workers 8 --device 0 --batch-size 6 --data data/chd_2022.yaml --img 640 640 --cfg cfg/training/ yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml --epochs 50
"""