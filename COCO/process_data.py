import json
import os
import cv2
import shutil
import numpy as np

file_path = "data/data2017/"


def normalize_bbox(bbox, img_height=640, img_width=640):
    bbox[0] = bbox[0] / img_width
    bbox[1] = bbox[1] / img_height
    bbox[2] = bbox[2] / img_width
    bbox[3] = bbox[3] / img_height
    return bbox


def convert_coco91_coco80(category_id):
    missing_values = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]

    if category_id < 12:
        return category_id - 1

    for idx, missing_value in enumerate(missing_values):
        if category_id < missing_value:
            return category_id - idx - 1


def generate_labels(save_path, annotation_path, img_id, h, w):
    with open(annotation_path) as f_json:
        annotations = json.load(f_json)

        with open(save_path + "/" + img_id + ".txt", "w") as f_txt:
            for annotation in annotations:

                class_id = convert_coco91_coco80(annotation["category_id"])
                bbox = normalize_bbox(np.array(annotation["bbox"]), h, w)

                transformed_bbox = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]]
                str_bbox = " ".join(str(value) for value in transformed_bbox)

                f_txt.write(str(class_id) + " " + str_bbox + "\n")


def create_dir(annotation_path, img_id, set_name="train"):
    os.makedirs("data/images/" + set_name, exist_ok=True)
    os.makedirs("data/labels/" + set_name, exist_ok=True)

    img_name = img_id + ".jpg"
    image = cv2.imread(file_path + img_name)
    h, w, c = image.shape

    shutil.copy(file_path + img_name, "data/images/" + set_name + "/" + img_name)
    generate_labels("data/labels/" + set_name, annotation_path, img_id, h, w)


def data_split(path, train_proc=0.7, val_proc=0.2):
    files = [file for file in os.listdir(path) if file[-5:] == ".json"]

    train_split = int(len(files) * train_proc)
    val_split = train_split + int(len(files) * val_proc)

    for file in files[:train_split]:
        create_dir(path + file, file[:-5], "train")

    for file in files[train_split:val_split]:
        create_dir(path + file, file[:-5], "val")

    for file in files[val_split:]:
        create_dir(path + file, file[:-5], "test")


if __name__ == "__main__":
    data_split(file_path)
