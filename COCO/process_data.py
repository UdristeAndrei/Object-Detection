import json
import os
import shutil
import numpy as np

file_path = "data/data2017/"
img_height = 640


def generate_labels(save_path, annotation_path, img_id):
    with open(annotation_path) as f_json:
        annotations = json.load(f_json)

        with open(save_path + "/" + img_id + ".txt", "w") as f_txt:
            for annotation in annotations:

                bbox = np.array(annotation["bbox"]) / img_height
                transformed_bbox = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]]
                str_bbox = " ".join(str(value) for value in transformed_bbox)

                f_txt.write(str(annotation["category_id"]) + " " + str_bbox + "\n")


def create_dir(annotation_path, img_id, set_name="train"):
    os.makedirs("data/images/" + set_name, exist_ok=True)
    os.makedirs("data/labels/" + set_name, exist_ok=True)

    generate_labels("data/labels/" + set_name, annotation_path, img_id)

    img_name = img_id + ".jpg"
    shutil.copy(file_path + img_name, "data/images/" + set_name + "/" + img_name)


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


if __name__=="__main__":
    data_split(file_path)
