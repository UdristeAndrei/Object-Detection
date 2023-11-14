import os
import json
import shutil
import numpy as np

file_path = "vehicle_damage_detection_dataset/vehicle_damage_detection_dataset/"
save_path = "data/"


def gen_labels(set_name, img_name, b_box, dmg_type):
    labels_path = save_path + f"labels/{set_name}/"
    os.makedirs(labels_path, exist_ok=True)

    transformed_bbox = [b_box[0] + b_box[2] / 2, b_box[1] + b_box[3] / 2, b_box[2], b_box[3]]
    str_bbox = " ".join(str(value) for value in transformed_bbox)

    with open(labels_path + str(img_name) + ".txt", "a+") as labels_f:
        labels_f.write(str(dmg_type) + " " + str_bbox + "\n")


def move_data(data_path, set_name="train"):
    img_path = save_path + f"images/{set_name}/"
    os.makedirs(img_path, exist_ok=True)

    with open(data_path + f"annotations/instances_{set_name}.json") as annotations_f:
        data = json.load(annotations_f)

        for idx in range(len(data["annotations"])):
            img_id = data["annotations"][idx]["image_id"]
            damage_type = data["annotations"][idx]["category_id"]

            b_box = np.array(data["annotations"][idx]["bbox"]) / data["images"][img_id]["height"]
            img_name = data["images"][img_id]["file_name"]

            shutil.copy(data_path + f"images/{set_name}/" + img_name, img_path + str(img_id) + ".jpg")
            gen_labels(set_name, img_id, b_box, damage_type)


move_data(file_path)
