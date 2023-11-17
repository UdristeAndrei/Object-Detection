import json
import cv2
import numpy as np

file_path = "vehicle_damage_detection_dataset/vehicle_damage_detection_dataset/"


def visualize_data(data_path, set_name="train", nr_images=5):
    with open(data_path + f"annotations/instances_{set_name}.json") as f:
        data = json.load(f)

        for idx in range(len(data["annotations"]))[:nr_images]:
            img_id = data["annotations"][idx]["image_id"]
            b_box = np.round(data["annotations"][idx]["bbox"]).astype(int)
            img_path = data_path + f"images/{set_name}/" + data["images"][img_id]["file_name"]

            image = cv2.imread(img_path)

            cv2.rectangle(image, (b_box[0], b_box[1]), (b_box[0] + b_box[2], b_box[1] + b_box[3]), (255, 0, 0))
            cv2.imshow("test", image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_data(file_path)
