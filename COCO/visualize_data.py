import os
import json
import cv2
import numpy as np

img_height = 640
imgs_path = "data/images/train/"
annotations_path = "data/labels/train/"


def transform_bbox(b_box):
    x1 = int(round((b_box[0] - b_box[2]/2) * img_height, 0))
    x2 = int(round((b_box[0] + b_box[2]/2) * img_height, 0))

    y1 = int(round((b_box[1] - b_box[3]/2) * img_height, 0))
    y2 = int(round((b_box[1] + b_box[3]/2) * img_height, 0))

    return x1, y1, x2, y2


def gen_bboxes(image, annotation_path):

    with open(annotation_path) as f:
        for line in f:

            line = [float(x) for x in line.split(" ")]
            bbox = transform_bbox(line[1:])
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), np.random.uniform(0, 256, 3))


def visualize_data(data_path, nr_img=5):
    for img_name in np.random.choice(os.listdir(data_path), nr_img):
        img_path = imgs_path + img_name
        annotation_path = annotations_path + img_name[:-4] + ".txt"


        image = cv2.imread(img_path)
        gen_bboxes(image, annotation_path)
        cv2.imshow("test", image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_data(imgs_path)
