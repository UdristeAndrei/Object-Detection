import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# model = YOLO("yolov8m.pt")
model = YOLO("runs/detect/exp_scratch/weights/best.pt")
imgs_path = "data/images/test/"
labls_path = "data/labels/test/"
nr_images = -1
iou_save_trsh = 0.98

class_total = np.zeros(80)
total_iou = np.zeros(80)
pred_total = np.zeros(80)

save_path = "results/trained_scratch/"
os.makedirs(save_path, exist_ok=True)


def transform_bbox(b_box, h, w):
    x1 = int(round((b_box[0] - b_box[2]/2) * w, 0))
    x2 = int(round((b_box[0] + b_box[2]/2) * w, 0))

    y1 = int(round((b_box[1] - b_box[3]/2) * h, 0))
    y2 = int(round((b_box[1] + b_box[3]/2) * h, 0))

    return x1, y1, x2, y2


def get_labels(image, annotation_path):
    h, w, c = image.shape

    bboxes = list()
    classes = list()

    with open(annotation_path) as f:
        for line in f:

            line = [float(x) for x in line.split(" ")]
            bbox = transform_bbox(line[1:], h, w)

            classes.append(line[0])
            bboxes.append(bbox)

    return classes, bboxes


def draw_bbox(image, bboxes, rgb):
    for bbox in bboxes:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), rgb)


def compute_iou(pred_bbox, label_bbox):
    x1 = max(pred_bbox[0], label_bbox[0])
    y1 = max(pred_bbox[1], label_bbox[1])
    x2 = min(pred_bbox[2], label_bbox[2])
    y2 = min(pred_bbox[3], label_bbox[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    label_area = (label_bbox[2] - label_bbox[0]) * (label_bbox[3] - label_bbox[1])

    iou = inter_area / (pred_area + label_area - inter_area)
    return iou, (x1, y1, x2, y2)


def compute_statistics(class_labels, pred_classes, bboxes_label, pred_bboxes, iou_trsh=0.3):
    iou_coordinates = list()
    iou_values = list()

    for l_idx, bbox_label in enumerate(bboxes_label):
        for p_idx, bbox_pred in enumerate(pred_bboxes):

            iou_val, iou_coord = compute_iou(bbox_pred, bbox_label)

            if iou_val > iou_trsh and (class_labels[l_idx] == pred_classes[p_idx]):

                total_iou[int(class_labels[l_idx])] += iou_val
                pred_total[int(class_labels[l_idx])] += 1
                iou_coordinates.append(iou_coord)
                iou_values.append(iou_val)

    return iou_coordinates, iou_values


def save_statistics():
    classes = [idx + 1 for idx in np.arange(80)]
    acc = [round(a, 2) for a in np.nan_to_num(pred_total / class_total)]
    avg_iou = [round(a, 2) for a in np.nan_to_num(total_iou / pred_total)]

    statistics = pd.DataFrame()
    statistics["classes"] = classes
    statistics["accuracy"] = acc
    statistics["average_iou"] = avg_iou
    statistics["correct_prediction"] = pred_total
    statistics["total_label"] = class_total

    global_acc = np.average(acc, weights=class_total / sum(class_total))
    print("The accuracy of the model is:", global_acc)
    statistics.to_csv(save_path + "statistics.csv")


def main(show_images=True, save_images=True):
    for img_name in os.listdir(imgs_path)[:nr_images]:
        results = model.predict(imgs_path + img_name)
        image = cv2.imread(imgs_path + img_name)
        class_labels, bboxes_label = get_labels(image, labls_path + img_name[:-4] + ".txt")

        for class_label in class_labels:
            class_total[int(class_label)] += 1

        if show_images or save_images:
            draw_bbox(image, bboxes_label, (255, 0, 0))

        for result in results:
            pred_classes = result.boxes.cls.detach().cpu().numpy()

            if pred_classes.any():
                pred_bboxes = np.round(result.boxes.xyxy.detach().cpu().numpy()).astype(int)

                iou_coordinates, iou_values = compute_statistics(class_labels, pred_classes, bboxes_label, pred_bboxes)

                if show_images or save_images:
                    draw_bbox(image, pred_bboxes, (0, 255, 0))
                    draw_bbox(image, iou_coordinates, (0, 0, 255))

                if any(iou_val > iou_save_trsh for iou_val in iou_values):
                    cv2.imwrite(save_path + img_name, image)

        if show_images:
            cv2.imshow("test", image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    save_statistics()


if __name__ == "__main__":
    main(False, True)
