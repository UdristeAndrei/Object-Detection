import cv2
import os
import numpy as np
from ultralytics import YOLO

img_height = img_width = 640

data_path = "data/images/test/"
labels_path = "data/labels/test/"
model = YOLO("runs/detect/exp_v5su/weights/best.pt")
save_path = "runs/inferance/yolov5su/"

os.makedirs(save_path, exist_ok=True)

class_total = np.zeros(8)
class_correct = np.zeros(8)
iou_values = np.zeros(8)


def transform_bbox(b_box):
    x1 = int(round((b_box[0] - b_box[2]/2) * img_height, 0))
    x2 = int(round((b_box[0] + b_box[2]/2) * img_height, 0))

    y1 = int(round((b_box[1] - b_box[3]/2) * img_height, 0))
    y2 = int(round((b_box[1] + b_box[3]/2) * img_height, 0))

    return x1, y1, x2, y2


def get_labels(path, img_name):
    label_path = path + img_name[:-3] + "txt"
    label_classes = list()
    label_bbox = list()

    with open(label_path) as f:
        for line in f:
            line = [float(x) for x in line.split(" ")]

            label_classes.append(line[0])
            label_bbox.append(transform_bbox(line[1:]))

    return label_classes, label_bbox


def draw_rect(img, bboxes_pred, bboxes_labels, pred_classes, conf_classes, label_classes):

    for idx, b_box in enumerate(bboxes_labels):
        bbox_label = cv2.rectangle(img, (b_box[0], b_box[1]), (b_box[2], b_box[3]), (0, 255, 0))
        # Can put the text out of bounds, need fix
        cv2.putText(bbox_label, str(label_classes[idx]), (b_box[0], b_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1)

    for idx, b_box in enumerate(bboxes_pred):
        bbox_predict = cv2.rectangle(img, (b_box[0], b_box[1]), (b_box[2], b_box[3]), (255, 0, 0))
        # Weird interaction if you remove the str(), shows all the decimals
        cv2.putText(bbox_predict, f"{pred_classes[idx]}: {str(conf_classes[idx])}%", (b_box[0] + 5, b_box[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)


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


def draw_iou(img, iou_value, inter_coord, iou_trsh_save=0.8):
    bbox_predict = cv2.rectangle(img, (inter_coord[0], inter_coord[1]), (inter_coord[2], inter_coord[3]),
                                 (0, 0, 255))
    # Can put the text out of bounds, need fix
    cv2.putText(bbox_predict, f"iou: {round(iou_value, 2)}", (inter_coord[2], inter_coord[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    if iou_value >= iou_trsh_save:
        cv2.imwrite(save_path + f"{iou_value}.jpg", bbox_predict)


def determine_statistics(img, pred_bboxes, label_bboxes, pred_classes, label_classes, iou_trsh=0.3):

    for l_idx, label_box in enumerate(label_bboxes):
        class_total[int(label_classes[l_idx] - 1)] += 1

        for p_idx, pred_box in enumerate(pred_bboxes):

            iou_value, inter_coord = compute_iou(pred_box, label_box)

            if iou_value > iou_trsh:

                if pred_classes[p_idx] == label_classes[l_idx]:
                    iou_values[int(pred_classes[p_idx] - 1)] += iou_value
                    class_correct[int(pred_classes[p_idx] - 1)] += 1

                draw_iou(img, iou_value, inter_coord)


def save_data():
    classes = [str(idx + 1) for idx in np.arange(8)]
    acc = [str(round(a, 2)) for a in np.nan_to_num(class_correct / class_total)]
    avg_iou = [str(round(a, 2)) for a in np.nan_to_num(iou_values / class_correct)]

    with open(save_path + "statistics.txt", "w") as f:
        f.write("class   " + "    ". join(classes) + "\n")
        f.write("acc     " + "  ".join(acc) + "\n")
        f.write("avg_iou " + "  ".join(avg_iou) + "\n")


def main():
    for img_name in os.listdir(data_path):

        results = model.predict(data_path + img_name)
        label_classes, label_bboxes = get_labels(labels_path, img_name)

        for result in results:
            pred_classes = result.boxes.cls.detach().cpu().numpy()

            if pred_classes.any():
                pred_bboxes = np.round(result.boxes.xyxy.detach().cpu().numpy()).astype(int)
                conf_classes = np.round(result.boxes.conf.detach().cpu().numpy() * 100, decimals=2)
                image = result.orig_img

                draw_rect(image, pred_bboxes, label_bboxes, pred_classes, conf_classes, label_classes)
                determine_statistics(image, pred_bboxes, label_bboxes, pred_classes, label_classes)

                cv2.imshow("test", image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

    save_data()


if __name__ == "__main__":
    main()
