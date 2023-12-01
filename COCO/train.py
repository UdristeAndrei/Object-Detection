from ultralytics import YOLO

model = YOLO("yolov8m.yaml")

model.train(data="train_file.yaml", epochs=50, imgsz=160, batch=2, workers=0, name="exp_scratch")