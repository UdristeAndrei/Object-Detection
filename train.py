from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(data="train_file.yaml", epochs=60, imgsz=352, batch=3, workers=0, name="exp_v8", dropout=0.1)
