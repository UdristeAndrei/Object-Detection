from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="train_file.yaml", epochs=120, imgsz=352, batch=3, workers=0, name="exp_v8n-120", dropout=0.3)
