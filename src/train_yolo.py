import os
from ultralytics import YOLO

# Fix OpenMP Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLOv8 model
model = YOLO("yolov8n.yaml")

# Train YOLO model
model.train(data="D:/Canada/Subjects/Semester -1/AIDI 1003_01_CAPSTONE TERM 1/Cosmic navigators/dataset/data.yaml", epochs=50, imgsz=640)

# Save trained model
model.export(format="onnx")
