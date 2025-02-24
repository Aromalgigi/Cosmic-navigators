import os
from ultralytics import YOLO

# Fix OpenMP Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load trained YOLOv8 model
model = YOLO("runs/detect/train7/weights/best.pt")

# Validate the model
results = model.val(data="dataset/data.yaml", imgsz=640)

# Print validation results
print(results)
