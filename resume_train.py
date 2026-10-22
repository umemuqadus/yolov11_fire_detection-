from ultralytics import YOLO
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

LAST = PROJECT / "Home_Fire_YOLO" / "runs" / "homefire_yolo11n_e50_sz640_b8_s42-2" / "weights" / "last.pt"

model = YOLO(LAST)
model.train(
    resume=True,     
    epochs=50,      
)
print("TRAINING FINISHED")
