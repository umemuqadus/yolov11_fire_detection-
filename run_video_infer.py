import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent

if len(sys.argv) < 2:
    sys.exit("usage: python run_video_infer.py <input_video> [confidence]")

VIDEO = Path(sys.argv[1])
CKPT = PROJECT / "Home_Fire_YOLO" / "runs" / "homefire_yolo11n_e50_sz640_b8_s42-2" / "weights" / "best.pt"
OUT_DIR = PROJECT / "outputs"
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.62
IMGSZ = 640

if not VIDEO.exists():
    sys.exit(f"video not found: {VIDEO}")
if not CKPT.exists():
    sys.exit(f"checkpoint not found: {CKPT}")

model = YOLO(str(CKPT))
print(f"classes: {model.names}")

results = model.predict(
    source=str(VIDEO),
    conf=CONF,
    imgsz=IMGSZ,
    save=True,
    project=str(OUT_DIR),
    name="video_final",
    exist_ok=True,
    verbose=True,
)


total = 0
for r in results:
    n = len(r.boxes)
    total += n
    if n:
        cls = [r.names[int(b.cls)] for b in r.boxes]
        print(f"frame {r.path}: {n} box(es) -> {cls}")
print(f"TOTAL boxes across video: {total}")

out = OUT_DIR / "video_final" / VIDEO.name
print(f"annotated video: {out} (exists={out.exists()})")
