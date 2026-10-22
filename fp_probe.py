from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent

BASE = PROJECT / "Home_Fire_YOLO" / "data" / "home_fire" / "val"
CKPT = PROJECT / "Home_Fire_YOLO" / "runs" / "homefire_yolo11n_e50_sz640_b8_s42-2" / "weights" / "best.pt"
CONF = 0.45

empty = [p for p in (BASE / "labels").glob("*.txt") if p.stat().st_size == 0]
print(f"true-negative images (no fire/smoke): {len(empty)}")

model = YOLO(str(CKPT))

total_fp, frames_with_fp, max_conf = 0, 0, 0.0
hits = []
for lbl in sorted(empty):
    img = BASE / "images" / (lbl.stem + ".jpg")
    if not img.exists():
        continue
    r = model.predict(source=str(img), conf=CONF, imgsz=640, verbose=False)[0]
    n = len(r.boxes)
    if n:
        frames_with_fp += 1
        total_fp += n
        c = max(float(b.conf) for b in r.boxes)
        max_conf = max(max_conf, c)
        hits.append((img.name, n, [r.names[int(b.cls)] for b in r.boxes], round(c, 3)))

print(f"\nimages tested: {len(empty)}")
print(f"images with FALSE POSITIVES: {frames_with_fp}  ({frames_with_fp}/{len(empty)})")
print(f"total false-positive boxes: {total_fp}")
print(f"highest false-positive confidence: {max_conf:.3f}  (threshold is {CONF})")
for h in hits:
    print(f"  {h[0]}: {h[1]} box(es) {h[2]} maxconf={h[3]}")
if not hits:
    print("  none - clean on non-fire scenes")
