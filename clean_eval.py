import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent

ROOT = PROJECT / "Home_Fire_YOLO" / "data" / "home_fire"
CLEAN = PROJECT / "Home_Fire_YOLO" / "data" / "home_fire_clean" / "val"
CKPT = PROJECT / "Home_Fire_YOLO" / "runs" / "homefire_yolo11n_e50_sz640_b8_s42-2" / "weights" / "best.pt"
YAML = PROJECT / "Home_Fire_YOLO" / "configs" / "data_clean.yaml"


def ahash(path, size=16):
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return (a > a.mean()).flatten().astype(np.uint8)


def rgb_sig(path, size=32):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
tr_files = sorted((ROOT / "train" / "images").glob("*.jpg"))
tr_hash = np.array([ahash(f) for f in tr_files])

va_files = sorted((ROOT / "val" / "images").glob("*.jpg"))
print(f"val images: {len(va_files)}")

clean_imgs, dup_count = [], 0
for f in va_files:
    d = POP[np.bitwise_xor(tr_hash, ahash(f)[None, :])].sum(axis=1)
    cand = np.argsort(d)[:5]
    best = min(float(((rgb_sig(f) - rgb_sig(tr_files[j])) ** 2).mean()) for j in cand)
    if best < 50:
        dup_count += 1
    else:
        clean_imgs.append(f)

print(f"duplicates removed: {dup_count}")
print(f"clean val images:   {len(clean_imgs)}")

# materialise clean dataset
for sub in ("images", "labels"):
    (CLEAN / sub).mkdir(parents=True, exist_ok=True)
copied = 0
for f in clean_imgs:
    dst = CLEAN / "images" / f.name
    if not dst.exists():
        shutil.copy2(f, dst)
    lbl = ROOT / "val" / "labels" / (f.stem + ".txt")
    if lbl.exists():
        shutil.copy2(lbl, CLEAN / "labels" / lbl.name)
    copied += 1
print(f"copied {copied} clean images + labels -> {CLEAN}")

# Write a portable YAML: paths relative to this file's own directory (configs/),
# so nothing machine-specific (user names, absolute paths) ends up in the file.
YAML.write_text(
    "# Paths are relative to this file's own directory (Home_Fire_YOLO/configs/).\n"
    "# Duplicate-free validation split (built by clean_eval.py).\n"
    "train: ../data/home_fire/train/images\n"
    "val: ../data/home_fire_clean/val/images\n"
    "nc: 2\n"
    "names: ['fire', 'smoke']\n",
    encoding="utf-8",
)

print("\n=== evaluating on CLEAN val set ===")
model = YOLO(str(CKPT))
metrics = model.val(data=str(YAML), imgsz=640, verbose=True, save_json=False, plots=False)
print("\n--- TRUE METRICS (duplicate-free val) ---")
print(f"  precision = {metrics.box.mp:.4f}")
print(f"  recall    = {metrics.box.mr:.4f}")
print(f"  mAP50     = {metrics.box.map50:.4f}")
print(f"  mAP50-95  = {metrics.box.map:.4f}")
print(f"  images    = {metrics.files_seen}")
