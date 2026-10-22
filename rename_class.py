import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent

RUN = PROJECT / "Home_Fire_YOLO" / "runs" / "homefire_yolo11n_e50_sz640_b8_s42-2"
BEST = RUN / "weights" / "best.pt"
BAK = RUN / "weights" / "best_with_flame_name.pt"
OLD, NEW = "flame", "fire"

# 1. backup (only once - keep the original-name version)
if not BAK.exists():
    shutil.copy2(BEST, BAK)
    print(f"backup -> {BAK}")

# 2. patch names inside the checkpoint
ckpt = torch.load(BEST, map_location="cpu", weights_only=False)
changed = []
for key in ("model", "ema"):
    obj = ckpt.get(key)
    names = getattr(obj, "names", None) if obj is not None else None
    if isinstance(names, dict) and OLD in names.values():
        idx = list(names.keys())[list(names.values()).index(OLD)]
        names[idx] = NEW
        changed.append(f"{key}.names[{idx}]")
print("patched:", changed or "nothing to do")

if changed:
    torch.save(ckpt, BEST)
    print("checkpoint saved")

# 3. configs
for yaml_path in (
    PROJECT / "Home_Fire_YOLO" / "configs" / "data.yaml",
    PROJECT / "Home_Fire_YOLO" / "configs" / "data_clean.yaml",
):
    if not yaml_path.exists():
        continue
    txt = yaml_path.read_text(encoding="utf-8")
    new_txt = txt.replace(f"0: {OLD}", f"0: {NEW}").replace(f"'{OLD}'", f"'{NEW}'")
    if new_txt != txt:
        yaml_path.write_text(new_txt, encoding="utf-8")
        print(f"updated {yaml_path.name}")

# 4. verify by actually loading the model
model = YOLO(str(BEST))
print("LOADED model.names ->", model.names)
assert model.names[0] == NEW, "rename failed"
print("OK: class 0 is now '" + NEW + "'")
