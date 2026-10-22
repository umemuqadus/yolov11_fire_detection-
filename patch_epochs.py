import shutil
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent

RUN = PROJECT / "Home_Fire_YOLO" / "runs" / "homefire_yolo11n_e50_sz640_b8_s42-2"
LAST = RUN / "weights" / "last.pt"
BAK = RUN / "weights" / "backup_before_epoch_patch"
TARGET_EPOCHS = 15

BAK.mkdir(exist_ok=True)
for f in ("last.pt", "best.pt"):
    dst = BAK / f
    if not dst.exists():
        shutil.copy2(LAST if f == "last.pt" else RUN / "weights" / f, dst)
        print(f"backed up {f} -> {dst}")

ckpt = torch.load(LAST, map_location="cpu", weights_only=False)
print("keys:", sorted(ckpt.keys()))

args = ckpt.get("train_args") or ckpt.get("args")
if isinstance(args, dict):
    print("old epochs:", args.get("epochs"))
    args["epochs"] = TARGET_EPOCHS
else:
    print("old epochs:", getattr(args, "epochs", None))
    args.epochs = TARGET_EPOCHS
torch.save(ckpt, LAST)
print("patched -> epochs =", TARGET_EPOCHS)

# sanity check
chk = torch.load(LAST, map_location="cpu", weights_only=False)
a = chk.get("train_args") or chk.get("args")
ep = a.get("epochs") if isinstance(a, dict) else getattr(a, "epochs", None)
print("verify:", ep, "| has model:", "model" in chk, "| has ema:", "ema" in chk)
