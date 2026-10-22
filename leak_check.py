from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent

ROOT = PROJECT / "Home_Fire_YOLO" / "data" / "home_fire"


def ahash(path, size=16):
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return (a > a.mean()).flatten().astype(np.uint8)


def load(split):
    d = ROOT / split / "images"
    files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
    H, names = [], []
    for f in files:
        try:
            H.append(ahash(f))
            names.append(f.name)
        except Exception:
            pass
    return np.array(H), names


print("loading train images...")
tr, tr_names = load("train")
print("loading val images...")
va, va_names = load("val")
print(f"train={len(tr)}  val={len(va)}")

# Hamming distance via popcount LUT on XOR
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def min_dist(v, t):
    best = np.full(len(v), 999, dtype=np.int32)
    for i in range(0, len(t), 512):
        chunk = t[i:i + 512]
        xor = np.bitwise_xor(v[:, None, :], chunk[None, :, :])
        dist = POP[xor].sum(axis=2)
        best = np.minimum(best, dist.min(axis=1))
    return best

d = min_dist(va, tr)
total = len(d)
for thr, label in ((0, "identical"), (4, "near-identical (<=4 bits)"), (8, "very similar (<=8 bits)"), (16, "similar (<=16 bits)")):
    c = int((d <= thr).sum())
    print(f"  val images {label}: {c}/{total} ({100*c/total:.1f}%)")

print(f"\nmean min distance: {d.mean():.1f} / 256 bits")
print("\nclosest val->train matches (most likely duplicates):")
order = np.argsort(d)[:8]
for i in order:
    xor = np.bitwise_xor(tr, va[i][None, :])
    j = int(np.argmin(xor.sum(axis=1)))
    print(f"  {va_names[i]}  dist={d[i]}  <->  {tr_names[j]}")
