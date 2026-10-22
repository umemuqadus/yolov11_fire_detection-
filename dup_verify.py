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
    files = sorted((ROOT / split / "images").glob("*.jpg"))
    H, N = [], []
    for f in files:
        try:
            H.append(ahash(f))
            N.append(f)
        except Exception:
            pass
    return np.array(H), N


def rgb_sig(path, size=32):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
tr, trn = load("train")
va, van = load("val")

# find ahash matches (dist 0) and verify with RGB MSE
confirmed, collisions, checked = 0, 0, 0
worst = []
for i, v in enumerate(va):
    xor = np.bitwise_xor(tr, v[None, :])
    d = POP[xor].sum(axis=1)
    if d.min() != 0:
        continue
    j = int(np.argmin(d))
    checked += 1
    a = rgb_sig(van[i])
    b = rgb_sig(trn[j])
    mse = float(((a - b) ** 2).mean())
    if mse < 50:          # effectively the same picture
        confirmed += 1
    else:
        collisions += 1
        worst.append((van[i].name, trn[j].name, round(mse, 1)))

print(f"ahash-identical pairs checked: {checked}")
print(f"  TRUE duplicates (RGB MSE<50):   {confirmed}")
print(f"  hash collisions (different img): {collisions}")
for w in sorted(worst, key=lambda x: -x[2])[:5]:
    print(f"    collision: {w[0]} vs {w[1]} MSE={w[2]}")

# overall near-duplicate rate using RGB MSE against best ahash candidate
real_dup = 0
sample_idx = np.linspace(0, len(va) - 1, min(300, len(va))).astype(int)
for i in sample_idx:
    xor = np.bitwise_xor(tr, va[i][None, :])
    d = POP[xor].sum(axis=1)
    order = np.argsort(d)[:5]
    best = min(float(((rgb_sig(van[i]) - rgb_sig(trn[j])) ** 2).mean()) for j in order)
    if best < 50:
        real_dup += 1
n = len(sample_idx)
print(f"\nsample of {n} val images: {real_dup} truly duplicated in train ({100*real_dup/n:.1f}%)")
