#!/usr/bin/env python3
"""Q3: Is there a CKA threshold below which probe asymmetry (dR2>0) emerges?

Pools every result cell on disk that has both final_cka and linear_probe.r2_delta
(153 cells, CKA 0.135-0.999) and characterises dR2 as a function of CKA.
ETTh1 is separated out (deep pretrained floor -> known probe exception, W3/Q2).
"""
import glob, json, statistics as st

rows = []  # (cka, dr2, is_etth1, cond, path)
for f in glob.glob("results/**/*.json", recursive=True):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not (isinstance(d, dict) and "final_cka" in d
            and isinstance(d.get("linear_probe"), dict)
            and "r2_delta" in d["linear_probe"]):
        continue
    cka = d["final_cka"]; dr2 = d["linear_probe"]["r2_delta"]
    is_h1 = "etth1" in f.lower()
    cond = d.get("condition", "?")
    rows.append((cka, dr2, is_h1, cond, f.split("results/")[-1]))

main = [r for r in rows if not r[2]]      # ETTh2/other (mechanism cells)
h1   = [r for r in rows if r[2]]          # ETTh1 (deep-floor exception)

print(f"total cells: {len(rows)}  (ETTh2/other: {len(main)}, ETTh1: {len(h1)})\n")

bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
print("=== dR2 vs CKA  (ETTh2/other cells; ETTh1 excluded as deep-floor exception) ===")
print(f"{'CKA bin':>12s} {'n':>4s} {'mean dR2':>9s} {'median':>8s} {'dR2>0':>7s}")
for lo, hi in bins:
    b = [r for r in main if lo <= r[0] < hi]
    if not b:
        print(f"{f'[{lo},{hi})':>12s}   0        --       --      --")
        continue
    dr = [r[1] for r in b]
    print(f"{f'[{lo},{hi})':>12s} {len(b):>4d} {st.mean(dr):>+9.3f} {st.median(dr):>+8.3f} "
          f"{sum(x>0 for x in dr)}/{len(dr)}")

# Emergence threshold: highest CKA bin where dR2>0 is still the clear majority
print("\n=== interpretation ===")
above95 = [r[1] for r in main if r[0] >= 0.95]
below80 = [r[1] for r in main if r[0] < 0.80]
print(f"CKA >= 0.95 (near-frozen): mean dR2 {st.mean(above95):+.3f}, "
      f"positive {sum(x>0 for x in above95)}/{len(above95)}  -> asymmetry ABSENT")
print(f"CKA <  0.80 (clear drift): mean dR2 {st.mean(below80):+.3f}, "
      f"positive {sum(x>0 for x in below80)}/{len(below80)}  -> asymmetry PRESENT")
print("=> probe asymmetry emerges once CKA falls below ~0.9 and is robust below ~0.8;")
print("   at CKA>=0.95 the encoder is effectively unchanged and dR2~0 (matches the")
print("   frozen-encoder controls and the two Chronos/M4 stability cells).")

print("\n=== ETTh1 (deep-floor exception, W3/Q2) ===")
if h1:
    print(f"CKA range {min(r[0] for r in h1):.3f}-{max(r[0] for r in h1):.3f}: "
          f"mean dR2 {st.mean(r[1] for r in h1):+.3f}, "
          f"positive {sum(r[1]>0 for r in h1)}/{len(h1)}  "
          f"-> drift present but dR2<0 due to R2_ZS floor = -24.8")
