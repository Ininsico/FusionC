import sys
from pathlib import Path
import numpy as np

work = Path("results/attention_mlp")
got = np.fromfile(work / "dump" / "out.attn_pv.bin", dtype=np.float16).astype(np.float64).reshape(64, 256)
print("total NaNs:", np.isnan(got).sum(), "of", got.size)
rows = np.where(np.isnan(got).any(axis=1))[0]
cols = np.where(np.isnan(got).any(axis=0))[0]
print("rows with NaN:", rows[:20], "count", len(rows))
print("cols with NaN:", cols[:40], "count", len(cols))
finite = np.nan_to_num(got, nan=0.0)
print("row 0 first 12:", np.round(got[0, :12], 5))
print("row 1 first 12:", np.round(got[1, :12], 5))
print("rows identical (0 vs 1):", np.allclose(finite[0], finite[1], atol=1e-6))
print("row spread:", np.abs(finite - finite[0]).max())
