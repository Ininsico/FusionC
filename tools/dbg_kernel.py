"""Insert debug prints into a generated translation unit and run it, to inspect device state."""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "driver"))
import fusionc_driver as d  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/attention_mlp/isolate_fused.cu")
work = src.parent
text = src.read_text(encoding="utf-8")

probe = (
    '    if (blockIdx.x == 0 && warp == 0 && lane == 0) {\n'
    '        printf("DBG afterStage Qs0=%f Kt00=%f Kt01=%f Kt10=%f\\n",\n'
    '               __half2float(Qs[0][0]), __half2float(St[0].h[0]),\n'
    '               __half2float(St[0].h[1]), __half2float(St[0].h[66]));\n'
    '    }\n'
)
anchor = "    float sc0 = -INFINITY, sc1 = -INFINITY;"
assert anchor in text, "anchor 1 not found"
text = text.replace(anchor, probe + anchor, 1)

probe2 = (
    '    if (blockIdx.x == 0 && warp == 0 && lane == 0) {\n'
    '        printf("DBG scores sc0=%f sc1=%f m=%f s=%f Pr00=%f\\n", sc0, sc1, m, s, Pr[0][0]);\n'
    '    }\n'
)
anchor2 = "        __syncwarp();   // P is consumed by all lanes of this warp in phase 2"
assert anchor2 in text, "anchor 2 not found"
text = text.replace(anchor2, anchor2 + "\n" + probe2, 1)

probe3 = ""

dbg = work / "dbg_fused.cu"
dbg.write_text(text, encoding="utf-8")
ok, log = d.nvcc_compile(dbg, work / "dbg_fused.exe", work / "dbg_fused.nvcc.log")
print("compiled:", ok)
if not ok:
    print(log[:3000])
    raise SystemExit(1)
proc = subprocess.run([str(work / "dbg_fused.exe"), "--inputs", str(work / "inputs"),
                       "--iters", "1", "--warmup", "0"], capture_output=True, text=True,
                      errors="replace")
for line in proc.stdout.splitlines():
    if line.startswith("DBG") or line.startswith("REGION") or line.startswith("CHECKSUM"):
        print(line)
print("stderr:", proc.stderr[:300])
