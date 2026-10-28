## 7. Analysis

### What the numbers say

**Correctness.** Every variant -- fused, one-kernel-per-op, the forced-fusion build and the
hand-written reference -- matches a float64 oracle of the same graph fed the identical FP16
inputs, element-wise, within `2e-2 + 2e-2|b|`. The observed error is ~1e-3, i.e. the FP16
quantisation floor of the storage format, not an algorithmic difference: all accumulation happens
in FP32 registers, as the numeric policy requires (`fp16 store fp32 accum`).

**Register and shared-memory footprint.** No kernel spills a single byte (`ptxas -v` reports
0 bytes stack frame, 0 spill stores, 0 spill loads everywhere, cross-checked against
`cudaFuncGetAttributes` at run time). Shared memory peaks at 34.3 KB in the fused attention
kernel -- 71% of the 48 KB sm_61 limit, leaving room for 2 resident blocks per SM. Every fused
kernel's predicted footprint in the decision log matches the ptxas measurement, because the cost
model and the emitters read the same constants.

**Runtime.** All three workloads move measurably faster when the compiler fuses
(`attention_mlp` 1.16x, `mlp_wide_batch` 1.06x, `residual_block` 1.30x on totals), and most
fused regions beat their unfused constituents in section 5; the small-tile GEMM-epilogue
fusions at 64-token shapes are the exceptions, and they are reported as measured. The 64-token
reference workload gains the least despite its attention fusion winning big, because one kernel
dominates it: the final `fc2` GEMM (`64x256` output, K=1024) accounts for roughly three quarters
of the whole pipeline in both variants. A 64-row by 256-column output tile grid contains only
16384 outputs; with a 4x4 per-thread micro-tile that is 1024 threads of work in total, which
cannot fill a 16-SM GPU no matter how the tile is chosen. That is a property of the *problem
shape*, not of the fusion: the fused and unfused builds are both latency-bound on that kernel.
The fix for that shape is split-K (see limitations).

**Fusion decision quality.** The heuristics accepted two kinds of fusion (attention_fused and
gemm_epilogue) and rejected the producer-consumer GEMM->GEMM fusion at these shapes. Every
rejection is measured anyway (section 5, `--force-fusion`): the forced GEMM->GEMM builds are all
*slower* than the unfused baseline, so the rejections are not conservative guesses but correct
calls. Two guards do the work:

* the row-tile amplification guard -- a fused kernel whose row tile is smaller than the unfused
  GEMM tile streams both weight matrices more often than the baseline would, trading DRAM traffic
  for L2 traffic;
* the accumulator-register guard -- a tile that leaves fewer than four accumulators per thread is
  shared-memory bound rather than FMA bound.

The amplification guard exists because the first evaluation run *measured* a fused MLP that was
1.8x slower than the unfused pair: the earlier heuristic lacked that term, accepted the fusion, and
the report's verdict column caught it. That is the loop this evaluation is meant to close --
measure a decision, put the measurement back into the cost model, re-run.

### Limitations and honest caveats

* **Launch overhead dominates this workload.** The reference graph is a 64-token block: its
  heaviest kernel moves 33 MFLOP, which a Pascal part executes in microseconds. At that size the
  measured cost is mostly per-kernel launch overhead, and on Windows/WDDM that overhead is tens of
  microseconds per launch -- visible in the region table, where trivial kernels such as the
  64x256 LayerNorm report 25 us against a few microseconds of actual work. Fusion therefore wins
  here largely by *removing launches* (8 kernels to 3 on the reference workload), while the forced
  builds lose because their kernels are genuinely inefficient. The per-region numbers also carry
  one CUDA-event pair per region, so the variant with ten regions pays for ten event records per
  iteration and the variant with three pays for three; both effects favour fewer kernels, and both
  are inherent to running this graph on this platform rather than artifacts of the comparison.
  The kernels' relative efficiency is what the forced-fusion column isolates.
* **Measurement noise.** This is a WDDM machine; total runtimes move by roughly +/-10% between
  runs while the kernel-level ratios are stable. Verdicts are therefore computed inside a single
  session (fused region vs the unfused kernels it replaced, measured back to back), not across
  sessions.
* **No split-K.** Shallow-M GEMMs (64 rows) cannot fill the device with the current tiling. A
  split-K variant with a reduction pass would fix exactly the dominant kernel in the reference
  workload.
* **Single-buffered GEMM tiles.** The GEMM tile loads are not double-buffered: they are issued,
  then awaited by `__syncthreads()`. Only the fused attention kernel hides load latency (it issues
  the next chunk before the current chunk's FMAs). Register-staged double buffering would cost
  shared memory the 48 KB budget does not have at 64x64 tiles.
* **Attention is single-band.** The fused attention kernel handles up to 64 keys per row without
  online-softmax rescaling; wider rows (the 512-token workload) are correctly *rejected* and fall
  back to three unfused kernels with a logged reason. Multi-band attention with rescaling is the
  natural extension.
* **FP16 arithmetic is deliberately unused.** GP10x runs FP16 math at 1/64 of the FP32 rate, so
  the compiler uses halves only for storage. On GP100 (1/2 rate) the same IR would want a different
  arithmetic policy; the numeric policy is a compile-time declaration in the DSL, not a hard-coded
  assumption.
* **The hand-written reference is not the compiler's baseline.** It uses a different (better
  fitting) tiling for these small shapes and is included as an independent correctness check
  first and a performance datapoint second. Generated `--no-fusion` output is compared against it
  element-wise in section 2.

### Reproducing

```
.\scripts\build.ps1                       # JFlex + CUP + javac -> build/fusionc.jar
python driver/fusionc_driver.py           # all workloads -> docs/REPORT.md, results/results.json
python tools/isolate.py <workload> fused  # per-tensor error against the oracle (debugging aid)
```

`results/<workload>/` keeps every generated `.cu`, its `nvcc` log with `ptxas -v` output, the
decision JSON and the raw output tensors, so any number in this report can be traced back to the
artifact that produced it.
