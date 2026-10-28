#!/usr/bin/env python3
"""FusionC live demo -- pick a workload, compile it WITH and WITHOUT fusion, run
both on the GPU, and print the before/after stats side by side.

Everything runs for real: java emits CUDA C, nvcc compiles it for sm_61, the
binaries run on the GPU, and the numbers are CUDA-event timings of that run.

Usage:
    python tools/live_demo.py                       # interactive workload picker
    python tools/live_demo.py attention_mlp          # run one directly
    python tools/live_demo.py -i 300 -w 50           # tuned iterations

Artifacts land in build/demo/<workload>/ (git-ignored).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "driver"))

import fusionc_driver as drv  # reuse the exact evaluation plumbing of the project

DEMO_DIR = ROOT / "build" / "demo"
SEED = 20261025  # same seed the evaluation report uses, so numbers stay comparable


def pick_workload(name):
    names = list(drv.WORKLOADS)
    if name:
        if name not in names:
            sys.exit(f"unknown workload '{name}' -- have: {', '.join(names)}")
        return name
    if not sys.stdin.isatty():
        sys.exit("no workload given -- pass one of: " + ", ".join(names))
    print("FusionC live demo -- pick a workload to compile and run:\n")
    for i, key in enumerate(names, 1):
        print(f"  {i}. {key:<16} {drv.WORKLOADS[key]['question']}")
    while True:
        choice = input(f"\nworkload number [1-{len(names)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]


def emit(java, dsl, flag, cu_path, decisions_path):
    """Run the compiler: DSL -> CUDA C (+ decision log)."""
    proc = subprocess.run(
        [str(java), "-jar", str(ROOT / "build" / "fusionc.jar"), "--input", str(dsl),
         flag, "--quiet", "--decisions", str(decisions_path), "-o", str(cu_path)],
        capture_output=True, text=True, errors="replace", timeout=300)
    if proc.returncode != 0:
        sys.exit((proc.stderr or proc.stdout).strip())
    lines = proc.stdout.strip().splitlines()
    return lines[-1] if lines else ""


def run_variant(java, name, spec, work, expected, tag, flag, iters, warmup, device):
    cu = work / f"{name}_{tag}.cu"
    decisions_path = work / f"{name}_{tag}.decisions.json"
    summary = emit(java, ROOT / spec["dsl"], flag, cu, decisions_path)
    print(f"  [{tag:<8}] {summary}")
    result = drv.compile_and_run(f"{name}_{tag}", cu, work, work / "inputs",
                                 iters, warmup, device=device)
    if not result.compiled:
        sys.exit(f"[{tag}] nvcc failed:\n{result.build_log[-2000:]}")
    if result.run_error:
        sys.exit(f"[{tag}] run failed: {result.run_error}")
    actual = drv.read_output(result.output_bytes, "fp16")
    result.correctness = drv.compare(actual, expected)
    result.decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    return result


def print_totals(unfused, fused):
    print("\n-- total runtime (CUDA events, mean per iteration) " + "-" * 30)
    print(f"  {'variant':<8} {'kernels':>7} {'total ms':>9} {'speedup':>8}   vs fp64 oracle")
    for tag, run in (("unfused", unfused), ("fused", fused)):
        kernels = len(run.decisions.get("regions", []))
        speedup = unfused.total_ms / run.total_ms if run.total_ms else float("nan")
        ok = "PASS" if run.correctness["ok"] else "FAIL"
        print(f"  {tag:<8} {kernels:>7} {run.total_ms:>9.4f} {speedup:>7.2f}x   "
              f"{ok} (max abs {run.correctness['max_abs']:.2e})")


def print_regions(unfused, fused):
    print("\n-- fused regions vs the unfused kernels they replaced " + "-" * 22)
    print(f"  {'region':<24} {'fused ms':>9} {'unfused sum ms':>15} {'change':>9}   verdict")
    for region in fused.decisions.get("regions", []):
        if region["kind"] == "single":
            continue
        fused_ms = fused.regions.get(region["region"])
        unfused_ms = sum(unfused.regions.get(member, 0.0) for member in region["members"])
        if fused_ms is None or unfused_ms <= 0:
            continue
        change = (unfused_ms - fused_ms) / unfused_ms * 100.0
        verdict = "improved" if change > 5.0 else ("degraded" if change < -5.0 else "neutral")
        print(f"  {region['region']:<24} {fused_ms:>9.4f} {unfused_ms:>15.4f} "
              f"{change:>+8.1f}%   {verdict}")


def print_ptxas(unfused, fused):
    print("\n-- ptxas per kernel (regs / smem / spills) " + "-" * 33)
    for tag, run in (("unfused", unfused), ("fused", fused)):
        for kernel, stats in sorted(run.ptxas.items()):
            spills = stats.get("spill_stores", 0) + stats.get("spill_loads", 0)
            print(f"  {tag:<8} {kernel:<28} regs={stats.get('regs', '?'):>3} "
                  f"smem={stats.get('smem', 0):>6} B   spills={spills} B")


def print_decisions(fused):
    print("\n-- compiler decisions " + "-" * 47)
    for decision in fused.decisions.get("decisions", []):
        tag = "FUSED " if decision["accepted"] else "REJECT"
        print(f"  {tag} {decision['kind']:<18} {', '.join(decision['members'])}")
        print(f"         {decision['reason'][:160]}")


def main():
    parser = argparse.ArgumentParser(
        description="FusionC live demo: fused vs unfused, measured on the GPU")
    parser.add_argument("workload", nargs="?", default=None,
                        help="workload name (omit for the interactive picker)")
    parser.add_argument("-i", "--iters", type=int, default=200)
    parser.add_argument("-w", "--warmup", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not (ROOT / "build" / "fusionc.jar").exists():
        sys.exit("build/fusionc.jar is missing -- run .\\scripts\\build.ps1 first")

    name = pick_workload(args.workload)
    spec = drv.WORKLOADS[name]
    java = drv.find_java()
    consts, tensors = drv.read_dsl(ROOT / spec["dsl"])
    data = drv.generate_inputs(tensors, seed=SEED)
    work = DEMO_DIR / name
    drv.write_inputs(work / "inputs", data)
    expected, _ = drv.ORACLES[spec["oracle"]](data, consts)

    print(f"\n=== {name} -- {spec['dsl']} ===")
    print(f"    {args.iters} timed iterations / {args.warmup} warm-up, device {args.device}\n")

    runs = {}
    for tag, flag in (("unfused", "--no-fusion"), ("fused", "--fuse")):
        runs[tag] = run_variant(java, name, spec, work, expected,
                                tag, flag, args.iters, args.warmup, args.device)

    unfused, fused = runs["unfused"], runs["fused"]
    print_totals(unfused, fused)
    print_regions(unfused, fused)
    print_ptxas(unfused, fused)
    print_decisions(fused)

    all_ok = unfused.correctness["ok"] and fused.correctness["ok"]
    print(f"\n[done] artifacts in {work.relative_to(ROOT)} -- "
          f"{'both variants match the fp64 oracle' if all_ok else 'CORRECTNESS FAILURE'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
