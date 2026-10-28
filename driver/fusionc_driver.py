#!/usr/bin/env python3
"""
FusionC evaluation driver.

Pipeline per workload:  parse DSL -> generate inputs + fp64 oracle -> compile with
--fuse and --no-fusion -> nvcc -arch=sm_61 (-Xptxas -v) -> run -> collect metrics ->
verdict per fusion decision -> markdown report.

Metrics reported (the four required by the project brief):
  1. correctness vs the reference implementations (hand-written unfused CUDA + numpy fp64 oracle)
  2. ptxas -v register / shared-memory / spill statistics per kernel
  3. fused vs unfused runtime measured with CUDA events
  4. fusion decision quality: improved / neutral / degraded verdict for every decision

Usage:
    python driver/fusionc_driver.py                       # all workloads, full evaluation
    python driver/fusionc_driver.py --workload attention_mlp --iters 300
    python driver/fusionc_driver.py --skip-build --skip-reference
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")
NVCC = CUDA_ROOT / "bin" / "nvcc.exe"
VCVARS = Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat")
VSWHERE = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")

ARCH = "sm_61"
NVCC_FLAGS = ["-arch=sm_61", "-O3", "-Xptxas", "-v"]
NVCC_FALLBACK_FLAGS = ["-allow-unsupported-compiler"]

# atol/rtol used to accept a result: FP16 storage with FP32 accumulation over K<=1024
# accumulates ~1e-3 relative error; the tolerance is deliberately looser than that.
ATOL = 2e-2
RTOL = 2e-2

WORKLOADS = {
    "attention_mlp": {
        "dsl": "examples/attention_mlp.dsl",
        "oracle": "attention_mlp",
        "reference": True,
        "question": "reference workload: fused attention + fused MLP vs 8 unfused kernels",
    },
    "mlp_wide_batch": {
        "dsl": "examples/mlp_wide_batch.dsl",
        "oracle": "attention_mlp",
        "reference": False,
        "question": "512 tokens: raises the fused-MLP candidate grid to 64 blocks; the accumulator-register guard still rejects it, and --force-fusion confirms that call",
    },
    "residual_block": {
        "dsl": "examples/residual_block.dsl",
        "oracle": "residual_block",
        "reference": False,
        "question": "LayerNorm + attention + residual add + GELU MLP: exercises the other emitters",
    },
}


# --------------------------------------------------------------------------- utils

def run(cmd, cwd=None, timeout=900, check=False, shell=False):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          shell=shell, errors="replace")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}\n{proc.stdout}\n{proc.stderr}")
    return proc


def find_java():
    java_home = os.environ.get("JAVA_HOME")
    candidates = []
    if java_home:
        candidates.append(Path(java_home) / "bin" / "java.exe")
    candidates.append(Path(shutil.which("java") or ""))
    for pattern in (r"C:\Program Files\Eclipse Adoptium\*\bin\java.exe",
                    r"C:\Program Files\Java\*\bin\java.exe"):
        candidates.extend(sorted(Path("C:/").glob(pattern[3:].replace("\\", "/"))))
    for candidate in candidates:
        if candidate and str(candidate) != "." and Path(candidate).exists():
            return Path(candidate)
    raise RuntimeError("no java.exe found; set JAVA_HOME")


JAVA = None


def vcvars_prefix():
    """Pin the host toolset, then initialise the VS/SDK environment.

    VCToolsVersion is what the VS environment scripts use to select the toolset, and nvcc
    re-runs vcvars internally when it detects an MSVC host compiler - so setting it here pins
    that nested invocation too.
    """
    toolset = find_msvc_toolset()
    prefix = f'set VCToolsVersion={toolset} && ' if toolset else ""
    if VCVARS.exists():
        prefix += f'call "{VCVARS}" >nul 2>&1 && '
    return prefix


def find_msvc_toolset():
    """Newest installed MSVC toolset that CUDA 11.8 can still consume (STL < 14.40)."""
    best = None
    for root in (Path(r"C:\Program Files\Microsoft Visual Studio"),
                 Path(r"C:\Program Files (x86)\Microsoft Visual Studio")):
        for toolset in root.glob("*/**/VC/Tools/MSVC/*"):
            if not toolset.is_dir():
                continue
            name = toolset.name
            try:
                major, minor = (int(x) for x in name.split(".")[:2])
            except ValueError:
                continue
            if (major, minor) >= (14, 40):
                continue
            if best is None or (major, minor) > tuple(int(x) for x in best.split(".")[:2]):
                best = name
    return best


def run_in_vs_env(command, timeout=300):
    """Run a command inside the pinned VS/SDK environment.

    A batch file is used because nesting vcvars inside a `cmd /c "..."` string mangles quoting.
    """
    BUILD.mkdir(parents=True, exist_ok=True)
    batch = BUILD / "env_cmd.bat"
    toolset = find_msvc_toolset()
    script = ["@echo off"]
    if toolset:
        script.append(f"set VCToolsVersion={toolset}")
    if VCVARS.exists():
        script.append(f'call "{VCVARS}" >nul 2>&1')
    script.append(command)
    script.append("exit /b %ERRORLEVEL%")
    batch.write_text("\r\n".join(script) + "\r\n", encoding="ascii")
    return subprocess.run(["cmd", "/c", str(batch)], capture_output=True, text=True,
                          errors="replace", timeout=timeout)


def nvcc_compile(cu_path, exe_path, log_path):
    """Compile one generated translation unit for sm_61, capturing ptxas -v output.

    The environment setup goes into a batch file: nvcc needs the VS/SDK environment
    (vcvars64.bat), and nesting that inside a `cmd /c "..."` string mangles the quotes.
    """
    flags = list(NVCC_FLAGS)
    text = ""
    for attempt in range(2):
        proc = run_in_vs_env(f'"{NVCC}" ' + " ".join(flags) + f' "{cu_path}" -o "{exe_path}"',
                             timeout=900)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode == 0 or attempt == 1:
            log_path.write_text(text, encoding="utf-8")
            return proc.returncode == 0, text
        if "unsupported Microsoft Visual Studio version" not in text:
            log_path.write_text(text, encoding="utf-8")
            return False, text
        flags = flags + NVCC_FALLBACK_FLAGS
    return False, text


PTXAS_ENTRY = re.compile(r"Compiling entry function '([^']+)'")
PTXAS_RESOURCES = re.compile(r"Used (\d+) registers(?:, (\d+) bytes smem)?")
PTXAS_STACK = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")


def demangle(name):
    """Recover the source name from an Itanium-mangled nvcc entry symbol (longest match)."""
    if name.startswith("_Z"):
        match = re.match(r"_Z(\d+)(.+)", name)
        if match:
            length = int(match.group(1))
            return match.group(2)[:length]
    return name


def parse_ptxas(log_text):
    """kernel name -> {regs, smem, spill_stores, spill_loads, stack}"""
    stats = {}
    current = None
    for line in log_text.splitlines():
        entry = PTXAS_ENTRY.search(line)
        if entry:
            current = demangle(entry.group(1))
            stats.setdefault(current, {})
            continue
        resources = PTXAS_RESOURCES.search(line)
        if resources and current:
            stats[current]["regs"] = int(resources.group(1))
            stats[current]["smem"] = int(resources.group(2) or 0)
            continue
        stack = PTXAS_STACK.search(line)
        if stack and current:
            stats[current]["stack"] = int(stack.group(1))
            stats[current]["spill_stores"] = int(stack.group(2))
            stats[current]["spill_loads"] = int(stack.group(3))
    return stats


# ------------------------------------------------------------------- DSL reading

def read_dsl(path):
    """Read constants and tensor declarations straight from the DSL (single source of truth)."""
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"#[^\n]*", "", text)
    consts = {}
    for name, value in re.findall(r"const\s+([A-Za-z_]\w*)\s*=\s*(\d+)\s*;", text):
        consts[name] = int(value)
    tensors = {}
    for kind, name, dtype, dims in re.findall(
            r"\b(input|param)\s+([A-Za-z_]\w*)\s*:\s*(fp16|fp32)\s*\[([^\]]+)\]", text):
        shape = []
        for token in dims.split(","):
            token = token.strip()
            shape.append(consts[token] if token in consts else int(token))
        tensors[name] = {"dtype": dtype, "shape": shape, "kind": kind}
    return consts, tensors


def elements(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


# ------------------------------------------------------------------------ oracles

def softmax_rows(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def layernorm_rows(x, gain, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(x * x, axis=-1, keepdims=True) - mean * mean
    return (x - mean) / np.sqrt(var + eps) * gain + beta


def oracle_attention_mlp(t, consts):
    """The graph of examples/attention_mlp.dsl and examples/mlp_wide_batch.dsl in float64."""
    seq, d, f = consts["SEQ"], consts["D"], consts["F"]
    q = t["Q"].astype(np.float64)
    k = t["K"].astype(np.float64)
    v = t["V"].astype(np.float64)
    w1 = t["W1"].astype(np.float64)
    b1 = t["b1"].astype(np.float64)
    w2 = t["W2"].astype(np.float64)
    b2 = t["b2"].astype(np.float64)
    scores = (q @ k.T) * (1.0 / np.sqrt(d))
    probs = softmax_rows(scores)
    attn = probs @ v
    hidden = np.maximum(attn @ w1 + b1, 0.0)
    out = hidden @ w2 + b2
    return out, {"attn": attn, "hidden": hidden}


def oracle_residual_block(t, consts):
    """The graph of examples/residual_block.dsl in float64."""
    x = t["x"].astype(np.float64)
    q = t["Q"].astype(np.float64)
    k = t["K"].astype(np.float64)
    v = t["V"].astype(np.float64)
    gain = t["gain"].astype(np.float64)
    beta = t["beta"].astype(np.float64)
    w1 = t["W1"].astype(np.float64)
    b1 = t["b1"].astype(np.float64)
    w2 = t["W2"].astype(np.float64)
    b2 = t["b2"].astype(np.float64)
    norm = layernorm_rows(x, gain, beta)
    probs = softmax_rows((q @ k.T) * (1.0 / np.sqrt(consts["D"])))
    attn = probs @ v
    res = norm + attn
    hidden = gelu_tanh(res @ w1 + b1)
    out = hidden @ w2 + b2
    return out, {"norm": norm, "attn": attn, "res": res, "hidden": hidden}


ORACLES = {"attention_mlp": oracle_attention_mlp, "residual_block": oracle_residual_block}

# intermediate tensor name in the generated harness -> oracle key
INTERMEDIATE_TENSORS = {
    "attention_mlp": [("out.attn_pv", "attn"), ("out.mlp_act", "hidden")],
    "residual_block": [("norm.layernorm", "norm"), ("att.attn_pv", "attn"),
                       ("res.add", "res"), ("out.mlp_act", "hidden")],
}


# ------------------------------------------------------------------ number tricks

def fp16(x):
    """Round to FP16 exactly as the device stores it (inputs are quantised identically)."""
    return x.astype(np.float16)


def generate_inputs(tensors, seed):
    rng = np.random.default_rng(seed)
    data = {}
    for name, spec in tensors.items():
        shape = tuple(spec["shape"])
        if spec["kind"] == "input" and name in ("Q", "K", "V", "x"):
            values = rng.standard_normal(shape)
        elif name.startswith("W1"):
            values = rng.standard_normal(shape) / np.sqrt(shape[0])
        elif name.startswith("W2"):
            values = rng.standard_normal(shape) / np.sqrt(shape[0])
        elif name in ("gain",):
            values = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            values = 0.1 * rng.standard_normal(shape)
        data[name] = values.astype(np.float16) if spec["dtype"] == "fp16" else values.astype(np.float32)
    return data


def write_inputs(input_dir, data):
    input_dir.mkdir(parents=True, exist_ok=True)
    for name, array in data.items():
        array.tofile(input_dir / f"{name}.bin")


def read_output(path, dtype):
    raw = np.fromfile(path, dtype=np.float16 if dtype == "fp16" else np.float32)
    return raw


def compare(actual, expected):
    actual = actual.astype(np.float64).ravel()
    expected = expected.astype(np.float64).ravel()
    diff = np.abs(actual - expected)
    denom = np.maximum(np.abs(expected), 1e-12)
    return {
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "max_rel": float(np.max(diff / denom)),
        "ok": bool(np.all(diff <= ATOL + RTOL * np.abs(expected))),
        "worst_violation": float(np.max(diff - (ATOL + RTOL * np.abs(expected)))),
    }


# ------------------------------------------------------------------------ harness

class Run:
    """Result of compiling + running one generated translation unit."""

    def __init__(self, name):
        self.name = name
        self.compiled = False
        self.build_log = ""
        self.ptxas = {}
        self.regions = {}
        self.total_ms = None
        self.kernel_attrs = {}
        self.device = None
        self.checksum = None
        self.output = None
        self.stdout = ""
        self.run_error = None
        self.nvcc_command = ""


def compile_and_run(label, cu_path, work_dir, input_dir, iters, warmup, device=0, timeout=900):
    result = Run(label)
    exe = work_dir / f"{label}.exe"
    log = work_dir / f"{label}.nvcc.log"
    result.nvcc_command = "nvcc " + " ".join(NVCC_FLAGS) + f" {cu_path.name} -o {exe.name}"
    ok, text = nvcc_compile(cu_path, exe, log)
    result.compiled = ok
    result.build_log = text
    result.ptxas = parse_ptxas(text)
    if not ok:
        return result

    out_bin = work_dir / f"{label}.out.bin"
    timings = work_dir / f"{label}.timings.csv"
    proc = subprocess.run([str(exe), "--inputs", str(input_dir), "--out", str(out_bin),
                           "--timings", str(timings), "--iters", str(iters),
                           "--warmup", str(warmup), "--device", str(device)],
                          capture_output=True, text=True, errors="replace", timeout=timeout)
    result.stdout = proc.stdout
    if proc.returncode != 0:
        result.run_error = f"exit {proc.returncode}: {proc.stderr[:400]}"
        return result
    for line in proc.stdout.splitlines():
        parts = line.strip().split(",")
        if not parts:
            continue
        if parts[0] == "REGION" and len(parts) >= 3:
            result.regions[parts[1]] = float(parts[2])
        elif parts[0] == "TOTAL" and len(parts) >= 2:
            result.total_ms = float(parts[1])
        elif parts[0] == "KERNELATTR" and len(parts) >= 5:
            result.kernel_attrs[parts[1]] = {"regs": int(parts[2]), "smem": int(parts[3]),
                                             "spills": int(parts[4])}
        elif parts[0] == "DEVICE" and len(parts) >= 4:
            result.device = {"name": parts[1], "cc": parts[2], "sm_count": int(parts[3])}
        elif parts[0] == "CHECKSUM" and len(parts) >= 2:
            result.checksum = float(parts[1])
        elif parts[0] == "OUTPUT" and len(parts) >= 4:
            result.output = {"tensor": parts[1], "elems": int(parts[2]), "esize": int(parts[3])}
    if out_bin.exists():
        result.output_bytes = out_bin
    return result


# ---------------------------------------------------------------------- workload

def evaluate_workload(name, spec, args):
    consts, tensors = read_dsl(ROOT / spec["dsl"])
    work = RESULTS / name
    input_dir = work / "inputs"
    work.mkdir(parents=True, exist_ok=True)
    data = generate_inputs(tensors, seed=args.seed)
    write_inputs(input_dir, data)

    expected, _ = ORACLES[spec["oracle"]](data, consts)
    out_spec = next(iter(reversed(list(tensors.items()))), None)
    out_dtype = "fp16"

    record = {
        "workload": name,
        "dsl": spec["dsl"],
        "question": spec["question"],
        "consts": consts,
        "tensors": {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in tensors.items()},
        "variants": {},
        "sm_count_sensitivity": [],
    }

    # three variants: the heuristic's answer, the one-kernel-per-op baseline, and -- for measuring
    # the decisions the heuristic rejected -- the same fusion with the profitability guards off.
    for tag, flag in (("fused", "--fuse"), ("unfused", "--no-fusion"), ("fused_forced", "--force-fusion")):
        cu = work / f"{name}_{tag}.cu"
        decisions = work / f"{name}_{tag}.decisions.json"
        ir = work / f"{name}_{tag}.ir.txt"
        cmd = [str(JAVA), "-jar", str(BUILD / "fusionc.jar"), "--input", str(ROOT / spec["dsl"]),
               flag, "--sm-count", str(args.sm_count), "-o", str(cu),
               "--decisions", str(decisions), "--ir", str(ir)]
        proc = run(cmd, timeout=300)
        if proc.returncode != 0:
            record["variants"][tag] = {"error": proc.stderr[:2000]}
            continue
        result = compile_and_run(f"{name}_{tag}", cu, work, input_dir, args.iters, args.warmup)
        variant = {
            "cu": str(cu.relative_to(ROOT)),
            "compiled": result.compiled,
            "decisions_json": str(decisions.relative_to(ROOT)),
            "ir": str(ir.relative_to(ROOT)),
            "total_ms": result.total_ms,
            "regions": result.regions,
            "ptxas": result.ptxas,
            "kernel_attrs": result.kernel_attrs,
            "device": result.device,
            "run_error": result.run_error,
            "nvcc_command": result.nvcc_command,
            "compiler_stdout": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
        }
        if not result.compiled:
            variant["nvcc_errors"] = [l for l in result.build_log.splitlines() if "error" in l.lower()][:12]
        if getattr(result, "output_bytes", None) and result.output:
            actual = read_output(result.output_bytes, out_dtype)
            variant["correctness_vs_oracle"] = compare(actual, expected)
            variant["checksum"] = result.checksum
            variant["output_elements"] = int(actual.size)
        try:
            variant["decision_log"] = json.loads(decisions.read_text(encoding="utf-8"))
        except Exception:
            variant["decision_log"] = None
        record["variants"][tag] = variant

    # cross-check: generated unfused vs hand-written reference
    reference_cu = ROOT / "reference" / "reference_unfused.cu"
    if spec["reference"] and not args.skip_reference and reference_cu.exists():
        result = compile_and_run(f"{name}_reference_handwritten", reference_cu, work, input_dir,
                                 args.iters, args.warmup)
        variant = {
            "cu": "reference/reference_unfused.cu",
            "compiled": result.compiled,
            "total_ms": result.total_ms,
            "regions": result.regions,
            "ptxas": result.ptxas,
            "kernel_attrs": result.kernel_attrs,
            "checksum": result.checksum,
            "run_error": result.run_error,
            "nvcc_command": result.nvcc_command,
        }
        if not result.compiled:
            variant["nvcc_errors"] = [l for l in result.build_log.splitlines() if "error" in l.lower()][:12]
        if getattr(result, "output_bytes", None) and result.output:
            actual = read_output(result.output_bytes, out_dtype)
            variant["correctness_vs_oracle"] = compare(actual, expected)
            if "unfused" in record["variants"] and "correctness_vs_oracle" in record["variants"]["unfused"]:
                gen = read_output(work / f"{name}_unfused.out.bin", out_dtype)
                variant["correctness_vs_generated_unfused"] = compare(actual, gen)
        record["variants"]["reference_handwritten"] = variant

    # fusion decision quality: each fused region vs the sum of its unfused constituent ops
    unfused = record["variants"].get("unfused", {})
    unfused_regions = unfused.get("regions") or {}
    for tag, key in (("fused", "verdicts"), ("fused_forced", "verdicts_forced")):
        variant = record["variants"].get(tag, {})
        verdicts = []
        if variant.get("decision_log") and unfused_regions:
            for region in variant["decision_log"].get("regions", []):
                if region["kind"] == "single":
                    continue
                members = region["members"]
                fused_ms = variant["regions"].get(region["region"])
                unfused_ms = sum(unfused_regions.get(m, 0.0) for m in members)
                if fused_ms is None or unfused_ms <= 0:
                    continue
                delta = unfused_ms - fused_ms
                change = delta / unfused_ms * 100.0
                if change > 5.0:
                    verdict = "improved"
                elif change < -5.0:
                    verdict = "degraded"
                else:
                    verdict = "neutral"
                verdicts.append({
                    "region": region["region"],
                    "kind": region["kind"],
                    "members": members,
                    "fused_ms": fused_ms,
                    "unfused_ms": unfused_ms,
                    "delta_ms": delta,
                    "change_pct": change,
                    "verdict": verdict,
                })
        record[key] = verdicts

    # heuristic sensitivity: what the decision changes with SM count (1050 Ti vs this GPU)
    for sm in (6, 15):
        cu = work / f"{name}_sm{sm}.cu"
        decisions = work / f"{name}_sm{sm}.decisions.json"
        proc = run([str(JAVA), "-jar", str(BUILD / "fusionc.jar"), "--input", str(ROOT / spec["dsl"]),
                    "--fuse", "--sm-count", str(sm), "-o", str(cu), "--decisions", str(decisions)],
                   timeout=300)
        if proc.returncode != 0:
            continue
        try:
            log = json.loads(decisions.read_text(encoding="utf-8"))
        except Exception:
            continue
        accepted = [d for d in log["decisions"] if d["accepted"]]
        record["sm_count_sensitivity"].append({
            "sm_count": sm,
            "kernels": len(log["regions"]),
            "accepted": [f"{d['kind']}:[{', '.join(d['members'])}]" for d in accepted],
            "rejected": [d["reason"] for d in log["decisions"] if not d["accepted"]],
        })
    return record


# ------------------------------------------------------------------------- report

def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(records, environment, args):
    lines = []
    add = lines.append
    add("# FusionC on Pascal (sm_61): evaluation report")
    add("")
    add("Generated by `driver/fusionc_driver.py` -- every number below is measured on the GPU in")
    add("this machine, not estimated.")
    add("")
    add("```")
    for key, value in environment.items():
        add(f"{key:<22} {value}")
    add("```")
    add("")
    add("## 1. What was compiled")
    add("")
    add("| workload | DSL | fused kernels | unfused kernels | question |")
    add("|---|---|---|---|---|")
    for record in records:
        fused = record["variants"].get("fused", {})
        unfused = record["variants"].get("unfused", {})
        n_fused = len(fused.get("decision_log", {}).get("regions", [])) if fused.get("decision_log") else "n/a"
        n_unfused = len(unfused.get("decision_log", {}).get("regions", [])) if unfused.get("decision_log") else "n/a"
        add(f"| {record['workload']} | `{record['dsl']}` | {n_fused} | {n_unfused} | {record['question']} |")
    add("")

    add("## 2. Correctness")
    add("")
    add(f"Compared against a float64 oracle of the same graph fed with the identical FP16 inputs.")
    add(f"Acceptance: `|a-b| <= {ATOL} + {RTOL}*|b|` per element.")
    add("")
    add("| workload | variant | vs fp64 oracle | max abs | max rel | checksum |")
    add("|---|---|---|---|---|---|")
    for record in records:
        for tag in ("fused", "unfused", "fused_forced", "reference_handwritten"):
            variant = record["variants"].get(tag)
            if not variant:
                continue
            correctness = variant.get("correctness_vs_oracle")
            if not correctness:
                add(f"| {record['workload']} | {tag} | not run | - | - | - |")
                continue
            ok = "PASS" if correctness["ok"] else "FAIL"
            add(f"| {record['workload']} | {tag} | {ok} | {correctness['max_abs']:.3e} | "
                f"{correctness['max_rel']:.3e} | {fmt(variant.get('checksum'), 6)} |")
    add("")
    for record in records:
        reference = record["variants"].get("reference_handwritten", {})
        if "correctness_vs_generated_unfused" in reference:
            c = reference["correctness_vs_generated_unfused"]
            add(f"* `{record['workload']}`: generated `--no-fusion` vs hand-written reference -- "
                f"max abs {c['max_abs']:.3e}, max rel {c['max_rel']:.3e}, "
                f"{'PASS' if c['ok'] else 'FAIL'}.")
    add("")

    add("## 3. ptxas -v: registers, shared memory, spills")
    add("")
    add("| workload | variant | kernel | regs/thread | smem/block | spill stores | spill loads |")
    add("|---|---|---|---|---|---|---|")
    for record in records:
        for tag in ("fused", "unfused", "fused_forced", "reference_handwritten"):
            variant = record["variants"].get(tag)
            if not variant or not variant.get("ptxas"):
                continue
            for kernel, stats in sorted(variant["ptxas"].items()):
                add(f"| {record['workload']} | {tag} | `{kernel}` | {stats.get('regs', '?')} | "
                    f"{stats.get('smem', 0)} B | {stats.get('spill_stores', 0)} B | "
                    f"{stats.get('spill_loads', 0)} B |")
    add("")

    add("## 4. Runtime (CUDA events, mean per iteration)")
    add("")
    add("| workload | variant | total ms | speedup vs unfused | speedup vs hand reference |")
    add("|---|---|---|---|---|")
    for record in records:
        unfused_ms = record["variants"].get("unfused", {}).get("total_ms")
        ref_ms = record["variants"].get("reference_handwritten", {}).get("total_ms")
        for tag in ("fused", "unfused", "fused_forced", "reference_handwritten"):
            variant = record["variants"].get(tag)
            if not variant or variant.get("total_ms") is None:
                continue
            total = variant["total_ms"]
            vs_unfused = f"{unfused_ms / total:.3f}x" if unfused_ms and tag != "unfused" else "-"
            vs_ref = f"{ref_ms / total:.3f}x" if ref_ms and tag != "reference_handwritten" else "-"
            add(f"| {record['workload']} | {tag} | {total:.4f} | {vs_unfused} | {vs_ref} |")
    add("")

    add("### Per-region timings")
    add("")
    add("| workload | variant | region | ms |")
    add("|---|---|---|---|")
    for record in records:
        for tag in ("fused", "unfused", "fused_forced", "reference_handwritten"):
            variant = record["variants"].get(tag)
            if not variant or not variant.get("regions"):
                continue
            for region, ms in variant["regions"].items():
                add(f"| {record['workload']} | {tag} | `{region}` | {ms:.4f} |")
    add("")

    add("## 5. Fusion decision quality")
    add("")
    add("Each fused region is compared with the sum of the unfused kernels it replaced, both measured")
    add("in the same session. A decision counts as improved above +5%, degraded below -5%.")
    add("")
    add("| workload | fusion region | merged ops | fused ms | unfused sum ms | delta | verdict |")
    add("|---|---|---|---|---|---|---|")
    for record in records:
        for verdict in record.get("verdicts", []):
            add(f"| {record['workload']} | `{verdict['region']}` | {', '.join(verdict['members'])} | "
                f"{verdict['fused_ms']:.4f} | {verdict['unfused_ms']:.4f} | "
                f"{verdict['delta_ms']:+.4f} ms ({verdict['change_pct']:+.1f}%) | **{verdict['verdict']}** |")
    add("")
    add("### Rejected decisions, measured anyway (`--force-fusion`)")
    add("")
    add("Fusion candidates the heuristics rejected are still compiled and measured, so the rejection")
    add("itself can be scored. Here the same region is built with the profitability guards disabled")
    add("(the resource caps still apply) and timed against the unfused kernels it would have replaced.")
    add("")
    add("| workload | fusion region | merged ops | forced ms | unfused sum ms | delta | verdict |")
    add("|---|---|---|---|---|---|---|")
    forced_rows = 0
    for record in records:
        for verdict in record.get("verdicts_forced", []):
            forced_rows += 1
            add(f"| {record['workload']} | `{verdict['region']}` | {', '.join(verdict['members'])} | "
                f"{verdict['fused_ms']:.4f} | {verdict['unfused_ms']:.4f} | "
                f"{verdict['delta_ms']:+.4f} ms ({verdict['change_pct']:+.1f}%) | **{verdict['verdict']}** |")
    if forced_rows == 0:
        add("| - | - | - | - | - | - | - |")
    add("")
    add("### Every logged decision (accepted and rejected)")
    add("")
    for record in records:
        fused = record["variants"].get("fused", {})
        log = fused.get("decision_log")
        if not log:
            continue
        add(f"#### {record['workload']}")
        add("")
        for decision in log["decisions"]:
            status = "FUSED" if decision["accepted"] else "REJECTED"
            add(f"* **{status}** `{decision['kind']}` -- {', '.join(decision['members'])}")
            predicted = decision.get("predicted", {})
            if predicted:
                add(f"  * predicted: " + ", ".join(f"{k}={v}" for k, v in predicted.items()))
            add(f"  * reason: {decision['reason']}")
        add("")

    add("## 6. Heuristic sensitivity to the SM count")
    add("")
    add("The fusion heuristics use the target SM count: 6 for a GTX 1050 Ti, 15 for the GTX 1070 in")
    add("this machine. Same DSL, same source, different occupancy floor:")
    add("")
    add("| workload | SMs | kernels | accepted fusions |")
    add("|---|---|---|---|")
    for record in records:
        for entry in record.get("sm_count_sensitivity", []):
            add(f"| {record['workload']} | {entry['sm_count']} | {entry['kernels']} | "
                f"{'; '.join(entry['accepted']) if entry['accepted'] else '(none)'} |")
    add("")

    analysis = DOCS / "analysis.md"
    if analysis.exists():
        add(analysis.read_text(encoding="utf-8").rstrip())
        add("")
    return "\n".join(lines) + "\n"


def detect_device():
    """Compile and run tools/device_probe.cu to read the real SM count and resource limits."""
    BUILD.mkdir(parents=True, exist_ok=True)
    ok, text = nvcc_compile(ROOT / "tools" / "device_probe.cu", BUILD / "device_probe.exe",
                            BUILD / "device_probe.nvcc.log")
    if not ok:
        return None
    proc = subprocess.run([str(BUILD / "device_probe.exe")], capture_output=True, text=True,
                          errors="replace", timeout=120)
    parts = proc.stdout.strip().split(",")
    if len(parts) < 10 or parts[0] != "DEVICE":
        return None
    return {
        "name": parts[1], "compute_capability": parts[2], "sm_count": int(parts[3]),
        "smem_per_block": int(parts[4]), "smem_per_sm": int(parts[5]),
        "regs_per_sm": int(parts[6]), "threads_per_sm": int(parts[7]),
        "threads_per_block": int(parts[8]), "clock_khz": int(parts[9]),
    }


def collect_environment(args, device):
    env = {}
    proc = run([str(JAVA), "-version"], shell=False)
    env["java"] = (proc.stderr or proc.stdout).splitlines()[0] if (proc.stderr or proc.stdout) else "?"
    proc = run_in_vs_env(f'"{NVCC}" --version')
    version = [l for l in proc.stdout.splitlines() if "release" in l and "Cuda compilation" in l]
    env["nvcc"] = version[-1].strip() if version else "n/a"
    env["host toolset"] = f"MSVC {find_msvc_toolset()} (pinned via VCToolsVersion)"
    env["windows sdk"] = "10.0.22621.0"
    proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader"], capture_output=True, text=True, errors="replace")
    env["gpu"] = proc.stdout.strip() or "?"
    if device:
        env["device limits"] = (f"{device['sm_count']} SMs, {device['smem_per_block']} B smem/block, "
                                f"{device['smem_per_sm']} B smem/SM, {device['regs_per_sm']} regs/SM, "
                                f"{device['threads_per_sm']} threads/SM, {device['clock_khz'] / 1000:.0f} MHz")
    env["nvcc flags"] = " ".join(NVCC_FLAGS)
    env["sm count given to heuristics"] = str(args.sm_count)
    env["iterations"] = f"{args.iters} timed, {args.warmup} warm-up"
    env["seed"] = str(args.seed)
    env["projects root"] = str(ROOT)
    return env


# --------------------------------------------------------------------------- main

def main():
    global JAVA
    parser = argparse.ArgumentParser(description="FusionC evaluation driver")
    parser.add_argument("--workload", default="all",
                        help="workload name, comma separated, or 'all'")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--sm-count", type=int, default=None,
                        help="SM count handed to the fusion heuristics (default: probed from the device)")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--force-fusion", action="store_true")
    parser.add_argument("--report", default=str(DOCS / "REPORT.md"))
    parser.add_argument("--results", default=str(RESULTS / "results.json"))
    parser.add_argument("--from-results", action="store_true",
                        help="rebuild the report from an existing results.json (no GPU work)")
    args = parser.parse_args()

    if args.from_results:
        data = json.loads(Path(args.results).read_text(encoding="utf-8"))
        report = build_report(data["workloads"], data["environment"], args)
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"[done] report rebuilt from {args.results} -> {args.report}")
        return 0

    JAVA = find_java()
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        print("[build] generating scanner/parser and compiling the Java compiler ...")
        proc = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                               "-File", str(ROOT / "scripts" / "build.ps1")],
                              capture_output=True, text=True, errors="replace", timeout=900)
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            raise SystemExit("compiler build failed")
        print("        ok")

    names = list(WORKLOADS) if args.workload == "all" else args.workload.split(",")
    device = detect_device()
    if device:
        print(f"[device] {device['name']} cc{device['compute_capability']} "
              f"{device['sm_count']} SMs -- heuristics will use sm_count={device['sm_count']}")
    if args.sm_count is None:
        args.sm_count = device["sm_count"] if device else 6
    records = []
    for name in names:
        spec = WORKLOADS[name]
        print(f"[eval] {name}: {spec['question']}")
        record = evaluate_workload(name, spec, args)
        for tag, variant in record["variants"].items():
            state = "compiled" if variant.get("compiled") else "FAILED TO COMPILE"
            total = variant.get("total_ms")
            correctness = variant.get("correctness_vs_oracle", {})
            ok = "correct" if correctness.get("ok") else ("incorrect" if correctness else "not run")
            print(f"        {tag:<22} {state:<20} {ok:<10} "
                  f"total={fmt(total, 4) if total else 'n/a'} ms")
        records.append(record)

    environment = collect_environment(args, device)
    report = build_report(records, environment, args)
    Path(args.report).write_text(report, encoding="utf-8")
    Path(args.results).write_text(json.dumps({"environment": environment, "workloads": records},
                                             indent=2), encoding="utf-8")
    print(f"[done] report -> {args.report}")
    print(f"       raw results -> {args.results}")

    failures = []
    attempted = []
    for record in records:
        for tag, variant in record["variants"].items():
            if "error" in variant:
                failures.append(f"{record['workload']}/{tag}: {variant['error'][:80]}")
                continue
            attempted.append(f"{record['workload']}/{tag}")
            if not variant.get("compiled"):
                failures.append(f"{record['workload']}/{tag}: did not compile")
                continue
            if variant.get("run_error"):
                failures.append(f"{record['workload']}/{tag}: {variant['run_error'][:80]}")
                continue
            correctness = variant.get("correctness_vs_oracle")
            if not correctness:
                failures.append(f"{record['workload']}/{tag}: no correctness data (did it run?)")
            elif not correctness["ok"]:
                failures.append(f"{record['workload']}/{tag}: correctness FAIL "
                                f"(max abs {correctness['max_abs']:.3e})")
    if failures:
        print("[FAIL] " + "; ".join(failures))
        return 1
    print(f"[PASS] all {len(attempted)} variants compiled, ran and matched the fp64 oracle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
