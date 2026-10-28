#!/usr/bin/env python3
"""FusionC dashboard API -- Flask backend for the React app in app/.

Every run reports a nine-step pipeline, streamed to the UI as it happens:

  input -> tokens -> parse -> semantics -> fusion -> emit -> build -> execute -> validate

  GET  /api/health                       liveness check
  GET  /api/workloads                    workloads + DSL source + tensors + consts
  POST /api/run                          {workload, iters, warmup} -> {job_id}
  GET  /api/jobs/<id>?since=N            status + all stages + log[N:]
  GET  /api/artifact/<workload>/<file>   generated .cu / .ir.txt / .decisions.json
  POST /api/ast                          {workload} -> parse tree text
  GET  /api/recorded                     results/results.json (the measured evaluation)

Run:  python app/server.py [--port 5000]
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "driver"))

import fusionc_driver as drv  # noqa: E402  (the project's evaluation plumbing)

JAR = ROOT / "build" / "fusionc.jar"
DEMO_DIR = ROOT / "build" / "demo"
SEED = 20261025  # same seed as the evaluation report, so live numbers stay comparable

STAGE_ORDER = ["input", "tokens", "parse", "semantics", "fusion",
               "emit", "build", "execute", "validate"]

SUMMARY_RE = re.compile(r"emitted .*?: (\d+) kernel\(s\), (\d+) IR node\(s\), (\d+) decision\(s\)")

app = Flask(__name__)

jobs = {}
jobs_lock = threading.Lock()
active_job = None  # only one GPU run at a time: runs share the build/demo workspace


class Job:
    def __init__(self, workload, iters, warmup):
        self.id = uuid.uuid4().hex[:12]
        self.workload = workload
        self.iters = iters
        self.warmup = warmup
        self.status = "queued"  # queued | running | done | failed
        self.progress = 0
        self.total_steps = len(STAGE_ORDER)
        self.log = []
        self.result = None
        self.error = None
        self.stages = [{"id": sid, "status": "pending", "data": None} for sid in STAGE_ORDER]

    def say(self, line):
        self.log.append(line)

    def stage(self, sid, status, data=None):
        for entry in self.stages:
            if entry["id"] == sid:
                entry["status"] = status
                if data is not None:
                    entry["data"] = {**(entry["data"] or {}), **data}
                break
        self.progress = sum(1 for e in self.stages if e["status"] in ("done", "failed"))


# ------------------------------------------------------------------ pipeline steps

def java_run(java, dsl_path, extra, timeout=300):
    """Run the compiler jar and return its stdout (raises on failure)."""
    proc = subprocess.run(
        [str(java), "-jar", str(JAR), "--input", str(dsl_path)] + [str(x) for x in extra],
        capture_output=True, text=True, errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip()[:2000])
    return proc.stdout


def split_summary(text):
    """Separate the trailing 'emitted ...' summary line from the payload text."""
    lines = text.rstrip("\n").splitlines()
    if lines and lines[-1].startswith("emitted "):
        return "\n".join(lines[:-1]), lines[-1]
    return text, None


def emit_variant(java, spec, workload, tag, flag, work, job):
    """Compile one build mode: run the compiler, keep CUDA C + IR + decisions."""
    cu = work / f"{workload}_{tag}.cu"
    decisions = work / f"{workload}_{tag}.decisions.json"
    ir = work / f"{workload}_{tag}.ir.txt"
    job.say(f"$ java -jar fusionc.jar --input {spec['dsl']} {flag} -o {cu.name}")
    out = java_run(java, ROOT / spec["dsl"],
                   [flag, "--quiet", "--decisions", decisions, "--ir", ir, "-o", cu])
    text, summary = split_summary(out)
    kernels = nodes = decisions_n = None
    if summary:
        match = SUMMARY_RE.search(summary)
        if match:
            kernels, nodes, decisions_n = (int(match.group(i)) for i in (1, 2, 3))
    job.say(f"    {summary or 'ok'}")
    return {
        "cu": cu, "decisions": decisions, "ir": ir,
        "kernels": kernels, "nodes": nodes, "decisions_count": decisions_n,
    }


def ir_stats(ir_text):
    """Count the real operations in the IR table (everything that is not a host tensor)."""
    ops = []
    for line in ir_text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[2] not in ("INPUT", "PARAM"):
            ops.append(parts[2])
    return ops


def run_job(job):
    global active_job
    try:
        job.status = "running"
        spec = drv.WORKLOADS[job.workload]
        java = drv.find_java()
        dsl_path = ROOT / spec["dsl"]
        work = DEMO_DIR / job.workload

        # -- 1 input: read the program, generate the tensors the harness will load --
        job.stage("input", "running")
        consts, tensors = drv.read_dsl(dsl_path)
        source = dsl_path.read_text(encoding="utf-8")
        data = drv.generate_inputs(tensors, seed=SEED)
        drv.write_inputs(work / "inputs", data)
        expected, _ = drv.ORACLES[spec["oracle"]](data, consts)
        job.say(f"workload {job.workload}: {spec['dsl']} -- {len(tensors)} host tensors, "
                f"{job.iters} timed iterations / {job.warmup} warm-up")
        job.say(f"inputs regenerated with seed {SEED} into "
                f"{(work / 'inputs').relative_to(ROOT)}")
        job.stage("input", "done", {
            "dsl": spec["dsl"],
            "source": source,
            "consts": consts,
            "tensors": [{"name": name, "kind": t["kind"], "dtype": t["dtype"], "shape": t["shape"]}
                        for name, t in tensors.items()],
            "seed": SEED,
        })

        # -- 2 tokens: the JFlex lexer splits the text into tokens --
        job.stage("tokens", "running")
        out = java_run(java, dsl_path, ["--dump-tokens", "--quiet", "-o", work / "tokens_dump.cu"])
        token_text, _ = split_summary(out)
        match = re.search(r"\((\d+) tokens\)", token_text)
        token_count = int(match.group(1)) if match else None
        job.say(f"lexed {token_count} tokens from the DSL source")
        job.stage("tokens", "done", {"text": token_text, "count": token_count})

        # -- 3 parse: the CUP grammar turns tokens into a tree --
        job.stage("parse", "running")
        out = java_run(java, dsl_path, ["--dump-ast", "--quiet", "-o", work / "parse_dump.cu"])
        ast_text, _ = split_summary(out)
        job.say("parsed the token stream into a syntax tree (0 conflicts, LALR(1))")
        job.stage("parse", "done", {"text": ast_text})

        # -- 4 semantics + 6 emit (no-fusion arm): shapes, graph, plain CUDA C --
        job.stage("semantics", "running")
        unfused = emit_variant(java, spec, job.workload, "unfused", "--no-fusion", work, job)
        ir_text = unfused["ir"].read_text(encoding="utf-8")
        ops = ir_stats(ir_text)
        job.stage("semantics", "done", {
            "ir": ir_text,
            "nodes": unfused["nodes"],
            "ops": len(ops),
            "op_kinds": sorted(set(ops)),
        })
        job.say(f"semantic analysis: {unfused['nodes']} IR nodes "
                f"({len(ops)} operations + host tensors), all shapes checked")
        job.stage("emit", "running", {
            "unfused": {"cu": unfused["cu"].name, "kernels": unfused["kernels"],
                        "bytes": unfused["cu"].stat().st_size},
        })

        # -- 5 fusion + 6 emit (fused arm): the compiler's plan --
        fused = emit_variant(java, spec, job.workload, "fused", "--fuse", work, job)
        decision_log = json.loads(fused["decisions"].read_text(encoding="utf-8"))
        accepted = [d for d in decision_log["decisions"] if d["accepted"]]
        rejected = [d for d in decision_log["decisions"] if not d["accepted"]]
        job.stage("fusion", "done", {
            "decision_log": decision_log,
            "regions": decision_log["regions"],
            "accepted": len(accepted),
            "rejected": len(rejected),
            "kernels_before": unfused["kernels"],
            "kernels_after": fused["kernels"],
            "ir": fused["ir"].read_text(encoding="utf-8"),
        })
        job.say(f"fusion: {len(accepted)} accepted, {len(rejected)} rejected -> "
                f"{fused['kernels']} kernels instead of {unfused['kernels']}")
        job.stage("emit", "done", {
            "fused": {"cu": fused["cu"].name, "kernels": fused["kernels"],
                      "bytes": fused["cu"].stat().st_size},
        })

        # -- 7 build, 8 execute, 9 validate: per variant, on the real GPU --
        job.stage("build", "running")
        job.stage("execute", "running")
        job.stage("validate", "running")
        variants = {}
        for tag, emitted in (("unfused", unfused), ("fused", fused)):
            job.say(f"[{tag}] nvcc -arch=sm_61 -O3 -Xptxas -v {emitted['cu'].name}  (compiling ...)")
            result = drv.compile_and_run(f"{job.workload}_{tag}", emitted["cu"], work,
                                         work / "inputs", job.iters, job.warmup)
            if not result.compiled:
                raise RuntimeError(f"[{tag}] nvcc failed:\n" + result.build_log[-1500:])
            if result.run_error:
                raise RuntimeError(f"[{tag}] run failed: {result.run_error}")

            ptxas = {kernel: {"regs": s.get("regs"), "smem": s.get("smem", 0),
                              "spills": s.get("spill_stores", 0) + s.get("spill_loads", 0)}
                     for kernel, s in result.ptxas.items()}
            for kernel, s in sorted(ptxas.items()):
                job.say(f"    ptxas {kernel}: {s['regs']} regs, {s['smem']} B smem, {s['spills']} B spills")
            job.stage("build", "running", {tag: {"ptxas": ptxas, "kernels": emitted["kernels"]}})

            device = (result.device or {}).get("name", "GPU")
            job.say(f"[{tag}] ran {job.iters} timed iterations on {device}")
            for region, ms in result.regions.items():
                job.say(f"    {region:<30} {ms:8.4f} ms")
            job.stage("execute", "running", {
                tag: {"total_ms": result.total_ms, "regions": result.regions, "device": device},
            })

            actual = drv.read_output(result.output_bytes, "fp16")
            correctness = drv.compare(actual, expected)
            job.say(f"[{tag}] correctness vs fp64 oracle: "
                    f"{'PASS' if correctness['ok'] else 'FAIL'} "
                    f"(max abs {correctness['max_abs']:.3e})")
            job.stage("validate", "running", {
                tag: {
                    "correctness": correctness,
                    "checksum": result.checksum,
                    "elements": int(actual.size),
                    "preview": [round(float(v), 5) for v in actual.ravel()[:6]],
                },
            })
            variants[tag] = {"emitted": emitted, "run": result, "correctness": correctness,
                             "ptxas": ptxas}

        job.stage("build", "done")
        job.stage("execute", "done")
        job.stage("validate", "done", {
            "oracle": {
                "checksum": round(float(expected.sum()), 4),
                "elements": int(expected.size),
                "preview": [round(float(v), 5) for v in expected.ravel()[:6]],
                "precision": "float64 (Python / numpy)",
            },
        })

        # -- race summary: the same verdict logic the evaluation report uses --
        unfused_run, fused_run = variants["unfused"]["run"], variants["fused"]["run"]
        regions = []
        for region in decision_log.get("regions", []):
            if region["kind"] == "single":
                continue
            fused_ms = fused_run.regions.get(region["region"])
            unfused_ms = sum(unfused_run.regions.get(member, 0.0) for member in region["members"])
            if fused_ms is None or unfused_ms <= 0:
                continue
            change = (unfused_ms - fused_ms) / unfused_ms * 100.0
            regions.append({
                "region": region["region"], "kind": region["kind"],
                "members": region["members"], "fused_ms": fused_ms, "unfused_ms": unfused_ms,
                "change_pct": change,
                "verdict": "improved" if change > 5.0 else ("degraded" if change < -5.0 else "neutral"),
            })
        speedup = unfused_run.total_ms / fused_run.total_ms if fused_run.total_ms else None
        job.result = {
            "workload": job.workload, "iters": job.iters, "warmup": job.warmup, "seed": SEED,
            "device": (fused_run.device or {}).get("name", "GPU"),
            "kernels": {"unfused": unfused["kernels"], "fused": fused["kernels"]},
            "totals": {"unfused_ms": unfused_run.total_ms, "fused_ms": fused_run.total_ms,
                       "speedup": speedup},
            "regions": regions,
            "correctness": {"unfused": variants["unfused"]["correctness"],
                            "fused": variants["fused"]["correctness"]},
            "artifacts": {
                "cu": {"unfused": unfused["cu"].name, "fused": fused["cu"].name},
                "ir": {"unfused": unfused["ir"].name, "fused": fused["ir"].name},
                "decisions": {"unfused": unfused["decisions"].name,
                              "fused": fused["decisions"].name},
            },
        }
        job.say("")
        job.say(f"fused {fused_run.total_ms:.4f} ms  vs  no-fusion {unfused_run.total_ms:.4f} ms"
                f"  ->  {speedup:.2f}x")
        job.status = "done"
        job.progress = job.total_steps
    except Exception as exc:  # noqa: BLE001 -- surface any failure to the UI
        job.status = "failed"
        job.error = str(exc)
        for entry in job.stages:
            if entry["status"] == "running":
                entry["status"] = "failed"
                entry["data"] = {**(entry["data"] or {}), "error": str(exc)}
        job.say(f"ERROR: {exc}")
        job.say(traceback.format_exc().splitlines()[-1])
    finally:
        with jobs_lock:
            active_job = None


# ------------------------------------------------------------------ parse lab

GRAMMAR_BNF = """\
program    ::= decl_list stmt_list ;

decl_list  ::= decl_list decl | (empty) ;
decl       ::= CONST IDENT EQ INT SEMI
             | DTYPE datatype STORE datatype ACCUM SEMI
             | INPUT IDENT COLON datatype dims SEMI
             | PARAM IDENT COLON datatype dims SEMI ;
datatype   ::= FP16 | FP32 ;
dims       ::= LBRACKET expr_list RBRACKET ;

stmt_list  ::= stmt_list stmt | (empty) ;
stmt       ::= IDENT EQ stage_list SEMI ;                  the only statement form
stage_list ::= stage_list ARROW stage | stage ;            '->' chains: left-recursive => left-associative
stage      ::= IDENT LPAREN arglist RPAREN | IDENT ;       op call, or a bare name
arglist    ::= expr_list | (empty) ;

expr_list  ::= expr_list COMMA expr | expr ;
expr       ::= INT | FLOAT | IDENT | LPAREN expr RPAREN ;  no operators => no precedence table
"""

VALID_SAMPLE = """\
// a VALID program: declarations first, then one pipeline statement
const N = 8;

dtype fp16 store fp32 accum;

input x : fp16 [N, N];
param w : fp16 [N, N];

out = matmul(x, w) -> relu();
"""

INVALID_SAMPLE = """\
// an INVALID program: the pipeline's second stage is missing
const N = 8;

dtype fp16 store fp32 accum;

input x : fp16 [N, N];
param w : fp16 [N, N];

out = matmul(x, w) -> ;
"""

PARSE_LAB_DIR = DEMO_DIR / "parse_lab"


def clean_frontend_error(proc):
    """Strip the 'error: <file>.dsl: ' prefix from a compiler error message."""
    text = (proc.stderr or proc.stdout).strip()
    return re.sub(r"^error:\s*\S+\.dsl:\s*", "", text, count=1)


@app.get("/api/parse-lab")
def api_parse_lab():
    return jsonify({"bnf": GRAMMAR_BNF, "valid": VALID_SAMPLE, "invalid": INVALID_SAMPLE})


@app.post("/api/parse")
def api_parse():
    """Parse an arbitrary snippet: token stream + tree, or token stream + the error."""
    body = request.get_json(force=True, silent=True) or {}
    source = str(body.get("source", ""))
    if not source.strip():
        return jsonify({"ok": False, "tokens": "", "ast": "", "error": "empty input"})
    PARSE_LAB_DIR.mkdir(parents=True, exist_ok=True)
    src = PARSE_LAB_DIR / "input.dsl"
    src.write_text(source, encoding="utf-8")
    java = drv.find_java()

    def jar(flag):
        return subprocess.run(
            [str(java), "-jar", str(JAR), "--input", str(src), flag, "--quiet",
             "-o", str(PARSE_LAB_DIR / "lab.cu")],
            capture_output=True, text=True, errors="replace", timeout=120)

    tokens_proc = jar("--dump-tokens")
    ast_proc = jar("--dump-ast")
    tokens_text, _ = split_summary(tokens_proc.stdout)
    ast_text, _ = split_summary(ast_proc.stdout)
    ok = tokens_proc.returncode == 0 and ast_proc.returncode == 0
    error = None
    if not ok:
        failed = ast_proc if ast_proc.returncode != 0 else tokens_proc
        error = clean_frontend_error(failed)
    return jsonify({"ok": ok, "tokens": tokens_text, "ast": ast_text, "error": error})


# ------------------------------------------------------------------ endpoints

@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "jar": JAR.exists()})


@app.get("/api/workloads")
def api_workloads():
    out = []
    for name, spec in drv.WORKLOADS.items():
        dsl_path = ROOT / spec["dsl"]
        consts, tensors = drv.read_dsl(dsl_path)
        out.append({
            "id": name,
            "dsl": spec["dsl"],
            "question": spec["question"],
            "source": dsl_path.read_text(encoding="utf-8"),
            "consts": consts,
            "tensors": [{"name": key, "dtype": val["dtype"], "shape": val["shape"],
                         "kind": val["kind"]} for key, val in tensors.items()],
        })
    return jsonify(out)


@app.post("/api/run")
def api_run():
    global active_job
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("workload")
    if name not in drv.WORKLOADS:
        return jsonify({"error": f"unknown workload '{name}'"}), 400
    iters = max(10, min(2000, int(body.get("iters", 150))))
    warmup = max(0, min(iters, int(body.get("warmup", 50))))
    with jobs_lock:
        if active_job is not None and jobs.get(active_job) and \
                jobs[active_job].status in ("queued", "running"):
            return jsonify({"error": "a run is already in progress"}), 409
        job = Job(name, iters, warmup)
        jobs[job.id] = job
        active_job = job.id
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return jsonify({"job_id": job.id})


@app.get("/api/jobs/<job_id>")
def api_job(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "no such job"}), 404
    since = max(0, int(request.args.get("since", 0)))
    with jobs_lock:
        return jsonify({
            "id": job.id,
            "workload": job.workload,
            "status": job.status,
            "progress": job.progress,
            "total_steps": job.total_steps,
            "log": job.log[since:],
            "next": len(job.log),
            "error": job.error,
            "result": job.result if job.status == "done" else None,
            "stages": job.stages,
        })


ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@app.get("/api/artifact/<workload>/<filename>")
def api_artifact(workload, filename):
    if workload not in drv.WORKLOADS or not ARTIFACT_NAME.match(filename):
        abort(404)
    folder = (DEMO_DIR / workload).resolve()
    path = (folder / filename).resolve()
    if not str(path).startswith(str(folder)) or not path.is_file():
        abort(404)
    if path.suffix == ".json":
        return send_file(path, mimetype="application/json")
    return send_file(path, mimetype="text/plain")


@app.post("/api/ast")
def api_ast():
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("workload")
    if name not in drv.WORKLOADS:
        return jsonify({"error": f"unknown workload '{name}'"}), 400
    scratch = DEMO_DIR / name / "ast_dump.cu"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    out = java_run(drv.find_java(), ROOT / drv.WORKLOADS[name]["dsl"],
                   ["--dump-ast", "--quiet", "-o", scratch])
    text, _ = split_summary(out)
    return jsonify({"ast": text})


@app.get("/api/recorded")
def api_recorded():
    path = ROOT / "results" / "results.json"
    if not path.is_file():
        return jsonify({"error": "results/results.json not found"}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.after_request
def add_cors(resp):
    # only needed if the page is ever served from a different origin than this API
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionC dashboard API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    if not JAR.exists():
        sys.exit("build/fusionc.jar is missing -- run .\\scripts\\build.ps1 first")
    print(f"FusionC dashboard API on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
