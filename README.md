# FusionC

**A source-to-source compiler that turns a small tensor-pipeline DSL into fused CUDA C for
consumer Pascal GPUs** (`sm_61`: GTX 1050 Ti / GTX 1070 — no Tensor Cores, no `cp.async`, no
BF16, 48 KB shared memory per block, 255 registers per thread, CUDA 11.x toolchain).

```
layer = attention(Q, K, V) -> MLP(W1, relu, W2, b1, b2);
```

becomes, with `--fuse`, three kernels instead of eight — and the fused kernels are *generated
from the IR*, never hand-written.

This is the semester project for the Compilers course. The viva topic is **the grammar rules
for statements and expressions of the language, including input and output**: that is
[Section 2](#2-the-language--statements-expressions-and-io) and
[Section 6](#6-live-demo-how-to-prove-it-works). Everything else is the machinery around it.

---

## The one-paragraph summary

* **Input:** a 10-line DSL program (tensor declarations + pipelines).
* **Output:** a self-contained `.cu` file (kernels + static memory plan + host harness) that
  compiles with `nvcc -arch=sm_61` and runs on the GPU.
* **In between:** JFlex lexer → CUP LALR parser → AST → shape/dtype inference → macro expansion
  of composite ops → 15-node IR DAG → fusion pass with a Pascal resource cost model → CUDA C
  emitter.
* **Proof:** 3 workloads × 3 build modes + a hand-written reference (10 compiled variants), all
  matching an fp64 oracle element-wise, **1.16x–1.30x fused-vs-unfused speedups**, zero register
  spills, and every fusion decision (accepted *and* rejected) logged with its predicted
  resources and scored by measurement. Full numbers: [`docs/REPORT.md`](docs/REPORT.md).

---

## Contents

1. [What this project is, and why](#1-what-this-project-is-and-why)
2. [The language — statements, expressions, and I/O](#2-the-language--statements-expressions-and-io)
3. [Compiler architecture (what happens to a program)](#3-compiler-architecture)
4. [What the compiler generates](#4-what-the-compiler-generates)
5. [Build from source](#5-build-from-source)
6. [Live demo: how to prove it works](#6-live-demo-how-to-prove-it-works)
7. [Evaluation: the four metrics and the measured results](#7-evaluation-the-four-metrics)
8. [Pascal-specific design decisions](#8-pascal-specific-design-decisions)
9. [Known limitations and future work](#9-known-limitations-and-future-work)
10. [Repository layout](#10-repository-layout)
11. [Viva cheat sheet (Q&A)](#11-viva-cheat-sheet)
12. [Appendix A: the `CyberScript/` folder](#appendix-a-the-cyberscript-folder)
13. [Appendix B: toolchain notes (this machine)](#appendix-b-toolchain-notes-this-machine)

---

## 1. What this project is, and why

### The problem

Transformer-style inference on a cheap consumer GPU is bottlenecked by three things on Pascal:

1. **Memory bandwidth.** A naive implementation materialises every intermediate tensor
   (attention scores, softmax probabilities, hidden activations...) as a full round-trip to
   DRAM. On a 1050 Ti that traffic costs more than the arithmetic.
2. **Kernel launch overhead.** A 64-token workload splits into many tiny kernels; on Windows
   (WDDM) each launch has tens of microseconds of overhead, which can dwarf the kernel itself.
3. **Hardware limits.** GP10x has no Tensor Cores, no async copy (`cp.async`), only 48 KB of
   shared memory per block, and runs FP16 *arithmetic* at 1/64 of the FP32 rate — so half
   precision can be used for **storage** only.

### What we built

A complete compiler that takes a graph description and produces CUDA C which avoids those
costs by **fusing producer→consumer operations** under an explicit, measurable resource model:

* intermediates stay in registers or shared memory instead of DRAM,
* fusing an op pair removes a kernel launch,
* every fusion is a *decision*: predicted cost first (`ResourceModel`), Pascal-specific
  acceptance rules (`PascalHeuristics`), and a human-readable reason logged whether it is
  accepted or rejected,
* rejected candidates are still compiled and measured (`--force-fusion`), so the rejections
  themselves are validated rather than assumed.

### The four required metrics

The brief (and the report) covers four things, all measured, never estimated:

| # | metric | where |
|---|---|---|
| 1 | correctness vs a reference implementation | fp64 numpy oracle + hand-written CUDA reference |
| 2 | `ptxas -v` register / shared-memory / spill statistics | per kernel, every variant |
| 3 | fused vs unfused runtime (CUDA events) | per region and per total |
| 4 | fusion decision quality (improved / neutral / degraded) | per decision, including rejected ones |

`driver/fusionc_driver.py` produces all four automatically into `docs/REPORT.md` and raw
`results/results.json`. It exits non-zero if anything fails to compile, run, or validate.

---

## 2. The language — statements, expressions, and I/O

Everything in this section is implemented by
`src/main/jflex/Scanner.flex` (lexer) and `src/main/cup/Parser.cup` (LALR grammar), which the
build script turns into `Scanner.java`, `parser.java` and `sym.java` under
`build/generated/io/fusionc/frontend/`.

### 2.1 A complete example

`examples/attention_mlp.dsl` — the reference workload:

```
// SEQ tokens, D model dim, F feed-forward dim
const SEQ = 64;
const D   = 256;
const F   = 1024;

dtype fp16 store fp32 accum;

input Q  : fp16 [SEQ, D];
input K  : fp16 [SEQ, D];
input V  : fp16 [SEQ, D];

param W1 : fp16 [D, F];
param b1 : fp32 [F];
param W2 : fp16 [F, D];
param b2 : fp32 [D];

out = attention(Q, K, V) -> MLP(W1, relu, W2, b1, b2);
```

### 2.2 Lexical structure (JFlex scanner)

| category | tokens |
|---|---|
| keywords | `const` `dtype` `store` `accum` `input` `param` `fp16` `fp32` |
| punctuation | `= ; , : ( ) [ ] ->` |
| literals | `INT` = `[0-9]+` · `FLOAT` = `1.5`, `2e-3`, `1.5e+2` · `IDENT` = `[A-Za-z_][A-Za-z0-9_]*` |
| comments | `// ...` (line) · `# ...` (line) · `/* ... */` (block) |
| whitespace | spaces, tabs, newlines — skipped |

Keywords are lowercase and case-sensitive. The scanner runs in JFlex `%cup` mode, so
`next_token()` returns `java_cup.runtime.Symbol` objects carrying `(type, line, column, value)`
with 1-based line/column for diagnostics. **Any illegal character is a hard error** — e.g.
`lexical error at line 3, column 7: illegal character '@'` — and the compiler exits with
status 1. It never guesses.

### 2.3 The whole grammar (as implemented in `Parser.cup`)

This is the exact BNF, CUP `terminal`/`non terminal` names included:

```
program    ::= decl_list stmt_list ;

decl_list  ::= decl_list decl | /* empty */ ;
decl       ::= CONST  IDENT EQ INT SEMI
             | DTYPE  datatype STORE datatype ACCUM SEMI
             | INPUT  IDENT COLON datatype dims SEMI
             | PARAM  IDENT COLON datatype dims SEMI ;
datatype   ::= FP16 | FP32 ;
dims       ::= LBRACKET expr_list RBRACKET ;

stmt_list  ::= stmt_list stmt | /* empty */ ;
stmt       ::= IDENT EQ stage_list SEMI ;
stage_list ::= stage_list ARROW stage | stage ;
stage      ::= IDENT LPAREN arglist RPAREN | IDENT ;
arglist    ::= expr_list | /* empty */ ;

expr_list  ::= expr_list COMMA expr | expr ;
expr       ::= INT | FLOAT | IDENT | LPAREN expr RPAREN ;
```

Two structural facts worth pointing out in the viva:

* A program is **declarations first, then statements** (`program ::= decl_list stmt_list`).
  Declarations cannot be interleaved with pipelines, and a constant/tensor can therefore never
  be used before it is declared.
* The `ARROW` token is `->` — the sequential-composition operator of the pipeline language.

Run through CUP, this grammar produces **22 terminals, 12 non-terminals, 26 productions and 56
LALR(1) parse states with 0 conflicts and 0 warnings** (the build prints this summary on every
run) — the unambiguous-parsing claim is printed by the toolchain, not asserted by us.

### 2.4 Grammar rules for **statements**

There is exactly one statement form: the **pipeline binding**.

```
stmt       ::= IDENT EQ stage_list SEMI ;
stage_list ::= stage_list ARROW stage | stage ;
stage      ::= IDENT LPAREN arglist RPAREN | IDENT ;
```

* `name = stage -> stage -> ... ;` binds the pipeline result to `name`, which becomes a new
  tensor. Later pipelines may reference it (see `examples/residual_block.dsl`, where `norm`,
  `att`, `res` are consumed by later statements).
* A **stage** is either an operation call (`attention(Q, K, V)`, `MLP(...)`, `relu(...)`) or a
  **bare identifier** (`res -> MLP(...)`).
* `->` means "feed the previous stage's result into this stage as its subject (first
  positional argument)".
* The rule `stage_list ::= stage_list ARROW stage` is **left-recursive, so `->` is
  left-associative**: `a -> b -> c` groups as `(a -> b) -> c`. This is the only associativity
  in the language, and it is unambiguous for LALR(1) — CUP parses it with no precedence
  declarations at all.
* `stmt_list` is also left-recursive: a program is a sequence of pipelines in source order.

### 2.5 Grammar rules for **expressions**

```
expr_list ::= expr_list COMMA expr | expr ;
expr      ::= INT | FLOAT | IDENT | LPAREN expr RPAREN ;
```

Expressions appear in exactly two places: **tensor dimensions** (`dims`) and **operator
arguments** (`arglist`). An expression is a fragment: an integer literal, a float literal, a
reference by name, or a parenthesised sub-expression (parentheses are grouping only — the
semantic action returns the inner expression unchanged).

**There are no infix operators, so the expression grammar has no precedence table — nothing
can be ambiguous.** This is a deliberate design decision: the language describes a *dataflow
graph*, not arithmetic. Computation is expressed by naming operations in pipelines; numeric
precision is expressed by the `dtype ... store ... accum` declaration. Names inside
expressions are resolved semantically:

* in `dims`, an `IDENT` must be a previously declared constant (`SEQ`, `D`, `F`); a non-integer
  or unknown dimension is a compile error (`tensor 'x' has a non-integer dimension`,
  `unknown constant 'FOO'`).
* in `arglist`, an `IDENT` must be a declared tensor — except in the activation slot of
  `MLP(W1, act, W2, ...)`, where `relu` / `gelu` name an activation function.

> If your examiner expects a *classic* expression grammar with precedence and associativity,
> the honest answer is: this language intentionally has none, because it has no expression
> operators; the only chaining construct is `->` at statement level, and its left
> associativity comes from the left-recursive `stage_list` rule. (The separate `CyberScript/`
> folder — [Appendix A](#appendix-a-the-cyberscript-folder) — contains an earlier exercise
> whose grammar *does* have precedence: `PLUS/MINUS` left-associative, `MULT/DIV` higher.)

### 2.6 What the parser builds (AST)

Each grammar rule has a semantic action building an `io.fusionc.ast.Ast` node
(`Program`, `ConstDecl`, `DtypeDecl`, `TensorDecl`, `Pipeline`, `Call`, `Name`, `IntLit`,
`FloatLit`, `Ref`). You can print the parse result with `--dump-ast` — real output for the
reference example:

```
Program
  const SEQ = 64
  const D = 256
  const F = 1024
  dtype store=fp16 accum=fp32
  input Q : fp16 [SEQ, D]
  input K : fp16 [SEQ, D]
  input V : fp16 [SEQ, D]
  param W1 : fp16 [D, F]
  param b1 : fp32 [F]
  param W2 : fp16 [F, D]
  param b2 : fp32 [D]
  pipeline out = attention(Q, K, V) -> MLP(W1, relu, W2, b1, b2)
```

### 2.7 Semantics: declarations → shapes → IR

The parser only checks *syntax*; `io.fusionc.semantics.GraphBuilder` then performs, in order:

1. **Constant table** — duplicates and non-positive values rejected.
2. **Tensor declarations** — duplicate names rejected; every tensor is 1-D (bias/gain vector)
   or 2-D (matrix), row-major, contiguous; dimensions resolve through the constant table.
3. **Numeric policy** — `dtype fp16 store fp32 accum;` sets the default storage and accumulate
   dtypes; a tensor declaration may override storage (`param b1 : fp32 [F];`). This is the only
   place precision is declared — the compiler never silently changes it.
4. **Pipeline expansion** — composite stages are **macros**: `attention(Q,K,V)` expands to
   `scores = matmul_nt(Q,K)` (with the `1/sqrt(D)` scale folded into the GEMM epilogue),
   `P = softmax(scores)`, `O = matmul(P,V)`; `MLP(W1, act, W2, b1, b2)` expands to
   `H = act(matmul(X,W1) + b1)`, `Y = matmul(H,W2) + b2`. So the IR only ever contains
   primitives (`matmul, matmul_nt, bias, relu, gelu, softmax, add, scale, layernorm`) — the
   reference workload becomes **15 IR nodes (8 ops + 7 host tensors)**, `residual_block`
   becomes **20 (10 ops + 10 host tensors)**.
5. **Shape and dtype inference** — bottom-up, with a hard error on any mismatch: attention head
   dims must match, `K` and `V` must have the same number of keys, MLP input features must
   match `W1`, hidden dims must match between `W1`/`W2`, bias sizes must match, matrix inner
   dims must match, arity is checked (e.g. `MLP` takes 3..5 arguments).
6. **DAG construction** — nodes carry shape, storage dtype, accumulation dtype, memory
   placement and dependency edges.

**Error handling policy: fail loudly.** Unknown/unsupported operations are a hard error that
lists what *is* supported — verified live:

```
$ java -jar build/fusionc.jar --input examples/unsupported.dsl --fuse -o build/out.cu
error: unsupported op 'conv2d' -- no emitter exists for it (the compiler refuses to guess);
       supported ops are: attention, MLP, matmul, matmul_nt, bias, relu, gelu, softmax, add,
       scale, layernorm
(exit status 1, no output file written)
```

Syntactic errors behave the same way, with position info: `syntax error at line 12, column 8:
unexpected 'while'`. The parser overrides CUP's `syntax_error`/`unrecovered_syntax_error` to
throw, so **no partial AST is ever handed to the next stage**.

### 2.8 Input and output of a FusionC **program**

| level | syntax | meaning |
|---|---|---|
| program inputs | `input Q : fp16 [SEQ, D];` | host-supplied tensor (the generated harness loads `Q.bin`) |
| program parameters | `param W1 : fp16 [D, F];` | also host-supplied; the distinction is documentation, preserved in the IR |
| program output | `out = ... ;` | every pipeline binds a result tensor; the last pipeline is the graph output |

At run time the generated harness loads one binary file per `input`/`param` (`NAME.bin`, raw
little-endian fp16/fp32 in row-major order) and dumps the output tensor — so "input and output"
is concrete and inspectable, not just conceptual.

### 2.9 Input and output of the **compiler**

| | |
|---|---|
| input | one `.dsl` file (`--input file.dsl`) |
| output | one `.cu` translation unit (`-o file.cu`) |
| optional outputs | `--ir file.txt` (annotated IR dump) · `--decisions file.json` (every fusion decision) · `--dump-ast` (parse tree to stdout) |
| stdout | human-readable decision log + one summary line: `emitted out.cu: 3 kernel(s), 15 IR node(s), 4 decision(s)` |
| exit status | 0 = success, 1 = any front-end/semantic/IO error |

Flags: `--fuse` (default) · `--no-fusion` (baseline: exactly one kernel per op) ·
`--force-fusion` (ignore profitability guards, keep the hard resource caps — used to measure
the decisions the heuristics *rejected*) · `--sm-count <n>` (occupancy floor, 6 = GTX 1050 Ti,
15 = GTX 1070) · `--quiet` · `--help`.

At run time, the generated translation unit is itself a small program with its own I/O:

```
app.exe --inputs <dir> --out <file.bin> --timings <file.csv> --iters 500 --warmup 100 --device 0
```

It prints machine-readable lines the driver parses: `DEVICE`, `KERNELATTR` (regs/smem/spills
from `cudaFuncGetAttributes`), `REGION` (per-kernel-group timing), `TOTAL`, `CHECKSUM`, and
writes the output tensor to `--out`. `--dump-all <dir>` additionally writes **every**
materialised intermediate tensor, which is how the debugging tools check per-tensor error.

---

## 3. Compiler architecture

Pass-by-pass, left to right:

| stage | file(s) | job |
|---|---|---|
| lexical analysis | `src/main/jflex/Scanner.flex` → `Scanner.java` | tokens, positions, lexical errors |
| syntax analysis | `src/main/cup/Parser.cup` → `parser.java`, `sym.java` | LALR(1) tables, AST construction, syntax errors |
| front-end driver | `src/main/java/io/fusionc/frontend/DslParser.java` | wires scanner→parser→AST |
| AST | `src/main/java/io/fusionc/ast/Ast.java` | node classes + dump |
| semantics | `src/main/java/io/fusionc/semantics/GraphBuilder.java` | declarations, macro expansion, shape/dtype inference, DAG |
| IR | `src/main/java/io/fusionc/ir/*.java` | `IrGraph`, `IrNode`, `OpKind`, `Shape`, `DType`, `MemSpace` |
| fusion pass | `src/main/java/io/fusionc/pass/*.java` | candidate grouping, cost model (`ResourceModel`), Pascal rules (`PascalHeuristics`), decision log (`FusionDecision`) |
| back end | `src/main/java/io/fusionc/backend/*.java` | CUDA C emitters: `GemmKernels`, `AttentionKernel`, `MlpKernel`, `ReduceKernels`, `ElementwiseKernels`, `Epilogue`, memory plan + harness (`CudaEmitter`) |
| CLI | `src/main/java/io/fusionc/Main.java` | flags, orchestration, decision log printing/JSON |

The **fusion pass** is the middle end. It walks the DAG and proposes three kinds of groups:

| kind | merges | idea |
|---|---|---|
| `attention_fused` | scores + softmax + PV | reduction-free chain; keeps the score tile out of DRAM |
| `gemm_epilogue` | matmul + bias/activation | the bias/act consumes the accumulator tile **straight out of registers** |
| `gemm_gemm_fused` | matmul + act + matmul | the hidden tile stays in **shared memory** instead of round-tripping through DRAM |

Each candidate is priced by `ResourceModel`: ~48 KB/block shared memory, 255 regs/thread,
2 blocks/SM occupancy floor, grid size, DRAM bytes saved, launches removed. `PascalHeuristics`
holds every tunable as a named constant with a comment explaining the Pascal fact behind it
(the tile-size chooser, the accumulator-register floor, the 64-key attention band, ...).
Every candidate ends in a logged decision, e.g. (real text from the reference workload):

```
REJECT gemm_gemm_fused   out.mlp_fc1 + out.mlp_b1 + out.mlp_act + out.mlp_fc2
       predicted: smem=21.0KB regs=50 occupancy=0.50 grid=32 dram_saved=512.0KB
       reason   : rejected: candidate tile 8x64 leaves only 2 accumulator registers per
                  thread (< 4): the inner loop would be shared-memory bound; keeping 4
                  unfused kernels (the 512.0 KB of DRAM traffic saved does not compensate
                  for the lost parallelism)
```

---

## 4. What the compiler generates

One **self-contained translation unit** per build (`results/attention_mlp/attention_mlp_fused.cu`
is a checked-in example). Its structure:

1. header comment: source name, mode, target, the Pascal constraints baked in;
2. device helpers: warp shuffles (`fx_warp_max/sum`, block reductions), GELU, fp16 load/store
   helpers, `FX_CUDA_CHECK`;
3. the generated kernels (one per fusion region):
   `k_attention_fused_0`, `k_gemm_epilogue_1`, `k_gemm_epilogue_2` for the fused reference
   workload — each with a comment stating its tiling, staging and shared-memory budget;
4. a **static memory plan**: one allocation per tensor, shared-memory stage unions overlaid so
   buffers are recycled between phases;
5. a **host harness**: loads `.bin` inputs, warms up, times every region with CUDA events,
   prints `KERNELATTR` lines from `cudaFuncGetAttributes` (so ptxas predictions can be
   cross-checked against the driver), verifies no launch error, dumps the output.

Compile and run it yourself with what the report uses:

```
nvcc -arch=sm_61 -O3 -Xptxas -v results/attention_mlp/attention_mlp_fused.cu -o attn_fused.exe
```

Design constraints reflected in every emitted kernel: FP16 **storage** with FP32
**accumulation**; no `cp.async` (cooperative global→shared copies + `__syncthreads()`, and in
the fused attention kernel the next chunk's loads are issued *before* the current chunk's FMAs
— the Pascal stand-in for an async copy); warp-level reductions so row reductions never touch
DRAM; padded shared-memory layout so the strided attention reads do not bank-conflict.

---

## 5. Build from source

```powershell
.\scripts\build.ps1
```

The script runs the four build steps:

1. **JFlex** `src/main/jflex/Scanner.flex` → `build/generated/io/fusionc/frontend/Scanner.java`
   (needs `tools/jflex.jar` + the CUP runtime because of `%cup` mode),
2. **CUP** `src/main/cup/Parser.cup` → `parser.java` + `sym.java` (the upstream jar ships no
   `Main-Class`, so `java_cup.Main` is invoked explicitly),
3. **javac** all sources → `build/classes`,
4. **jar** `build/fusionc.jar` with the CUP runtime bundled.

Requirements: JDK 11+ on PATH (or `JAVA_HOME`). The jars already live in `tools/`:
`jflex.jar` (1.9.1), `java-cup.jar` (0.11b), `java-cup-runtime.jar` (11b). A built
`build/fusionc.jar` is already present in this folder.

---

## 6. Live demo: how to prove it works

A short, scriptable demo for the viva. Every command below has been run on this machine.

**Step 1 — compile the DSL (the default is the fused back end):**

```powershell
java -jar build/fusionc.jar --input examples/attention_mlp.dsl --fuse -o build/attn_fused.cu
```

Expected: the decision log on stdout — three FUSED regions and one REJECT, each with predicted
resources and a reason — and the summary line
`emitted build\attn_fused.cu: 3 kernel(s), 15 IR node(s), 4 decision(s)`.

**Step 2 — show the parse tree (statements + expressions):**

```powershell
java -jar build/fusionc.jar --input examples/attention_mlp.dsl --dump-ast --quiet -o build/attn.cu
```

Prints the `Program ... pipeline out = attention(Q, K, V) -> MLP(...)` dump from §2.6.

**Step 3 — show the IR and the decision log as files:**

```powershell
java -jar build/fusionc.jar --input examples/attention_mlp.dsl --fuse `
     --ir build/attn.ir.txt --decisions build/attn.decisions.json -o build/attn_fused.cu
python tools/show_decisions.py build/attn.decisions.json     # pretty-print the decisions
```

**Step 4 — the baseline for comparison (one kernel per op):**

```powershell
java -jar build/fusionc.jar --input examples/attention_mlp.dsl --no-fusion -o build/attn_unfused.cu
```

Expected: `... 8 kernel(s), 15 IR node(s), 0 decision(s)`.

**Step 5 — demonstrate failing loudly (negative test):**

```powershell
java -jar build/fusionc.jar --input examples/unsupported.dsl --fuse -o build/out.cu
echo $LASTEXITCODE   # -> 1
```

Expected message quoted in §2.7; no output file is created.

**Step 6 — regenerate the measurement report** (full GPU evaluation; builds, compiles with
nvcc, runs, validates, writes `docs/REPORT.md` + `results/results.json`, and prints
`[PASS] all 10 variants compiled, ran and matched the fp64 oracle`):

```powershell
python driver/fusionc_driver.py
# or a single workload:
python driver/fusionc_driver.py --workload attention_mlp --iters 300
# or rebuild the report from the stored results without touching the GPU:
python driver/fusionc_driver.py --from-results
```

The driver returns exit status 1 if anything failed to compile, crashed, or disagreed with the
oracle — so "it works" is a machine-checked claim, not a screenshot.

**Step 7 — open the evidence:** `docs/REPORT.md` (the four metrics, in tables),
`results/results.json` (every raw number), `results/<workload>/` (generated `.cu`, `ptxas -v`
log, decision JSON, IR dump, timings CSV, output tensors), and `docs/DSL_SPEC.md` (the language
reference).

---

## 7. Evaluation: the four metrics

Full tables live in [`docs/REPORT.md`](docs/REPORT.md); this is the summary as measured on
this machine (GTX 1070, 16 SMs, sm_61; 500 timed iterations, 100 warm-up; seed 20261025).

### 7.1 Correctness

Every variant is compared **element-wise** against a float64 numpy oracle of the same graph fed
the identical FP16 inputs, with acceptance `|a-b| <= 0.02 + 0.02*|b|` (looser than the ~1e-3
accumulated error, deliberately):

* all **10 compiled variants PASS** — fused, unfused, forced-fusion, and the hand-written
  reference, on all three workloads;
* strongest evidence: generated `--no-fusion` output vs the **independent hand-written CUDA
  reference** on the reference workload is **bit-identical** (`max abs 0.0`), and both match
  the oracle;
* observed error (~3e-4 .. 2e-3) is the FP16 quantisation floor, not an algorithmic difference —
  all accumulation happens in FP32 registers, as the numeric policy demands.

### 7.2 `ptxas -v` statistics

* **Zero spills everywhere**: every kernel reports 0 bytes stack, 0 spill stores, 0 spill
  loads (and the harness's `KERNELATTR` lines, read from `cudaFuncGetAttributes` at run time,
  agree).
* Shared memory peaks at **34,304 B** in the fused attention kernel — 71% of the 48 KB sm_61
  limit, leaving room for 2 resident blocks/SM.
* Every fused kernel's predicted footprint in the decision log **matches the ptxas
  measurement**, because the cost model and the emitters read the same constants.

### 7.3 Runtime (CUDA events, mean per iteration)

| workload | fused | unfused | speedup |
|---|---|---|---|
| `attention_mlp` (64 tokens) | 0.2138 ms | 0.2490 ms | **1.16x** |
| `mlp_wide_batch` (512 tokens) | 0.5562 ms | 0.5920 ms | **1.06x** |
| `residual_block` (LN + GELU MLP) | 0.1874 ms | 0.2440 ms | **1.30x** |

For scale: the hand-written reference on `attention_mlp` runs 0.1594 ms — included as an
independent correctness check first, and as a performance datapoint second; the generated
`--no-fusion` output matches it bit-for-bit but is slower (the reference uses a different,
better-fitting tiling for these small shapes). Closing that gap (split-K, double buffering) is
in [§9](#9-known-limitations-and-future-work).

### 7.4 Fusion decision quality

Each fused region is compared against the sum of the unfused kernels it replaced, measured in
the same session; > +5% is *improved*, < −5% is *degraded*.

| workload | region | fused ms | unfused sum ms | verdict |
|---|---|---|---|---|
| attention_mlp | `attention_fused#0` | 0.0726 | 0.0960 | **improved** (+24.4%) |
| attention_mlp | `gemm_epilogue#1` | 0.0662 | 0.0824 | **improved** (+19.7%) |
| attention_mlp | `gemm_epilogue#2` | 0.1940 | 0.1715 | degraded (−13.1%) |
| mlp_wide_batch | `gemm_epilogue#0` | 0.1707 | 0.1650 | neutral (−3.5%) |
| mlp_wide_batch | `gemm_epilogue#1` | 0.1804 | 0.1718 | degraded (−5.0%) |
| residual_block | `attention_fused#0` | 0.0348 | 0.0832 | **improved** (+58.1%) |
| residual_block | `gemm_epilogue#1` | 0.0463 | 0.0670 | **improved** (+30.9%) |
| residual_block | `gemm_epilogue#2` | 0.1254 | 0.1419 | **improved** (+11.6%) |

**And the rejections were validated by measurement.** Every rejected candidate is rebuilt with
`--force-fusion` and timed anyway; all forced `gemm_gemm_fused` builds are clearly *slower*
than the unfused kernels they would have replaced:

| workload | forced region | forced ms | unfused sum ms | verdict |
|---|---|---|---|---|
| attention_mlp | `gemm_gemm_fused#1` | 0.3849 | 0.2539 | degraded (−51.6%) |
| mlp_wide_batch | `gemm_gemm_fused#0` | 0.9700 | 0.3367 | degraded (−188.0%) |
| residual_block | `gemm_gemm_fused#1` | 0.5400 | 0.2088 | degraded (−158.6%) |

So the rejections are correct calls, not conservative guesses — and the two small-tile
GEMM-epilogue fusions that do measure slightly negative are reported as measured: the verdict
column exists precisely so the report cannot quietly claim more than the data shows. (The
analysis section of the report explains the 64-token shapes: launch overhead dominates at that
size, and a 64×256 output tile is 1,024 threads of work — it cannot fill a 16-SM GPU at all.)

### 7.5 Heuristic sensitivity to the SM count

The occupancy floor (2 blocks/SM) is the one heuristic input that depends on the target part;
`--sm-count 6` (GTX 1050 Ti) vs `15` (GTX 1070) is reported for every workload. Same DSL,
same compiler, different target — the decision log shows where the SM count changes a verdict.

---

## 8. Pascal-specific design decisions

* **FP16 storage, FP32 accumulate.** GP10x runs FP16 arithmetic at 1/64 of the FP32 rate, so
  halves are used for *bandwidth* only; every FMA happens in FP32 registers. Enforced by the
  numeric policy, and by promoting reduction inputs (softmax logits) to FP32 storage.
* **No `cp.async`.** Every staged tile is a cooperative `global → shared` copy followed by
  `__syncthreads()`. Where it matters (fused attention), the next chunk's loads are issued
  *before* the current chunk's FMAs and published with a single barrier per chunk — the Pascal
  stand-in for an async copy.
* **48 KB / 255 registers.** Fused kernels are rejected, with a logged reason, when the
  predicted shared memory or register footprint exceeds the caps, or when the predicted
  occupancy and grid size fall below the occupancy floor. Rejections are decisions too.
* **Warp-level reductions** (`__shfl_down_sync`) for softmax/LayerNorm/attention, so row
  reductions never touch DRAM.
* **Shared-memory bank behaviour.** Attention K rows are padded by two halves so the strided
  key access lands on distinct banks; the V stage is FP32 so the PV reads are conflict-free;
  the K buffer is recycled for V between phases to halve the footprint.
* **GEMM tile chosen per shape.** A 64x64 tile is best when the problem can fill the device,
  but a 64-row GEMM tiled 64x64 runs as a *single block* on a 16-SM part, so the tile chooser
  steps down (64x64 → 64x32 → 32x64 → 32x32) while the grid cannot reach
  `GEMM_TILE_MIN_BLOCKS`. Stepping down further measured slower, so it stops there.
* **All thresholds are named constants** in `io.fusionc.pass.PascalHeuristics`, each with a
  comment explaining the Pascal fact behind the number.

---

## 9. Known limitations and future work

* **No split-K.** Shallow-M GEMMs (64 rows) cannot fill the device with the current tiling. A
  split-K variant with a reduction pass would fix exactly the dominant kernel (`fc2`) in the
  reference workload.
* **Single-buffered GEMM tiles.** GEMM tile loads are issued, then awaited by
  `__syncthreads()`; only the fused attention kernel hides load latency. Register-staged double
  buffering would cost shared memory the 48 KB budget does not have at 64x64 tiles.
* **Attention is single-band.** The fused attention kernel handles up to 64 keys per row
  without online-softmax rescaling; the 512-token workload is correctly *rejected* (with a
  logged reason) and falls back to three unfused kernels. Multi-band attention with rescaling
  is the natural extension.
* **Launch overhead dominates the 64-token workload.** Its cost is mostly per-kernel launch
  overhead (tens of microseconds per launch on Windows/WDDM), and each timed region also pays
  for its own CUDA-event pair — both effects favour fewer kernels. The forced-fusion column
  isolates kernel *efficiency* from launch count.
* **Measurement noise.** WDDM: totals move roughly ±10% between runs, kernel-level ratios are
  stable. Verdicts are therefore computed inside a single session, back to back.
* **FP16 arithmetic is deliberately unused** (1/64 rate on GP10x). On GP100 (1/2 rate) the
  same IR would want a different policy; the policy is a compile-time declaration, not a
  hard-coded assumption.

---

## 10. Repository layout

| path | what |
|---|---|
| `src/main/jflex/Scanner.flex` | JFlex lexer spec (`%cup` mode, returns `java_cup.runtime.Symbol`) |
| `src/main/cup/Parser.cup` | CUP LALR grammar (produces `parser.java` + `sym.java`) |
| `src/main/java/io/fusionc/frontend` | front-end driver (JFlex scanner + CUP parser) |
| `src/main/java/io/fusionc/ast` | AST node classes |
| `src/main/java/io/fusionc/semantics` | declaration processing, macro expansion, shape/dtype inference, DAG build |
| `src/main/java/io/fusionc/ir` | IR: op kind, shape, storage/accumulate dtype, memory placement, dependency edges |
| `src/main/java/io/fusionc/pass` | fusion pass, Pascal heuristics (all tunables), resource cost model, decision log |
| `src/main/java/io/fusionc/backend` | CUDA C emitters: per-op kernels, fused kernels, memory plan, host harness |
| `reference/reference_unfused.cu` | independent hand-written unfused implementation (correctness + perf datapoint) |
| `driver/fusionc_driver.py` | parse → fuse → emit → `nvcc -arch=sm_61` → run → collect metrics → report |
| `examples/*.dsl` | workloads: reference graph, wide batch, residual block, negative test |
| `docs/DSL_SPEC.md` | language reference |
| `docs/REPORT.md` | generated measurement report (correctness, ptxas stats, runtime, verdicts) |
| `docs/analysis.md` | narrative analysis, folded into the generated report |
| `results/` | per-workload artifacts: generated `.cu`, `ptxas -v` logs, decision JSON, IR dumps, timing CSVs, output tensors |
| `tools/` | `device_probe.cu` (device limits), `isolate.py` (per-tensor error vs the oracle), `bisect_launch.py` (first faulting launch), `show_decisions.py`, `show_kernel.py`, `show_named_kernel.py`, `show_results.py`, + the JFlex/CUP jars |
| `scripts/build.ps1` | JFlex + CUP + javac + jar |
| `build/` | generated sources, classes, `fusionc.jar`, device probe |
| `CyberScript/` | separate earlier lexer/parser exercise (see [Appendix A](#appendix-a-the-cyberscript-folder)) |

`results/<workload>/` keeps every generated `.cu`, its nvcc log with `ptxas -v` output, the
decision JSON and the raw output tensors, so any number in the report can be traced back to the
artifact that produced it.

Debugging tools for a miscompile:

```
python tools/isolate.py <workload> fused     # per-tensor error of every materialised tensor
python tools/bisect_launch.py <file.cu>      # sync after each launch, prints the first faulting one
python tools/show_decisions.py <json>        # fusion decisions with predicted resources and reasons
python tools/show_kernel.py <file.cu>        # print one generated kernel
```

---

## 11. Viva cheat sheet

**"What have you built, in one sentence?"**
A source-to-source compiler: it takes a small tensor-pipeline DSL and emits self-contained CUDA C
for Pascal GPUs, fusing producer-consumer operations under a resource cost model whose every
decision is logged, and validated by measuring the rejected candidates too.

**"What are the statements of your language?"**
Exactly one kind: the pipeline binding `name = stage -> stage -> ... ;`, plus the declarations
(`const`, `dtype`, `input`/`param`) that must all precede statements. Rules:
`stmt ::= IDENT EQ stage_list SEMI`, `stage_list ::= stage_list ARROW stage | stage`,
`stage ::= IDENT LPAREN arglist RPAREN | IDENT`. `->` chains left-associatively (left-recursive
rule), feeds each stage's result into the next stage as its first argument, and each bound name
becomes a tensor that later pipelines can consume.

**"What are the expression rules?"**
`expr ::= INT | FLOAT | IDENT | LPAREN expr RPAREN`, with comma-separated `expr_list`s used in
two places: tensor dimensions (`[SEQ, D]`) and op arguments (`attention(Q, K, V)`). No infix
operators → no precedence; parentheses are grouping only. Identifiers resolve semantically: a
constant in a dimension, a tensor (or `relu`/`gelu` in the activation slot) in an argument list.

**"Why does your expression grammar have no precedence?"**
By design — the language describes a dataflow graph, not arithmetic. Computation is named
operations in pipelines; precision is a `dtype` declaration. The only chaining operator is `->`
at statement level, left-associative by the left-recursive `stage_list` rule. (If asked for a
richer one: the older `CyberScript` grammar in the repo has `PLUS/MINUS` and `MULT/DIV` with
`precedence left` declarations — Appendix A.)

**"How does input and output work?"**
Three levels. *In the language*: `input`/`param` declare host tensors, every pipeline binds an
output, the last one is the graph result. *For the compiler*: `.dsl` in, `.cu` out, plus
optional IR dump and decision JSON; exit status 1 on any error. *At run time*: the generated
harness loads one `.bin` per tensor, times every region with CUDA events, prints
`KERNELATTR`/`REGION`/`TOTAL`/`CHECKSUM` lines, writes the output tensor, and can dump every
intermediate tensor with `--dump-all`.

**"Walk me through `out = attention(Q,K,V) -> MLP(...);`"**
JFlex tokenises → CUP parses it as `Pipeline("out", [Call("attention",[Q,K,V]),
Call("MLP",[W1, relu, W2, b1, b2])])` → semantics expands the two macros into 8 primitive ops
with shape/dtype inference (15 IR nodes) → the fusion pass accepts `attention_fused` (scores +
softmax + PV into one kernel) and the two `gemm_epilogue` fusions, rejects `gemm_gemm_fused`
with a reason → the back end emits 3 kernels + memory plan + harness; with `--no-fusion` it is
8 kernels instead.

**"What errors does it catch?"**
Illegal characters; syntax errors with line/column (no partial AST — the parser throws);
duplicate constants/tensors; non-positive constants; unknown constant/tensor; non-integer or
wrong-rank dimensions; unsupported ops (with the supported list); shape mismatches everywhere
(attention head dims, K/V key counts, MLP feature/hidden dims, bias sizes, matmul inner dims);
wrong argument counts; unsupported activations. All of them: hard error, exit 1, no output
file — fail loudly, never guess.

**"How do you know the generated CUDA is correct?"**
Three independent checks: (1) every variant vs a float64 numpy oracle of the same graph fed
identical FP16 inputs, element-wise — all 10 variants PASS; (2) generated `--no-fusion` vs an
independently hand-written CUDA reference — bit-identical (max abs 0.0); (3) per-tensor
`isolate.py` against the oracle for every materialised intermediate. Plus the driver exits
non-zero if *anything* fails, so the PASS is machine-checked.

**"Fusion sounds great — why reject anything?"**
Because fusing costs resources: a fused kernel with too small a row tile re-streams both weight
matrices, and a tile that leaves fewer than 4 accumulators per thread is shared-memory-bound.
The heuristics reject `gemm_gemm_fused` at these shapes — and we prove the rejection was right
by compiling it anyway with `--force-fusion`: it is 51.6% / 188% / 158.6% slower than the
unfused kernels. Rejections are validated, not assumed.

**"What is `--sm-count` for?"**
The occupancy floor is 2 blocks/SM, so the target's SM count participates in the decision:
6 (GTX 1050 Ti, the brief) vs 15 (GTX 1070, this machine). The report prints the decision log
for both settings.

**"Why JFlex and CUP?"**
They are the course toolchain, and they fit: JFlex `%cup` mode makes the scanner speak
`java_cup.runtime.Scanner` directly; CUP gives an LALR(1) parser with error hooks we override
to fail loudly; both are generated at build time from the specs in `src/main/`.

**"What is the biggest limitation?"**
The dominant kernel in the 64-token workload (the `fc2` GEMM, 1,024 threads of work) cannot
fill a 16-SM GPU at this size — no tiling fixes that; split-K would. And attention is
single-band (≤64 keys/row) by design, with multi-band online softmax as the natural extension.

---

## Appendix A: the `CyberScript/` folder

`CyberScript/` is a **separate, earlier lexer/parser exercise** — not part of the CUDA
compiler pipeline — written in the same JFlex/CUP stack. It defines a little network-security
scripting language:

* **Statements**: `NETWORK id`, `DEVICE id`, `SET id = expr`, `SCAN id | SCAN PORT n`,
  `ALERT "msg" | ALERT ADMIN`, `LOG "msg"`, `BLOCK SOURCE`, `REPORT`, and the block
  `IF condition stmt_list END` with conditions `PORT n OPEN|CLOSED`,
  `ATTACK DDOS|BRUTEFORCE|PORTSCAN DETECTED`, or `expr relop expr`.
* **Expressions**: `expr ::= expr PLUS|MINUS|MULT|DIV expr | (expr) | ID | NUMBER | TRAFFIC`
  with real operator precedence declared in the CUP file
  (`precedence left PLUS, MINUS;` then `precedence left MULT, DIV;` — so `MULT/DIV` binds
  tighter, and both are left-associative), plus relational operators `> < >= <= == !=`.
* `Cup/` contains the full pipeline (`lexer.jflex`, `parser.cup`, `Main.java`, sample
  `test.cys`); `Standalone/` is a `%standalone` lexer-only demo that prints one line per
  recognised token, with multi-line comments and explicit unterminated-string/comment errors,
  driven by `input.txt`.
* It validates and prints (`Syntax valid`, `IF statement parsed`, ...) — it has no semantics or
  code generation; that is what the main compiler adds.

Known inconsistency, worth fixing before anyone runs it: `Cup/test.cys` says `IF PORT OPEN`
but the grammar requires a port number (`PORT NUMBER OPEN`), so that sample would be rejected
as written; `Standalone/input.txt` is the consistent full sample.

## Appendix B: toolchain notes (this machine)

* **CUDA 11.8 is required**: CUDA 13 removed `sm_61` code generation entirely, and `nvprof`
  only exists in the 11.x line.
* nvcc 11.8 cannot host-compile with the VS 18 / MSVC 14.51 STL (`STL1002: Unexpected compiler
  version`), so the driver pins the host toolset to `vcvars64.bat -vcvars_ver=14.34` — the
  newest toolset CUDA 11.8 supports.
* The evaluation machine has a **GTX 1070** (same `sm_61` ISA, 15/16 SMs instead of the brief's
  6-SM GTX 1050 Ti), so the driver reports both SM settings and the report shows where the SM
  count changes a fusion decision.
* Measured environment: OpenJDK 21, nvcc 11.8 (`-arch=sm_61 -O3 -Xptxas -v`), MSVC 14.34,
  Windows SDK 10.0.22621, 500 timed iterations / 100 warm-up, seed 20261025.
