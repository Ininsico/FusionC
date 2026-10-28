package io.fusionc.backend;

import io.fusionc.ir.IrNode;
import io.fusionc.ir.OpKind;
import io.fusionc.pass.FusionGroup;
import io.fusionc.pass.PascalHeuristics;

import java.util.List;

/**
 * Tiled GEMM emitter. Used for standalone matmuls ({@code --no-fusion}) and for the
 * epilogue-fused variant ({@code --fuse}): the bias / activation chain is applied to the
 * accumulator tile while it is still in registers, so the intermediate never reaches DRAM.
 *
 * <p>Tile: {@code BM x BN} chosen by the fusion pass from {@code PascalHeuristics.GEMM_TILES}
 * (largest tile whose grid still fills the device), BK=32 depth step, 4x4 micro-tile per thread.
 * Operands are staged through shared memory as FP16 and converted to FP32 before every FMA --
 * GP10x executes FP16 arithmetic at 1/64 of the FP32 rate, so halves are for bandwidth only.
 */
final class GemmKernels {

    private GemmKernels() {
    }

    static void emit(CudaWriter w, FusionGroup group, EmitContext ctx) {
        IrNode producer = group.members.get(0);
        List<IrNode> chain = EmitContext.epilogueAfter(group, producer);
        IrNode out = group.external.get(group.external.size() - 1);
        IrNode a = producer.inputs.get(0);
        IrNode b = producer.inputs.get(1);

        int m = producer.shape.rows;
        int n = producer.shape.cols;
        int k = a.shape.cols;
        int bm = group.params.getOrDefault("bm", PascalHeuristics.GEMM_BM);
        int bn = group.params.getOrDefault("bn", PascalHeuristics.GEMM_BN);
        int bk = group.params.getOrDefault("bk", PascalHeuristics.GEMM_BK);
        int threads = group.params.getOrDefault("threads", PascalHeuristics.GEMM_THREADS);
        boolean transB = producer.transB();
        boolean colTail = n % bn != 0;
        boolean rowTail = m % bm != 0;
        boolean kTail = k % bk != 0;
        double scale = producer.scale();

        StringBuilder ops = new StringBuilder();
        for (IrNode node : chain) {
            ops.append(ops.length() == 0 ? "" : "+").append(node.op.name().toLowerCase());
        }
        w.line("// kernel %d: %s", group.id, group.kind.label);
        w.line("//   %s = %s%s  (%dx%d = %dx%d %s %s)", out.name, producer.name,
                chain.isEmpty() ? "" : " -> " + ops, m, n, m, k, transB ? "x transpose" : "x",
                transB ? "(" + n + "x" + k + ")" : "(" + k + "x" + n + ")");
        w.line("//   tile %dx%d, depth step %d, %d threads, 4x4 per thread", bm, bn, bk, threads);
        w.line("__global__ void %s(%s) {",
                FusionGroup.kernelName(group.kind, group.id, group.members),
                String.join(", ", ctx.kernelParams(group)));
        w.in();
        w.line("__shared__ __half smA[%d][%d];", bm, bk);
        w.line("__shared__ __half smB[%d][%d];", bk, bn);
        w.line("const int tx = threadIdx.x %% %d;   // %d column groups of 4", bn / 4, bn / 4);
        w.line("const int ty = threadIdx.x / %d;    // %d row groups of 4", bn / 4, bm / 4);
        w.line("const int rowBase = blockIdx.y * %d;", bm);
        w.line("const int colBase = blockIdx.x * %d;", bn);
        w.line("float acc[4][4];");
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 4; ++i) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;");
        w.out();
        w.line("}");
        w.blank();
        w.line("#pragma unroll");
        w.line("for (int k0 = 0; k0 < %d; k0 += %d) {", k, bk);
        w.in();
        w.line("// cooperative tile load. Pascal has no cp.async, so global->shared is a plain");
        w.line("// load/store pair guarded by __syncthreads() instead of an async copy.");
        if ((bm * bk) % threads != 0 || (bk * bn) % threads != 0) {
            throw new IllegalStateException("tile " + bm + "x" + bn + " with BK=" + bk + " and " + threads
                    + " threads needs " + (bm * bk) + "/" + (bk * bn)
                    + " elements per tile, which is not a multiple of the block size");
        }
        w.line("#pragma unroll");
        w.line("for (int t = 0; t < %d; ++t) {   // A tile: %d elements", (bm * bk) / threads, bm * bk);
        w.in();
        w.line("const int idx = threadIdx.x + t * %d;", threads);
        w.line("const int ar = idx / %d;", bk);
        w.line("const int ac = idx - ar * %d;", bk);
        if (kTail || rowTail) {
            w.line("smA[ar][ac] = __float2half(0.0f);");
            String guard = (kTail ? "k0 + ac < " + k : "1")
                    + (rowTail ? " && rowBase + ar < " + m : "");
            w.line("if (%s) smA[ar][ac] = %s;", guard,
                    ctx.loadScalarAsHalf(a, "(rowBase + ar) * " + k + " + (k0 + ac)"));
        } else {
            w.line("smA[ar][ac] = %s;", ctx.loadScalarAsHalf(a, "(rowBase + ar) * " + k + " + (k0 + ac)"));
        }
        w.out();
        w.line("}");
        w.line("#pragma unroll");
        w.line("for (int t = 0; t < %d; ++t) {   // B tile: %d elements", (bk * bn) / threads, bk * bn);
        w.in();
        w.line("const int idx = threadIdx.x + t * %d;", threads);
        if (transB) {
            // B is [N,K]: read the k index from the fast-varying lanes so the global load is coalesced
            w.line("const int bmRow = idx - (idx / %d) * %d;   // k index inside the tile", bk, bk);
            w.line("const int bmCol = idx / %d;                  // n index inside the tile", bk);
        } else {
            w.line("const int bmRow = idx / %d;                  // k index inside the tile", bn);
            w.line("const int bmCol = idx - bmRow * %d;          // n index inside the tile", bn);
        }
        String bIndex = transB
                ? "(colBase + bmCol) * " + k + " + (k0 + bmRow)"
                : "(k0 + bmRow) * " + n + " + (colBase + bmCol)";
        if (kTail || colTail) {
            w.line("smB[bmRow][bmCol] = __float2half(0.0f);");
            String guard = (kTail ? "k0 + bmRow < " + k : "1")
                    + (colTail ? " && colBase + bmCol < " + n : "");
            w.line("if (%s) smB[bmRow][bmCol] = %s;", guard, ctx.loadScalarAsHalf(b, bIndex));
        } else {
            w.line("smB[bmRow][bmCol] = %s;", ctx.loadScalarAsHalf(b, bIndex));
        }
        w.out();
        w.line("}");
        w.line("__syncthreads();");
        w.blank();
        w.line("#pragma unroll");
        w.line("for (int kk = 0; kk < %d; ++kk) {", bk);
        w.in();
        w.line("float av[4], bv[4];");
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 4; ++i) av[i] = __half2float(smA[ty * 4 + i][kk]);");
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) bv[j] = __half2float(smB[kk][tx * 4 + j]);");
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 4; ++i) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.line("__syncthreads();");
        w.out();
        w.line("}");
        w.blank();
        w.line("// epilogue: %s", chain.isEmpty() && scale == 1.0 ? "store" : buildDescription(chain, scale));
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 4; ++i) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) {");
        w.in();
        w.line("const int row = rowBase + ty * 4 + i;");
        w.line("const int col = colBase + tx * 4 + j;");
        if (colTail) {
            w.line("if (col >= %d) continue;", n);
        }
        if (rowTail) {
            w.line("if (row >= %d) continue;", m);
        }
        String value = "acc[i][j]";
        if (scale != 1.0) {
            value = "(" + value + " * " + Epilogue.fmt(scale) + "f)";
        }
        for (IrNode op : chain) {
            if (op.op == OpKind.BIAS_ADD) {
                IrNode bias = op.inputs.get(1);
                value = "(" + value + " + " + ctx.loadScalar(bias, Epilogue.biasIndex(bias, "col")) + ")";
            } else if (op.op == OpKind.ADD) {
                value = "(" + value + " + " + ctx.loadScalar(op.inputs.get(1), "row * " + n + " + col") + ")";
            } else {
                value = Epilogue.unary(op.op, value, op.scale());
            }
        }
        w.line(ctx.storeScalar(out, "row * " + n + " + col", value));
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.blank();
    }

    private static String buildDescription(List<IrNode> chain, double scale) {
        StringBuilder sb = new StringBuilder();
        if (scale != 1.0) {
            sb.append("scale(").append(Epilogue.fmt(scale)).append(") ");
        }
        for (IrNode node : chain) {
            sb.append(node.op.name().toLowerCase()).append(' ');
        }
        return sb.toString().trim();
    }

    static LaunchConfig launch(FusionGroup group, EmitContext ctx) {
        IrNode producer = group.members.get(0);
        int bm = group.params.getOrDefault("bm", PascalHeuristics.GEMM_BM);
        int bn = group.params.getOrDefault("bn", PascalHeuristics.GEMM_BN);
        int threads = group.params.getOrDefault("threads", PascalHeuristics.GEMM_THREADS);
        return new LaunchConfig(
                LaunchConfig.ceilDiv(producer.shape.cols, bn),
                LaunchConfig.ceilDiv(producer.shape.rows, bm),
                threads, 0);
    }
}
