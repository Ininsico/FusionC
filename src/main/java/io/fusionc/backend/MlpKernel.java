package io.fusionc.backend;

import io.fusionc.ir.IrNode;
import io.fusionc.ir.OpKind;
import io.fusionc.pass.FusionGroup;
import io.fusionc.pass.PascalHeuristics;

import java.util.List;

/**
 * Fused {@code GEMM -> activation -> GEMM} (the MLP of a transformer block).
 *
 * <p>The hidden tile is produced directly into shared memory and consumed from there, so the
 * {@code [rows x hidden]} intermediate never reaches DRAM - that round trip is exactly what the
 * fusion buys. Phase 1 computes {@code X = act(A x W1 + b1)} into shared memory over hidden
 * chunks; phase 2 accumulates {@code Y += X x W2} into registers. Every load is a cooperative
 * global -> shared copy followed by {@code __syncthreads()}, the Pascal substitute for cp.async.
 */
final class MlpKernel {

    private MlpKernel() {
    }

    static void emit(CudaWriter w, FusionGroup group, EmitContext ctx) {
        IrNode fc1 = group.members.get(0);
        int fc2Index = -1;
        for (int i = 1; i < group.members.size(); i++) {
            if (group.members.get(i).op == OpKind.MATMUL) {
                fc2Index = i;
                break;
            }
        }
        if (fc2Index < 0) {
            throw new IllegalStateException("gemm_gemm group " + group.id + " has no consumer matmul");
        }
        IrNode fc2 = group.members.get(fc2Index);
        List<IrNode> middle = group.members.subList(1, fc2Index);
        List<IrNode> tail = group.members.subList(fc2Index + 1, group.members.size());
        IrNode out = group.external.get(group.external.size() - 1);
        IrNode a = fc1.inputs.get(0);
        IrNode w1 = fc1.inputs.get(1);
        IrNode w2 = fc2.inputs.get(1);

        int m = fc1.shape.rows;
        int d = a.shape.cols;
        int hidden = fc1.shape.cols;
        int n = fc2.shape.cols;
        int rm = group.params.get("rm");
        int cn = group.params.get("cn");
        int bk = group.params.get("bk");
        int hf = group.params.get("hf");
        int threads = group.params.get("threads");

        int colGroups = hf / 4;
        int subTiles = (rm / 4) * colGroups;
        int yColGroups = cn / 4;
        int ySlots = (rm / 2) * yColGroups;
        if (subTiles > threads || ySlots > threads || hf > threads) {
            throw new IllegalStateException("fused MLP tile " + rm + "x" + cn + " hidden " + hf
                    + " does not map onto " + threads + " threads (subTiles=" + subTiles
                    + ", ySlots=" + ySlots + ")");
        }
        if (m % rm != 0 || n % cn != 0 || d % bk != 0 || hidden % hf != 0) {
            throw new IllegalStateException("fused MLP tile " + rm + "x" + cn + " hidden " + hf
                    + " does not divide the problem " + m + "x" + n + " k=" + d + " hidden=" + hidden);
        }

        w.line("// kernel %d: gemm_gemm_fused -- %s x %s -> %s -> %s x %s", group.id, fc1.name, w1.name,
                middle.isEmpty() ? "(no activation)" : describe(middle), fc2.name, w2.name);
        w.line("//   Y[%dx%d] = act(A[%dx%d] x W1[%dx%d] + b1) x W2[%dx%d] ; hidden stays in shared memory",
                m, n, m, d, d, hidden, hidden, n);
        w.line("//   tile %dx%d, hidden chunk %d, K step %d, %d threads", rm, cn, hf, bk, threads);
        w.line("__global__ void %s(%s) {",
                FusionGroup.kernelName(group.kind, group.id, group.members),
                String.join(", ", ctx.kernelParams(group)));
        w.in();
        w.line("__shared__ __half As[%d][%d];", rm, bk);
        w.line("__shared__ __half W1s[%d][%d];", bk, hf);
        w.line("__shared__ __half Xs[%d][%d];", rm, hf);
        w.line("__shared__ __half W2s[%d][%d];", hf, cn);
        if (hasBias(middle)) {
            w.line("__shared__ float b1s[%d];", hf);
        }
        w.line("const int rowBase = blockIdx.y * %d;", rm);
        w.line("const int colBase = blockIdx.x * %d;", cn);

        w.line("// thread -> hidden sub-tile mapping (4x4 micro-tile per thread)");
        w.line("const int st = threadIdx.x;");
        w.line("const int h0 = (st - (st / %d) * %d) * 4;", colGroups, colGroups);
        w.line("const int r0 = (st / %d) * 4;", colGroups);
        w.line("const bool hiddenActive = st < %d;", subTiles);
        w.line("// thread -> output sub-tile mapping (2 rows x 4 columns per thread)");
        w.line("const int yst = threadIdx.x;");
        w.line("const int cY0 = (yst - (yst / %d) * %d) * 4;", yColGroups, yColGroups);
        w.line("const int rY0 = (yst / %d) * 2;", yColGroups);
        w.line("const bool outActive = yst < %d;", ySlots);
        w.line("float accY[2][4];");
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 2; ++i) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) accY[i][j] = 0.0f;");
        w.out();
        w.line("}");
        w.blank();
        w.line("for (int hc = 0; hc < %d; hc += %d) {", hidden, hf);
        w.in();
        w.line("// ---- phase 1: hidden tile = act(A x W1 + b1), written straight into shared memory ----");
        if (hasBias(middle)) {
            IrNode bias = biasOf(middle);
            w.line("if (threadIdx.x < %d) b1s[threadIdx.x] = %s;", hf,
                    ctx.loadScalar(bias, "hc + threadIdx.x"));
        }
        w.line("float part[4][4];");
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 4; ++i) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) part[i][j] = 0.0f;");
        w.out();
        w.line("}");
        w.line("#pragma unroll");
        w.line("for (int k0 = 0; k0 < %d; k0 += %d) {", d, bk);
        w.in();
        w.line("for (int t = threadIdx.x; t < %d; t += %d) {", rm * bk, threads);
        w.in();
        w.line("const int r = t / %d, c = t - (t / %d) * %d;", bk, bk, bk);
        w.line("As[r][c] = %s;", ctx.loadScalarAsHalf(a, "(rowBase + r) * " + d + " + (k0 + c)"));
        w.out();
        w.line("}");
        w.line("for (int t = threadIdx.x; t < %d; t += %d) {", bk * hf, threads);
        w.in();
        w.line("const int r = t / %d, c = t - (t / %d) * %d;", hf, hf, hf);
        w.line("W1s[r][c] = %s;", ctx.loadScalarAsHalf(w1, "(k0 + r) * " + hidden + " + (hc + c)"));
        w.out();
        w.line("}");
        w.line("__syncthreads();");
        w.line("if (hiddenActive) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int kk = 0; kk < %d; ++kk) {", bk);
        w.in();
        w.line("float av[4], wv[4];");
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 4; ++i) av[i] = __half2float(As[r0 + i][kk]);");
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) wv[j] = __half2float(W1s[kk][h0 + j]);");
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 4; ++i) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) part[i][j] = fmaf(av[i], wv[j], part[i][j]);");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.line("__syncthreads();");
        w.out();
        w.line("}");
        w.line("// bias + activation applied on the way into shared memory (never through DRAM)");
        w.line("if (hiddenActive) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 4; ++i) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) {");
        w.in();
        String value = "part[i][j]";
        for (IrNode node : middle) {
            if (node.op == OpKind.BIAS_ADD) {
                value = "(" + value + " + b1s[h0 + j])";
            } else {
                value = Epilogue.unary(node.op, value, node.scale());
            }
        }
        w.line("Xs[r0 + i][h0 + j] = __float2half(%s);", value);
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.line("__syncthreads();");
        w.blank();
        w.line("// ---- phase 2: Y += hidden x W2 ----");
        w.line("for (int t = threadIdx.x; t < %d; t += %d) {", hf * cn, threads);
        w.in();
        w.line("const int r = t / %d, c = t - (t / %d) * %d;", cn, cn, cn);
        w.line("W2s[r][c] = %s;", ctx.loadScalarAsHalf(w2, "(hc + r) * " + n + " + (colBase + c)"));
        w.out();
        w.line("}");
        w.line("__syncthreads();");
        w.line("if (outActive) {");
        w.in();
        w.line("for (int h = 0; h < %d; ++h) {", hf);
        w.in();
        w.line("const float x0 = __half2float(Xs[rY0][h]);");
        w.line("const float x1 = __half2float(Xs[rY0 + 1][h]);");
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) {");
        w.in();
        w.line("const float wj = __half2float(W2s[h][cY0 + j]);");
        w.line("accY[0][j] = fmaf(x0, wj, accY[0][j]);");
        w.line("accY[1][j] = fmaf(x1, wj, accY[1][j]);");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.line("__syncthreads();");
        w.out();
        w.line("}");
        w.blank();
        w.line("// ---- store Y with the trailing epilogue ----");
        w.line("if (outActive) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < 2; ++i) {");
        w.in();
        w.line("#pragma unroll");
        w.line("for (int j = 0; j < 4; ++j) {");
        w.in();
        w.line("const int row = rowBase + rY0 + i;");
        w.line("const int col = colBase + cY0 + j;");
        String store = "accY[i][j]";
        for (IrNode node : tail) {
            if (node.op == OpKind.BIAS_ADD) {
                IrNode bias = node.inputs.get(1);
                store = "(" + store + " + " + ctx.loadScalar(bias, Epilogue.biasIndex(bias, "col")) + ")";
            } else if (node.op == OpKind.ADD) {
                store = "(" + store + " + " + ctx.loadScalar(node.inputs.get(1), "row * " + n + " + col") + ")";
            } else {
                store = Epilogue.unary(node.op, store, node.scale());
            }
        }
        w.line(ctx.storeScalar(out, "row * " + n + " + col", store));
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.blank();
    }

    private static boolean hasBias(List<IrNode> middle) {
        for (IrNode node : middle) {
            if (node.op == OpKind.BIAS_ADD) {
                return true;
            }
        }
        return false;
    }

    private static IrNode biasOf(List<IrNode> middle) {
        for (IrNode node : middle) {
            if (node.op == OpKind.BIAS_ADD) {
                return node.inputs.get(1);
            }
        }
        throw new IllegalStateException("no bias in chain");
    }

    private static String describe(List<IrNode> middle) {
        StringBuilder sb = new StringBuilder();
        for (IrNode node : middle) {
            sb.append(node.op.name().toLowerCase()).append(' ');
        }
        return sb.toString().trim();
    }

    static LaunchConfig launch(FusionGroup group, EmitContext ctx) {
        IrNode fc1 = group.members.get(0);
        int rm = group.params.get("rm");
        int cn = group.params.get("cn");
        return new LaunchConfig(LaunchConfig.ceilDiv(fc1.shape.cols == 0 ? 1 : maxCols(group), cn),
                LaunchConfig.ceilDiv(fc1.shape.rows, rm), group.params.get("threads"), 0);
    }

    private static int maxCols(FusionGroup group) {
        for (IrNode node : group.members) {
            if (node.op == OpKind.MATMUL && node != group.members.get(0)) {
                return node.shape.cols;
            }
        }
        return group.members.get(0).shape.cols;
    }
}
