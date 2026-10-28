package io.fusionc.backend;

import io.fusionc.ir.IrNode;
import io.fusionc.pass.FusionGroup;
import io.fusionc.pass.PascalHeuristics;

/**
 * Fused attention: {@code QK^T/sqrt(D) -> softmax -> PV} in a single kernel.
 *
 * <p>Pascal-specific design points:
 * <ul>
 *   <li>one warp per query row: the whole row reduction is a {@code __shfl_down} tree and the
 *       normalisation never touches DRAM;</li>
 *   <li>K/V are staged through shared memory in dimension chunks with an explicit two-stage
 *       pipeline. sm_61 has no {@code cp.async}, so the next chunk's <em>plain global loads are
 *       issued before the current chunk's FMAs</em> and a single {@code __syncthreads()} per
 *       chunk publishes them - that is the Pascal substitute for an async copy;</li>
 *   <li>the K buffer is recycled for V (different phases) through a union, halving the footprint:
 *       K rows are FP16 with a 2-half pad so the strided key access hits distinct banks, V rows
 *       are FP32 so the PV reads are bank-conflict free;</li>
 *   <li>scores stay in registers, probabilities are exchanged through shared memory.</li>
 * </ul>
 */
final class AttentionKernel {

    private AttentionKernel() {
    }

    static final int PAD = 2;

    static void emit(CudaWriter w, FusionGroup group, EmitContext ctx) {
        IrNode scores = group.members.get(0);
        IrNode pv = group.members.get(2);
        IrNode q = scores.inputs.get(0);
        IrNode k = scores.inputs.get(1);
        IrNode v = pv.inputs.get(1);

        int b = (int) scores.shape.rows;
        // head dim comes from Q, not from the [B x B] score matrix
        int d = q.shape.cols;
        int rpb = group.params.get("rowsPerBlock");
        int dc = group.params.get("dimChunk");
        int threads = rpb * PascalHeuristics.WARP_SIZE;
        int nchunk = d / dc;
        int perChunk = dc / 32;
        int slots = d / 32;
        double scale = scores.scale();
        boolean tail = b < 64;
        int khalfs = b * (dc + PAD);
        int vfloats = b * dc;

        w.line("// kernel %d: attention_fused -- QK^T*scale -> softmax -> PV, one kernel", group.id);
        w.line("//   %d query rows x %d head dim; %d rows/block (1 warp each), %d dimension chunks of %d",
                b, d, rpb, nchunk, dc);
        w.line("//   shared per stage: K %d halves (padded) overlaid with V %d floats, x2 stages",
                khalfs, vfloats);
        w.line("typedef union { __half h[%d]; float f[%d]; } fx_stage%d_t;", khalfs, vfloats, group.id);
        w.line("__global__ void %s(%s) {",
                FusionGroup.kernelName(group.kind, group.id, group.members),
                String.join(", ", ctx.kernelParams(group)));
        w.in();
        w.line("__shared__ fx_stage%d_t St[2];", group.id);
        w.line("__shared__ __half Qs[%d][%d];", rpb, d);
        w.line("__shared__ float  Pr[%d][%d];", rpb, b);
        w.line("const int lane = threadIdx.x & 31;");
        w.line("const int warp = threadIdx.x >> 5;");
        w.line("const int row = blockIdx.x * %d + warp;", rpb);
        w.line("const int j0 = lane;        // keys this lane scores: j0 and j1");
        w.line("const int j1 = lane + 32;");
        w.line("float acc[%d];", slots);
        w.line("#pragma unroll");
        w.line("for (int i = 0; i < %d; ++i) acc[i] = 0.0f;", slots);
        w.line("float invSum = 0.0f;");
        w.blank();
        w.line("if (row < %d) {   // stage this warp's query row once, reused by every chunk", b);
        w.in();
        w.line("for (int dd = lane; dd < %d; dd += 32) Qs[warp][dd] = %s;", d,
                ctx.loadScalarAsHalf(q, "row * " + d + " + dd"));
        w.out();
        w.line("}");
        w.blank();
        w.line("// ------- phase 1: S = Q x K^T (register accumulation over dimension chunks) -------");
        emitStage(w, ctx, k, "St[0].h", b, d, dc, dc + PAD, threads, "0", false);
        w.line("__syncthreads();");
        w.line("float sc0 = 0.0f, sc1 = 0.0f;   // accumulate from zero: -INFINITY would poison the sum");
        w.line("for (int c = 0; c < %d; ++c) {", nchunk);
        w.in();
        w.line("const int cur = c & 1, nxt = (c + 1) & 1;");
        w.line("if (c + 1 < %d) {", nchunk);
        w.in();
        w.line("// issue the next K chunk before the FMAs below: Pascal has no cp.async, so the");
        w.line("// global latency is hidden behind this chunk's arithmetic.");
        emitStage(w, ctx, k, "St[nxt].h", b, d, dc, dc + PAD, threads, "(c + 1) * " + dc, false);
        w.out();
        w.line("}");
        w.line("if (row < %d) {", b);
        w.in();
        w.line("#pragma unroll");
        w.line("for (int dd = 0; dd < %d; ++dd) {", dc);
        w.in();
        w.line("const float qv = __half2float(Qs[warp][c * %d + dd]);", dc);
        if (tail) {
            w.line("if (j0 < %d) sc0 += qv * __half2float(St[cur].h[j0 * %d + dd]);", b, dc + PAD);
            w.line("if (j1 < %d) sc1 += qv * __half2float(St[cur].h[j1 * %d + dd]);", b, dc + PAD);
        } else {
            w.line("sc0 += qv * __half2float(St[cur].h[j0 * %d + dd]);", dc + PAD);
            w.line("sc1 += qv * __half2float(St[cur].h[j1 * %d + dd]);", dc + PAD);
        }
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.line("__syncthreads();");
        w.out();
        w.line("}");
        w.blank();
        w.line("// ------- softmax: warp shfl_down max, exp, warp shfl_down sum -------");
        w.line("if (row < %d) {", b);
        w.in();
        if (tail) {
            w.line("// keys beyond the sequence length must not contribute to max/sum: mask them to");
            w.line("// -INFINITY *after* the accumulation (exp(-inf - m) is exactly 0)");
            w.line("if (j0 >= %d) sc0 = -INFINITY;", b);
            w.line("if (j1 >= %d) sc1 = -INFINITY;", b);
        }
        w.line("sc0 *= %sf;", Epilogue.fmt(scale));
        w.line("sc1 *= %sf;", Epilogue.fmt(scale));
        w.line("float m = fx_warp_max(fmaxf(sc0, sc1));");
        w.line("m = __shfl_sync(0xffffffffu, m, 0);");
        w.line("const float p0 = expf(sc0 - m);");
        w.line("const float p1 = expf(sc1 - m);");
        w.line("Pr[warp][j0] = p0;");
        w.line("Pr[warp][j1] = p1;");
        w.line("float s = fx_warp_sum(p0 + p1);");
        w.line("s = __shfl_sync(0xffffffffu, s, 0);");
        w.line("invSum = 1.0f / s;");
        w.line("__syncwarp();   // P is consumed by all lanes of this warp in phase 2");
        w.out();
        w.line("}");
        w.blank();
        w.line("// ------- phase 2: O = P x V (same pipeline, the stage buffer now carries V as FP32) -------");
        w.line("__syncthreads();");
        emitStage(w, ctx, v, "St[0].f", b, d, dc, dc, threads, "0", true);
        w.line("__syncthreads();");
        w.line("for (int c = 0; c < %d; ++c) {", nchunk);
        w.in();
        w.line("const int cur = c & 1, nxt = (c + 1) & 1;");
        w.line("if (c + 1 < %d) {", nchunk);
        w.in();
        emitStage(w, ctx, v, "St[nxt].f", b, d, dc, dc, threads, "(c + 1) * " + dc, true);
        w.out();
        w.line("}");
        w.line("if (row < %d) {", b);
        w.in();
        w.line("for (int j = 0; j < %d; ++j) {", b);
        w.in();
        w.line("const float p = Pr[warp][j];");
        if (perChunk == 2) {
            w.line("acc[2 * c]     = fmaf(p, St[cur].f[j * %d + lane], acc[2 * c]);", dc);
            w.line("acc[2 * c + 1] = fmaf(p, St[cur].f[j * %d + lane + 32], acc[2 * c + 1]);", dc);
        } else {
            w.line("acc[c] = fmaf(p, St[cur].f[j * %d + lane], acc[c]);", dc);
        }
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.line("__syncthreads();");
        w.out();
        w.line("}");
        w.blank();
        w.line("// ------- store O, normalised by the softmax denominator -------");
        w.line("if (row < %d) {", b);
        w.in();
        w.line("#pragma unroll");
        w.line("for (int c = 0; c < %d; ++c) {", nchunk);
        w.in();
        if (perChunk == 2) {
            w.line(ctx.storeScalar(pv, "row * " + d + " + c * " + dc + " + lane", "acc[2 * c] * invSum"));
            w.line(ctx.storeScalar(pv, "row * " + d + " + c * " + dc + " + lane + 32", "acc[2 * c + 1] * invSum"));
        } else {
            w.line(ctx.storeScalar(pv, "row * " + d + " + c * " + dc + " + lane", "acc[c] * invSum"));
        }
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.out();
        w.line("}");
        w.blank();
    }

    /**
     * Cooperative global -> shared staging of one [B keys x dimChunk] slice.
     *
     * <p>FP16 operands with a 16-byte-aligned row start are copied eight halves at a time (one
     * 16-byte transaction per thread instead of eight 2-byte ones). Trade-off: the K tile keeps
     * its 2-half row padding to stay bank-conflict free, which breaks 16-byte alignment inside
     * shared memory, so the wide load is followed by eight scalar shared stores. The count of
     * latency-exposed global transactions drops 8x, which is what matters on a part with no
     * cp.async.
     */
    private static void emitStage(CudaWriter w, EmitContext ctx, IrNode src, String dstArray,
                                  int b, int d, int dc, int rowStride, int threads,
                                  String chunkBase, boolean convert) {
        boolean vector = src.store == io.fusionc.ir.DType.FP16 && d % 8 == 0 && dc % 8 == 0;
        if (vector) {
            w.line("for (int i = threadIdx.x; i < %d; i += %d) {   // 8 halves per global transaction",
                    (b * dc) / 8, threads);
            w.in();
            w.line("const int key = i / %d, off = (i - key * %d) * 8;", dc / 8, dc / 8);
            w.line("const uint4 raw = *((const uint4*)(%s + key * %d + %s + off));",
                    ctx.ptr(src), d, chunkBase);
            w.line("const __half* hv = (const __half*)&raw;");
            w.line("#pragma unroll");
            w.line("for (int t = 0; t < 8; ++t) %s[key * %d + off + t] = %s;", dstArray, rowStride,
                    convert ? "__half2float(hv[t])" : "hv[t]");
            w.out();
            w.line("}");
        } else {
            w.line("for (int i = threadIdx.x; i < %d; i += %d) {", b * dc, threads);
            w.in();
            w.line("const int key = i / %d, dd = i - (i / %d) * %d;", dc, dc, dc);
            w.line("%s[key * %d + dd] = %s;", dstArray, rowStride,
                    convert ? ctx.loadScalar(src, "key * " + d + " + " + chunkBase + " + dd")
                            : ctx.loadScalarAsHalf(src, "key * " + d + " + " + chunkBase + " + dd"));
            w.out();
            w.line("}");
        }
    }

    static LaunchConfig launch(FusionGroup group, EmitContext ctx) {
        int b = group.params.get("seq");
        int rpb = group.params.get("rowsPerBlock");
        return new LaunchConfig(LaunchConfig.ceilDiv(b, rpb), 1, rpb * PascalHeuristics.WARP_SIZE, 0);
    }
}
