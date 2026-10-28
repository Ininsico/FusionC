package io.fusionc.backend;

import io.fusionc.ir.IrNode;
import io.fusionc.pass.FusionGroup;

import java.util.List;

/**
 * Row-wise reduction kernels: softmax and LayerNorm.
 *
 * <p>One block per row; the reduction is a warp {@code __shfl_down} tree followed by a
 * cross-warp pass through shared memory. Row reductions are the one place where Pascal's
 * absent fast-FP16 path matters least, so the logits are kept in FP32 storage
 * (see {@code PROMOTE_REDUCTION_INPUTS_TO_FP32}).
 */
final class ReduceKernels {

    private ReduceKernels() {
    }

    static void emit(CudaWriter w, FusionGroup group, EmitContext ctx) {
        IrNode node = group.members.get(0);
        IrNode out = node;
        IrNode x = node.inputs.get(0);
        int rows = node.shape.rows;
        int cols = node.shape.cols;
        int threads = group.threads();
        int warps = (threads + 31) / 32;

        w.line("// kernel %d: %s over %d rows x %d columns, %d threads/block", group.id,
                node.op.name().toLowerCase(), rows, cols, threads);
        w.line("__global__ void %s(%s) {",
                FusionGroup.kernelName(group.kind, group.id, group.members),
                String.join(", ", ctx.kernelParams(group)));
        w.in();
        w.line("__shared__ float scratch[%d];", Math.max(warps, 1));
        w.line("const int row = blockIdx.x;");
        if (node.op.name().equals("SOFTMAX")) {
            w.line("float m = -INFINITY;");
            w.line("for (int j = threadIdx.x; j < %d; j += %d) m = fmaxf(m, %s);", cols, threads,
                    ctx.loadScalar(x, "(long)row * " + cols + " + j"));
            w.line("m = fx_block_max(m, scratch);");
            w.line("float s = 0.0f;");
            w.line("for (int j = threadIdx.x; j < %d; j += %d) s += expf(%s - m);", cols, threads,
                    ctx.loadScalar(x, "(long)row * " + cols + " + j"));
            w.line("s = fx_block_sum(s, scratch);");
            w.line("const float inv = 1.0f / s;");
            w.line("for (int j = threadIdx.x; j < %d; j += %d) {", cols, threads);
            w.in();
            w.line(ctx.storeScalar(out, "(long)row * " + cols + " + j",
                    "expf(" + ctx.loadScalar(x, "(long)row * " + cols + " + j") + " - m) * inv"));
            w.out();
            w.line("}");
        } else {
            IrNode gain = node.inputs.get(1);
            IrNode bias = node.inputs.get(2);
            double eps = node.scale();
            w.line("float s = 0.0f, ss = 0.0f;");
            w.line("for (int j = threadIdx.x; j < %d; j += %d) {", cols, threads);
            w.in();
            w.line("const float v = %s;", ctx.loadScalar(x, "(long)row * " + cols + " + j"));
            w.line("s += v; ss += v * v;");
            w.out();
            w.line("}");
            w.line("s = fx_block_sum(s, scratch);");
            w.line("ss = fx_block_sum(ss, scratch);");
            w.line("const float mean = s / %d.0f;", cols);
            w.line("const float var = ss / %d.0f - mean * mean;", cols);
            w.line("const float rstd = rsqrtf(var + %sf);", Epilogue.fmt(eps));
            w.line("for (int j = threadIdx.x; j < %d; j += %d) {", cols, threads);
            w.in();
            w.line("const float v = (%s - mean) * rstd * %s + %s;",
                    ctx.loadScalar(x, "(long)row * " + cols + " + j"),
                    ctx.loadScalar(gain, "j"), ctx.loadScalar(bias, "j"));
            w.line(ctx.storeScalar(out, "(long)row * " + cols + " + j", "v"));
            w.out();
            w.line("}");
        }
        w.out();
        w.line("}");
        w.blank();
    }

    static LaunchConfig launch(FusionGroup group, EmitContext ctx) {
        IrNode node = group.members.get(0);
        return new LaunchConfig(node.shape.rows, 1, group.threads(), 0);
    }
}
