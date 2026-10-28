package io.fusionc.backend;

import io.fusionc.ir.IrNode;
import io.fusionc.ir.OpKind;
import io.fusionc.pass.FusionGroup;
import io.fusionc.pass.PascalHeuristics;

import java.util.List;

/**
 * Elementwise kernels: one op per kernel for {@code --no-fusion}, or a fused chain
 * ({@code bias -> relu -> ...}) that reads DRAM once and writes DRAM once for the whole chain.
 */
final class ElementwiseKernels {

    private ElementwiseKernels() {
    }

    static void emit(CudaWriter w, FusionGroup group, EmitContext ctx) {
        List<IrNode> members = group.members;
        IrNode out = group.external.get(group.external.size() - 1);
        IrNode source = members.get(0).inputs.get(0);
        int cols = members.get(0).shape.cols;
        long elements = members.get(0).shape.elements();

        w.line("// kernel %d: %s (%d elementwise op(s), %d elements)", group.id, group.kind.label,
                members.size(), elements);
        w.line("__global__ void %s(%s) {",
                FusionGroup.kernelName(group.kind, group.id, group.members),
                String.join(", ", ctx.kernelParams(group)));
        w.in();
        w.line("const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;");
        w.line("if (i >= %dL) return;", elements);
        String value = ctx.loadScalar(source, "i");
        for (IrNode node : members) {
            switch (node.op) {
                case BIAS_ADD:
                    value = "(" + value + " + " + ctx.loadScalar(node.inputs.get(1),
                            Epilogue.biasIndex(node.inputs.get(1), "(int)(i % " + cols + "L)")) + ")";
                    break;
                case ADD:
                    value = "(" + value + " + " + ctx.loadScalar(node.inputs.get(1), "i") + ")";
                    break;
                default:
                    value = Epilogue.unary(node.op, value, node.scale());
                    break;
            }
        }
        w.line(ctx.storeScalar(out, "i", value));
        w.out();
        w.line("}");
        w.blank();
    }

    static LaunchConfig launch(FusionGroup group, EmitContext ctx) {
        long elements = group.members.get(0).shape.elements();
        int threads = PascalHeuristics.ELEMENTWISE_THREADS;
        return new LaunchConfig((int) ((elements + threads - 1) / threads), 1, threads, 0);
    }
}
