package io.fusionc.pass;

/** The shapes of kernel the back end knows how to generate from a fused subgraph. */
public enum GroupKind {
    /** One primitive op, one kernel (the {@code --no-fusion} baseline). */
    SINGLE("single"),
    /** A chain of elementwise ops fused into one kernel. */
    ELEMENTWISE_FUSED("elementwise_fused"),
    /** matmul + elementwise epilogue chain, resident in registers. */
    GEMM_EPILOGUE("gemm_epilogue"),
    /** QK^T -> softmax -> PV as one flash-style kernel. */
    ATTENTION_FUSED("attention_fused"),
    /** GEMM -> activation -> GEMM with the intermediate tile kept in shared memory. */
    GEMM_GEMM_FUSED("gemm_gemm_fused");

    public final String label;

    GroupKind(String label) {
        this.label = label;
    }
}
