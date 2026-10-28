package io.fusionc.ir;

/** Primitive operators of the IR. Composites (attention, MLP) are expanded before the IR. */
public enum OpKind {
    /** Host-supplied graph input. */
    INPUT(true),
    /** Host-supplied weight / bias tensor. */
    PARAM(true),
    /** C = A x B (row-major, attr {@code transB} selects A x B^T). */
    MATMUL(false),
    /** C[i][j] = X[i][j] + b[j] */
    BIAS_ADD(true),
    RELU(true),
    GELU(true),
    /** X * k */
    SCALE(true),
    /** elementwise A + B */
    ADD(true),
    /** row-wise softmax over the last dimension */
    SOFTMAX(false),
    /** row-wise LayerNorm over the last dimension, with gain/bias */
    LAYERNORM(false);

    /** True when every output element depends on exactly one input element (no reduction). */
    public final boolean elementwise;

    OpKind(boolean elementwise) {
        this.elementwise = elementwise;
    }

    public boolean isLeaf() {
        return this == INPUT || this == PARAM;
    }

    public boolean isMatmul() {
        return this == MATMUL;
    }
}
