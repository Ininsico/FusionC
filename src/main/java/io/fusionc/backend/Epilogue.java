package io.fusionc.backend;

import io.fusionc.ir.IrNode;
import io.fusionc.ir.OpKind;

import java.util.List;

/** Shared code generation for epilogue chains (activation / bias / scale). */
final class Epilogue {

    private Epilogue() {
    }

    /** Applies a unary epilogue op to a float expression. */
    static String unary(OpKind op, String expr, double scale) {
        switch (op) {
            case RELU:
                return "fmaxf(" + expr + ", 0.0f)";
            case GELU:
                return "fx_gelu(" + expr + ")";
            case SCALE:
                return "(" + expr + " * " + fmt(scale) + "f)";
            default:
                throw new IllegalArgumentException("op " + op + " is not a unary epilogue");
        }
    }

    static String fmt(double value) {
        String s = String.format(java.util.Locale.ROOT, "%.9g", value);
        if (s.contains("e")) {
            s = String.format(java.util.Locale.ROOT, "%.9f", value);
        }
        return s;
    }

    /** Index expression for a bias vector of {@code cols} elements, given the flat element index. */
    static String biasIndex(IrNode bias, String colExpr) {
        int cols = bias.shape.cols;
        if (bias.shape.rows == 1 || bias.shape.rows != cols) {
            return colExpr + " % " + cols;
        }
        return "(" + colExpr + " / " + cols + ") % " + cols;
    }

    /** True when the op needs a second tensor operand. */
    static boolean isBinary(OpKind op) {
        return op == OpKind.BIAS_ADD || op == OpKind.ADD;
    }
}
