package io.fusionc.backend;

/** Tiny indented line writer for generated CUDA source. */
public final class CudaWriter {

    private final StringBuilder sb = new StringBuilder();
    private int indent;

    public CudaWriter line(String text) {
        if (!text.isEmpty()) {
            indent();
            sb.append(text);
        }
        sb.append('\n');
        return this;
    }

    public CudaWriter line(String format, Object... args) {
        return line(String.format(java.util.Locale.ROOT, format, args));
    }

    public CudaWriter blank() {
        sb.append('\n');
        return this;
    }

    public CudaWriter in() {
        indent++;
        return this;
    }

    public CudaWriter out() {
        indent = Math.max(0, indent - 1);
        return this;
    }

    private void indent() {
        for (int i = 0; i < indent; i++) {
            sb.append("    ");
        }
    }

    public String text() {
        return sb.toString();
    }
}
