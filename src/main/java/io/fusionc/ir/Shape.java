package io.fusionc.ir;

/** Row-major 2-D tensor shape. Batches are folded into {@link #rows}. */
public final class Shape {

    public final int rows;
    public final int cols;

    public Shape(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    public long elements() {
        return (long) rows * cols;
    }

    /** Bytes needed to store this shape at {@code dtype}. */
    public long bytes(DType dtype) {
        return elements() * dtype.sizeBytes;
    }

    @Override
    public String toString() {
        return rows + "x" + cols;
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof Shape)) {
            return false;
        }
        Shape s = (Shape) other;
        return s.rows == rows && s.cols == cols;
    }

    @Override
    public int hashCode() {
        return rows * 31 + cols;
    }
}
