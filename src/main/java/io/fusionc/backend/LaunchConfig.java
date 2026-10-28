package io.fusionc.backend;

/** Grid/block/shared-memory configuration of an emitted kernel. */
public final class LaunchConfig {

    public final int gridX;
    public final int gridY;
    public final int block;
    public final int dynamicSmemBytes;

    public LaunchConfig(int gridX, int gridY, int block, int dynamicSmemBytes) {
        this.gridX = gridX;
        this.gridY = gridY;
        this.block = block;
        this.dynamicSmemBytes = dynamicSmemBytes;
    }

    public String gridExpression() {
        return gridY <= 1 ? Integer.toString(gridX) : "dim3(" + gridX + ", " + gridY + ")";
    }

    public static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }
}
