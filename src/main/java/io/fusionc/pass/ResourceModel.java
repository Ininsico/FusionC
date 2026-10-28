package io.fusionc.pass;

import io.fusionc.ir.IrNode;

/**
 * Analytic cost model used by the fusion heuristics. Every formula here mirrors the code the
 * emitters actually produce, so "predicted" values in the decision log can be compared against
 * {@code ptxas -v} output (see docs/REPORT.md, section "prediction quality").
 */
public final class ResourceModel {

    private ResourceModel() {
    }

    public static final class Est {
        public final int threads;
        public final int smemBytes;
        public final int regsPerThread;
        public final int gridBlocks;
        public final double occupancy;

        Est(int threads, int smemBytes, int regsPerThread, int gridBlocks) {
            this.threads = threads;
            this.smemBytes = smemBytes;
            this.regsPerThread = regsPerThread;
            this.gridBlocks = gridBlocks;
            this.occupancy = occupancy(threads, regsPerThread, smemBytes);
        }
    }

    /** ptxas rounds the register count per warp up to a multiple of 256 registers (= 8 regs/thread). */
    public static int roundRegs(int regs) {
        int g = PascalHeuristics.REG_ALLOC_GRANULARITY;
        return ((regs + g - 1) / g) * g;
    }

    public static int residentBlocks(int threads, int regsPerThread, int smemBytes) {
        int byThreads = PascalHeuristics.MAX_THREADS_PER_SM / threads;
        int byBlocks = PascalHeuristics.MAX_BLOCKS_PER_SM;
        int byRegs = (int) ((long) PascalHeuristics.REGS_PER_SM / ((long) roundRegs(regsPerThread) * threads));
        int bySmem = smemBytes <= 0 ? byBlocks : PascalHeuristics.SMEM_BYTES_PER_SM / smemBytes;
        int blocks = Math.min(Math.min(byThreads, byBlocks), Math.min(byRegs, bySmem));
        return Math.max(1, blocks);
    }

    public static double occupancy(int threads, int regsPerThread, int smemBytes) {
        int resident = residentBlocks(threads, regsPerThread, smemBytes);
        double warps = (double) (resident * threads) / PascalHeuristics.MAX_THREADS_PER_SM;
        return Math.min(1.0, warps);
    }

    private static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }

    /**
     * Picks the GEMM tile: the largest candidate whose grid still reaches {@code GEMM_TILE_MIN_BLOCKS},
     * else the smallest. A 64-row GEMM tiled 64x64 runs as a single block on a 16-SM part, which is
     * why the step-down exists at all.
     */
    public static int[] chooseGemmTile(int m, int n) {
        int floor = PascalHeuristics.GEMM_TILE_MIN_BLOCKS;
        int[] smallest = PascalHeuristics.GEMM_TILES[PascalHeuristics.GEMM_TILES.length - 1];
        for (int[] tile : PascalHeuristics.GEMM_TILES) {
            if (ceilDiv(m, tile[0]) * ceilDiv(n, tile[1]) >= floor) {
                return tile;
            }
        }
        return smallest;
    }

    public static int gemmThreads(int[] tile) {
        return (tile[0] / 4) * (tile[1] / 4);
    }

    /** Tiled GEMM (also the epilogue-fused variant: the epilogue costs a couple of registers). */
    public static Est gemm(int m, int n, int epilogueOps) {
        return gemm(m, n, epilogueOps, chooseGemmTile(m, n));
    }

    public static Est gemm(int m, int n, int epilogueOps, int[] tile) {
        int bm = tile[0];
        int bn = tile[1];
        int threads = gemmThreads(tile);
        int smem = (bm * PascalHeuristics.GEMM_BK + PascalHeuristics.GEMM_BK * bn) * 2;
        int accum = bm * bn / threads;
        int regs = accum + PascalHeuristics.REG_BASE_OVERHEAD + 2 * epilogueOps;
        int grid = ceilDiv(m, bm) * ceilDiv(n, bn);
        return new Est(threads, smem, regs, grid);
    }

    /** Vectorised elementwise kernel. */
    public static Est elementwise(int chainOps) {
        int regs = PascalHeuristics.REG_BASE_OVERHEAD + 2 * Math.max(1, chainOps);
        return new Est(PascalHeuristics.ELEMENTWISE_THREADS, 0, regs, 1);
    }

    /** Row reduction kernel (softmax, layernorm): warp scratch + block scratch. */
    public static Est rowReduce(int rows, int cols) {
        int threads = Math.min(PascalHeuristics.ROW_REDUCE_THREADS, ((cols + 31) / 32) * 32);
        threads = Math.max(32, threads);
        int smem = (threads / 32) * 4;
        int regs = PascalHeuristics.REG_BASE_OVERHEAD + 8;
        return new Est(threads, smem, regs, Math.max(1, rows));
    }

    /**
     * Fused attention (QK^T + softmax + PV), one warp per query row, double-buffered dimension
     * chunks staged through shared memory because Pascal has no cp.async. The stage buffer is a
     * union of the padded FP16 K tile and the FP32 V tile, so only the larger of the two counts.
     */
    public static Est attention(int b, int d, int rowsPerBlock, int dimChunk) {
        int stages = PascalHeuristics.SMEM_STAGES_NO_ASYNC;
        int kBytes = b * (dimChunk + AttentionPad.HALVES) * 2;
        int vBytes = b * dimChunk * 4;
        int smem = stages * Math.max(kBytes, vBytes)
                + rowsPerBlock * d * 2                              // Q rows
                + rowsPerBlock * b * 4;                             // probabilities, fp32
        int accum = d / PascalHeuristics.WARP_SIZE;
        int regs = accum + 2 + PascalHeuristics.REG_BASE_OVERHEAD / 2;
        int grid = ceilDiv(b, rowsPerBlock);
        return new Est(rowsPerBlock * PascalHeuristics.WARP_SIZE, smem, regs, grid);
    }

    /** Kept here so the cost model and the emitter cannot drift apart. */
    public static final class AttentionPad {
        public static final int HALVES = 2;

        private AttentionPad() {
        }
    }

    /** Fused two-GEMM MLP: hidden tile produced into shared memory and consumed in place. */
    public static Est mlp(int m, int n, int rm, int cn, int bk, int hf) {
        int smem = (rm * bk + bk * hf + rm * hf + hf * cn) * 2;
        int yAccum = rm * cn / PascalHeuristics.MLP_THREADS;
        int hAccum = rm * hf / PascalHeuristics.MLP_THREADS;
        int regs = yAccum + hAccum + PascalHeuristics.REG_BASE_OVERHEAD + 8;
        int grid = ceilDiv(m, rm) * ceilDiv(n, cn);
        return new Est(PascalHeuristics.MLP_THREADS, smem, regs, grid);
    }

    /** Registers per thread needed for a producer/consumer pair to stay in registers. */
    public static int accumRegsForMlp(int rm, int cn) {
        return rm * cn / PascalHeuristics.MLP_THREADS;
    }

    /**
     * DRAM traffic removed by producing {@code producer} straight into the consumer instead of
     * spilling it (value written once + read back by each consumer that is fused away).
     */
    public static long bytesSavedByFusing(long producedBytes, int consumersInsideGroup, int internalStages) {
        return producedBytes * (1 + consumersInsideGroup) + producedBytes * 2L * internalStages;
    }

    /** True when the estimate fits the sm_61 caps with the configured safety margins. */
    public static boolean fits(Est est) {
        double smemCap = PascalHeuristics.MAX_SMEM_BYTES_PER_BLOCK * PascalHeuristics.SMEM_SAFETY_MARGIN;
        double regCap = PascalHeuristics.MAX_REGS_PER_THREAD * PascalHeuristics.REG_SAFETY_MARGIN;
        return est.smemBytes <= smemCap && est.regsPerThread <= regCap
                && est.threads <= PascalHeuristics.MAX_THREADS_PER_BLOCK;
    }

    /** Grid parallelism guard: a fused kernel must still fill the device. */
    public static boolean fillsDevice(Est est) {
        return est.gridBlocks >= PascalHeuristics.MIN_GRID_BLOCKS_PER_SM * PascalHeuristics.targetSmCount;
    }

    public static String describe(Est est) {
        return String.format(java.util.Locale.ROOT,
                "smem=%.1fKB regs=%d threads=%d grid=%d occupancy=%.2f",
                est.smemBytes / 1024.0, est.regsPerThread, est.threads, est.gridBlocks, est.occupancy);
    }

    /** Sum of DRAM bytes a set of nodes must materialise (used for the unfused side of a decision). */
    public static long materialisedBytes(Iterable<IrNode> nodes) {
        long total = 0;
        for (IrNode n : nodes) {
            if (!n.isLeaf()) {
                total += n.bytes();
            }
        }
        return total;
    }
}
