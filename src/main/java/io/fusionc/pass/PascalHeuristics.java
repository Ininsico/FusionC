package io.fusionc.pass;

/**
 * Every tunable threshold in the compiler, named, with the Pascal hardware fact that
 * justifies it. Nothing in the compiler hard-codes a number that is not listed here.
 *
 * <p>Target: NVIDIA sm_61 (Pascal GP10x). GTX 1050 Ti = GP107, 6 SMs; GTX 1070 = GP104,
 * 15 SMs; identical per-SM resources and identical ISA limits.
 */
public final class PascalHeuristics {

    private PascalHeuristics() {
    }

    // ---------------------------------------------------------------------------------
    // Hardware limits (CUDA 11.x, sm_61)
    // ---------------------------------------------------------------------------------

    /** sm_61 static shared memory limit per block. Pascal needs cudaFuncAttribute opt-in (64 KB,
     *  dynamic smem) to exceed this, which the back end deliberately does not do: occupancy is
     *  worth more than a bigger tile on a 1.5 GHz consumer part. */
    public static final int MAX_SMEM_BYTES_PER_BLOCK = 48 * 1024;

    /** Unified 96 KB L1/shared split per SM on GP10x. Two 48 KB blocks can be resident at once. */
    public static final int SMEM_BYTES_PER_SM = 96 * 1024;

    /** 255 registers per thread is the sm_61 architectural maximum. */
    public static final int MAX_REGS_PER_THREAD = 255;

    /** 64K 32-bit registers per SM on GP10x. */
    public static final int REGS_PER_SM = 64 * 1024;

    /** ptxas allocates registers per warp in units of 256 registers = 8 registers/thread. */
    public static final int REG_ALLOC_GRANULARITY = 8;

    /** sm_61 limit for blockDim.x * blockDim.y * blockDim.z. */
    public static final int MAX_THREADS_PER_BLOCK = 1024;

    /** 2048 resident threads per SM on GP10x (64 warps). */
    public static final int MAX_THREADS_PER_SM = 2048;

    /** 32 resident blocks per SM on GP10x. */
    public static final int MAX_BLOCKS_PER_SM = 32;

    /** Pascal warp is 32 lanes with a single program counter (no independent thread scheduling). */
    public static final int WARP_SIZE = 32;

    /** 32 shared-memory banks of 4 bytes; conflicts are avoided by keeping strided rows 4-byte wide. */
    public static final int SMEM_BANK_BYTES = 4;

    /** GP10x (all consumer Pascal) run FP16 arithmetic at 1/64 of the FP32 rate. This is why the
     *  numeric policy is FP16 storage / FP32 accumulate: FP16 buys bandwidth, never math throughput. */
    public static final int FP16_MATH_RATE_DIVISOR = 64;

    // ---------------------------------------------------------------------------------
    // Target device
    // ---------------------------------------------------------------------------------

    /** SMs used by occupancy heuristics. Default 6 = GTX 1050 Ti (the brief's target; a GTX 1070
     *  reports 15). The driver overrides this with the real device property. */
    public static int targetSmCount = 6;

    // ---------------------------------------------------------------------------------
    // Kernel tile configuration (shared by the cost model and the emitters - they must agree)
    // ---------------------------------------------------------------------------------

    /** Default tiled GEMM tile: 64x64 output tile, BK depth step. 2*(64*32)*2B = 8 KB of smem. */
    public static final int GEMM_BM = 64;
    public static final int GEMM_BN = 64;
    public static final int GEMM_BK = 32;
    public static final int GEMM_THREADS = 256;

    /**
     * GEMM tile candidates {BM, BN}, tried largest-first by the tile chooser.
     *
     * <p>A 64x64 tile is the best shape when the problem is big enough to fill the device, but the
     * transformer in the brief has 64 rows: 64x64 tiles leave a 64x1024 GEMM with 16 blocks and a
     * 64x64 score GEMM with a single block. The chooser therefore walks down to the largest tile
     * whose grid still reaches MIN_GRID_BLOCKS_PER_SM blocks per SM. All candidates use the same
     * 4x4 per-thread micro-tile, so only the tile shape and the block size change.
     * Threads = (BM/4) * (BN/4).
     */
    public static final int[][] GEMM_TILES = {
        {64, 64},
        {64, 32},
        {32, 64},
        {32, 32},
    };

    /** A GEMM tile is stepped down only while its grid cannot reach this many blocks at all.
     *  Measured on this part (docs/REPORT.md): stepping further down costs more than it buys -
     *  a 64x64 tile (256 threads, 16 accumulators/thread) beat a 32x32 tile (64 threads) on every
     *  K-heavy GEMM even though the small tile has 4x the grid - while a 64x64 tile on the 64x64
     *  score GEMM leaves a single block running on a 16-SM part, which must be avoided. */
    public static final int GEMM_TILE_MIN_BLOCKS = 4;

    /** Elementwise / reduction kernels: 1 block per row for softmax and layernorm. */
    public static final int ELEMENTWISE_THREADS = 256;
    public static final int ROW_REDUCE_THREADS = 256;

    /** Fused attention: dim-chunk staged through shared memory. 2 stages * (B*(DC*2 + DC*2))
     *  must stay under the 48 KB cap; 64 == the largest power-of-two chunk for D=256 at B=64. */
    public static final int ATTN_DIM_CHUNK = 64;
    /** Query rows per block (one warp per row). More rows = fewer K/V reloads, fewer blocks. */
    public static final int ATTN_ROWS_PER_BLOCK = 2;
    /** No cp.async on Pascal: prefetch is register-staged, so two smem stages are required. */
    public static final int SMEM_STAGES_NO_ASYNC = 2;

    /** Fused MLP (GEMM -> GEMM): rows per block x columns per block, hidden chunk, K step.
     *  smem = RM*BK + BK*HF + RM*HF + HF*CN halves = 2+8+8+16 = 34 KB at the defaults. */
    public static final int MLP_RM = 32;
    public static final int MLP_CN = 64;
    public static final int MLP_BK = 32;
    public static final int MLP_HF = 128;
    public static final int MLP_THREADS = 256;

    // ---------------------------------------------------------------------------------
    // Fusion policy -- resource guards
    // ---------------------------------------------------------------------------------

    /** Fraction of the 48 KB/block cap a fused kernel is allowed to predict. */
    public static final double SMEM_SAFETY_MARGIN = 0.90;

    /** Fraction of the 255 register/thread cap a fused kernel is allowed to predict. */
    public static final double REG_SAFETY_MARGIN = 0.90;

    /** Predicted occupancy below this is considered harmful; fusion is then rejected unless the
     *  estimated global-traffic saving is very large (see OCCUPANCY_OVERRIDE_SAVING_BYTES). */
    public static final double MIN_PREDICTED_OCCUPANCY = 0.25;

    /** Escape hatch for the occupancy guard: a fusion that removes this many bytes of DRAM traffic
     *  is accepted even if it drops predicted occupancy below MIN_PREDICTED_OCCUPANCY. */
    public static final long OCCUPANCY_OVERRIDE_SAVING_BYTES = 512 * 1024;

    /** A fused kernel that removes less than this many bytes of DRAM traffic is not worth the
     *  register/occupancy cost: keep two small kernels instead. */
    public static final long MIN_GLOBAL_BYTES_SAVED = 8 * 1024;

    /** A fused kernel must still fill the device: grid >= this many blocks per SM. The GTX 1050 Ti
     *  has only 6 SMs, so a 4-block kernel idles 2/3 of the part. */
    public static final int MIN_GRID_BLOCKS_PER_SM = 2;

    /** Elementwise epilogues consume the producer tile straight out of registers, so they are
     *  always profitable; this switch exists to measure the cost of the decision being wrong. */
    public static final boolean FUSE_ELEMENTWISE_EPILOGUE = true;

    /** Maximum number of elementwise ops chained onto a producer (bias -> relu -> ... ) */
    public static final int MAX_EPILOGUE_CHAIN = 4;

    /** Recomputing a producer tile instead of spilling it to DRAM is only allowed while the extra
     *  weight traffic stays within this factor of what the unfused tiling would read: a fused
     *  kernel that streams the weights more often than the baseline trades DRAM savings for L2
     *  traffic and loses. Measured on this part: the 2x-amplified fused MLP ran 1.8x slower than
     *  the unfused pair (docs/REPORT.md section 5). */
    public static final double MAX_PRODUCER_REREAD_AMPLIFICATION = 1.0;

    /** Iterations of the fusion pipeline before the pass declares a fixpoint. */
    public static final int FUSION_FIXPOINT_ITERATIONS = 4;

    // ---------------------------------------------------------------------------------
    // Numerically-driven policy constants
    // ---------------------------------------------------------------------------------

    /** Softmax/LayerNorm inputs are promoted to FP32 storage even if the graph policy is FP16:
     *  a 10-bit mantissa on the attention logits costs ~1e-2 of probability mass after expf. */
    public static final boolean PROMOTE_REDUCTION_INPUTS_TO_FP32 = true;

    /** Register overhead assumed for addressing, indices and loop bookkeeping in the cost model. */
    public static final int REG_BASE_OVERHEAD = 32;

    /** Fixed cost per kernel launch (1.5 GHz Pascal, WDDM) approximated for decision logging. */
    public static final double LAUNCH_COST_US = 4.0;
}
