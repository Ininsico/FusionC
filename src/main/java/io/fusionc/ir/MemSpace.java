package io.fusionc.ir;

/** Where a tensor value physically resides for the duration of a fused kernel. */
public enum MemSpace {
    /** DRAM, addressed through global pointers. */
    GLOBAL,
    /** Cooperative shared memory (48 KB/block cap on Pascal without opt-in). */
    SHARED,
    /** Per-thread registers (255/thread cap on Pascal). */
    REGISTER
}
