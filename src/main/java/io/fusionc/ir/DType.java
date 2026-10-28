package io.fusionc.ir;

/**
 * Storage / arithmetic precision.
 *
 * <p>Policy for consumer Pascal (sm_61): <b>FP16 storage, FP32 arithmetic</b>.
 * GP104/GP107 have no fast FP16 units — native half arithmetic runs at 1/64 of
 * the FP32 rate — so FP16 is used only to halve DRAM and shared-memory traffic,
 * and every multiply-accumulate is performed in FP32 registers.
 */
public enum DType {
    FP16("__half", 2, "half"),
    FP32("float", 4, "float");

    /** CUDA C type name emitted by the back end. */
    public final String cType;
    /** Bytes per element in device memory. */
    public final int sizeBytes;
    /** Short name used in the IR dump. */
    public final String irName;

    DType(String cType, int sizeBytes, String irName) {
        this.cType = cType;
        this.sizeBytes = sizeBytes;
        this.irName = irName;
    }

    public static DType fromDsl(String name) {
        if ("fp16".equals(name)) {
            return FP16;
        }
        if ("fp32".equals(name)) {
            return FP32;
        }
        throw new IllegalArgumentException("unknown datatype '" + name + "'");
    }
}
