package io.fusionc.ir;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One node of the data-flow DAG: an operator instance annotated with tensor shape,
 * storage/accumulate dtype, memory placement and dependency edges.
 */
public final class IrNode {

    // Attribute keys used by the IR and consumed by the back end / fusion pass.
    public static final String ATTR_TRANS_B = "transB";
    public static final String ATTR_SCALE = "scale";
    public static final String ATTR_HOST_TENSOR = "hostTensor";
    public static final String ATTR_PIPELINE = "pipeline";
    public static final String ATTR_ORIGIN = "origin";

    public final int id;
    public final String name;
    public final OpKind op;

    /** Producer nodes (data dependencies). */
    public final List<IrNode> inputs = new ArrayList<>();
    /** Inverse edges, filled in by {@link IrGraph#finish()}. */
    public final List<IrNode> consumers = new ArrayList<>();

    public Shape shape;
    public DType store;
    public DType accum = DType.FP32;
    public MemSpace placement = MemSpace.GLOBAL;
    /** False when the value never needs a DRAM buffer (fused producer/consumer). */
    public boolean materialized = true;
    /** Fusion group id, or -1 while the node stands alone. */
    public int group = -1;

    public final Map<String, Object> attrs = new LinkedHashMap<>();

    public IrNode(int id, String name, OpKind op) {
        this.id = id;
        this.name = name;
        this.op = op;
    }

    public boolean isLeaf() {
        return op.isLeaf();
    }

    public String hostTensor() {
        Object v = attrs.get(ATTR_HOST_TENSOR);
        return v == null ? null : v.toString();
    }

    public boolean transB() {
        return Boolean.TRUE.equals(attrs.get(ATTR_TRANS_B));
    }

    public double scale() {
        Object v = attrs.get(ATTR_SCALE);
        return v == null ? 1.0 : ((Number) v).doubleValue();
    }

    public long bytes() {
        return shape.bytes(store);
    }

    /** Name of the pipeline this node was lowered from. */
    public String pipeline() {
        Object v = attrs.get(ATTR_PIPELINE);
        return v == null ? "" : v.toString();
    }

    @Override
    public String toString() {
        return name + ":" + op + "[" + shape + "," + store.irName + "/" + accum.irName + ","
                + placement + (materialized ? "" : ",fused") + "]";
    }
}
