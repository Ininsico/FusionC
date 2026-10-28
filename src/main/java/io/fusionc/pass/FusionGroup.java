package io.fusionc.pass;

import io.fusionc.ir.IrNode;
import io.fusionc.ir.MemSpace;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A scheduled kernel: one or more IR nodes emitted as a single CUDA kernel, plus the
 * resource estimate that justified (or rejected) the fusion.
 */
public final class FusionGroup {

    public final int id;
    public final GroupKind kind;
    /** Member nodes, in execution order. */
    public final List<IrNode> members = new ArrayList<>();
    /** Nodes whose value must be written to DRAM (everything else stays in registers/smem). */
    public final List<IrNode> external = new ArrayList<>();
    /** Nodes whose value stays inside the kernel. */
    public final List<IrNode> internal = new ArrayList<>();
    /** Chosen kernel parameters (tile sizes, chunk sizes, threads). */
    public final Map<String, Integer> params = new LinkedHashMap<>();

    public FusionDecision decision;

    public FusionGroup(int id, GroupKind kind) {
        this.id = id;
        this.kind = kind;
    }

    /** Kernel name as emitted in the CUDA source. */
    public String kernelName() {
        return kernelName(kind, id, members);
    }

    public static String kernelName(GroupKind kind, int id, List<IrNode> members) {
        String base = kind == GroupKind.SINGLE ? members.get(0).name : kind.label + "_" + id;
        return "k_" + sanitize(base);
    }

    public static String sanitize(String name) {
        return name.replace('.', '_');
    }

    /** Region label used by the generated harness timing table (must match across back ends). */
    public String regionName() {
        return kind == GroupKind.SINGLE ? members.get(0).name : kind.label + "#" + id;
    }

    public int threads() {
        Integer t = params.get("threads");
        return t == null ? 256 : t;
    }

    public int smemBytes() {
        Integer s = params.get("smemBytes");
        return s == null ? 0 : s;
    }

    /** Total DRAM traffic (bytes) this group must move, for cost accounting. */
    public long globalBytes() {
        long bytes = 0;
        for (IrNode n : external) {
            bytes += n.bytes();
        }
        return bytes;
    }

    public String memberNames() {
        List<String> names = new ArrayList<>();
        for (IrNode n : members) {
            names.add(n.name);
        }
        return String.join(",", names);
    }

    public void classify() {
        for (IrNode n : members) {
            if (materialized(n)) {
                external.add(n);
                n.materialized = true;
            } else {
                internal.add(n);
                n.materialized = false;
                n.placement = MemSpace.SHARED;
            }
            n.group = id;
        }
    }

    /** A member stays in DRAM when it is a graph output or has a consumer outside the group. */
    private boolean materialized(IrNode node) {
        if (kind == GroupKind.SINGLE) {
            return true;
        }
        for (IrNode consumer : node.consumers) {
            if (!members.contains(consumer)) {
                return true;
            }
        }
        for (IrNode output : outputs) {
            if (output == node) {
                return true;
            }
        }
        return false;
    }

    private List<IrNode> outputs = new ArrayList<>();

    public void setGraphOutputs(List<IrNode> graphOutputs) {
        this.outputs = graphOutputs;
    }

    @Override
    public String toString() {
        return kind.label + "#" + id + "[" + memberNames() + "]";
    }
}
