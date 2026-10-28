package io.fusionc.backend;

import io.fusionc.ir.DType;
import io.fusionc.ir.IrGraph;
import io.fusionc.ir.IrNode;
import io.fusionc.pass.FusionGroup;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Shared state and naming helpers for the kernel emitters. */
public final class EmitContext {

    public final IrGraph graph;
    public final boolean fused;
    private final Map<IrNode, Integer> bufferIndex = new IdentityHashMap<>();
    private final List<IrNode> buffers = new ArrayList<>();
    /** Region (kernel group) per node, used by the harness timing table. */
    private final Map<IrNode, FusionGroup> kernelOf = new IdentityHashMap<>();

    public EmitContext(IrGraph graph, List<FusionGroup> groups, boolean fused) {
        this.graph = graph;
        this.fused = fused;
        for (FusionGroup group : groups) {
            for (IrNode node : group.members) {
                kernelOf.put(node, group);
            }
        }
        for (IrNode node : graph.nodes) {
            if (node.materialized) {
                bufferIndex.put(node, buffers.size());
                buffers.add(node);
            }
        }
    }

    public List<IrNode> buffers() {
        return buffers;
    }

    public int bufferOf(IrNode node) {
        Integer index = bufferIndex.get(node);
        if (index == null) {
            throw new IllegalStateException("node '" + node.name + "' has no device buffer "
                    + "(materialised=" + node.materialized + ")");
        }
        return index;
    }

    public boolean hasBuffer(IrNode node) {
        return bufferIndex.containsKey(node);
    }

    public FusionGroup groupOf(IrNode node) {
        return kernelOf.get(node);
    }

    /** Device symbol for a tensor value: P_* for host tensors, T_* for produced tensors. */
    public String ptr(IrNode node) {
        String base = sanitize(node.name);
        return node.isLeaf() ? "P_" + base : "T_" + base;
    }

    public static String sanitize(String name) {
        return name.replace('.', '_').replace('-', '_');
    }

    /** Load a single element as FP32 (FP16 storage is converted, never used for math on Pascal). */
    public String loadScalar(IrNode tensor, String indexExpr) {
        String p = ptr(tensor);
        if (tensor.store == DType.FP16) {
            return "__half2float(" + p + "[" + indexExpr + "])";
        }
        return p + "[" + indexExpr + "]";
    }

    /** Store an FP32 expression into a tensor of the tensor's storage dtype. */
    public String storeScalar(IrNode tensor, String indexExpr, String value) {
        String p = ptr(tensor);
        if (tensor.store == DType.FP16) {
            return p + "[" + indexExpr + "] = __float2half(" + value + ");";
        }
        return p + "[" + indexExpr + "] = " + value + ";";
    }

    /** Load an element as a native FP16 value (for filling FP16 shared tiles). */
    public String loadScalarAsHalf(IrNode tensor, String indexExpr) {
        String p = ptr(tensor);
        if (tensor.store == DType.FP16) {
            return p + "[" + indexExpr + "]";
        }
        return "__float2half(" + p + "[" + indexExpr + "])";
    }

    public static String cType(DType dtype) {
        return dtype.cType;
    }

    /**
     * Kernel parameters: the group's externally visible outputs first, then every input that is
     * not produced inside this group (host tensors and tensors written by other kernels).
     */
    public List<IrNode> kernelParamNodes(FusionGroup group) {
        List<IrNode> params = new ArrayList<>(group.external);
        java.util.Set<IrNode> seen = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        seen.addAll(params);
        for (IrNode member : group.members) {
            for (IrNode in : member.inputs) {
                if (group.members.contains(in)) {
                    continue;
                }
                if (seen.add(in)) {
                    params.add(in);
                }
            }
        }
        return params;
    }

    public List<String> kernelParams(FusionGroup group) {
        List<IrNode> nodes = kernelParamNodes(group);
        List<String> params = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            IrNode node = nodes.get(i);
            boolean produced = i < group.external.size();
            params.add((produced ? "" : "const ") + cType(node.store) + "* __restrict__ " + ptr(node));
        }
        return params;
    }

    /** Arguments passed at the launch site, in the same order as {@link #kernelParams}. */
    public String launchArgs(FusionGroup group) {
        List<String> args = new ArrayList<>();
        for (IrNode node : kernelParamNodes(group)) {
            args.add(ptr(node));
        }
        return String.join(", ", args);
    }

    /** Maps a group's member nodes to the epilogue chain that follows a producer inside it. */
    public static List<IrNode> epilogueAfter(FusionGroup group, IrNode producer) {
        List<IrNode> chain = new ArrayList<>();
        boolean seen = false;
        for (IrNode member : group.members) {
            if (member == producer) {
                seen = true;
                continue;
            }
            if (seen) {
                chain.add(member);
            }
        }
        return chain;
    }
}
