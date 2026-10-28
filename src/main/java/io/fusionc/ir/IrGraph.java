package io.fusionc.ir;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** The dependency DAG: nodes in creation order (already a topological order) plus edges. */
public final class IrGraph {

    public final List<IrNode> nodes = new ArrayList<>();
    public final Map<String, IrNode> byName = new LinkedHashMap<>();
    /** Pipeline outputs, in declaration order; the last one is the module result. */
    public final List<IrNode> outputs = new ArrayList<>();
    /** DSL name (pipeline result or host tensor) -> node, for back-reference from later pipelines. */
    public final Map<String, IrNode> namedOutputs = new LinkedHashMap<>();

    private int nextId;

    public IrNode create(String name, OpKind op) {
        if (byName.containsKey(name)) {
            throw new IllegalStateException("duplicate IR node name '" + name + "'");
        }
        IrNode node = new IrNode(nextId++, name, op);
        nodes.add(node);
        byName.put(name, node);
        return node;
    }

    /** Wires producer edges and fills the reverse (consumer) edges. */
    public void connect(IrNode node, IrNode... inputs) {
        for (IrNode in : inputs) {
            node.inputs.add(in);
            in.consumers.add(node);
        }
    }

    /** Creates a fresh, unused node name derived from {@code base}. */
    public String uniqueName(String base) {
        if (!byName.containsKey(base)) {
            return base;
        }
        for (int i = 2; ; i++) {
            String candidate = base + "_" + i;
            if (!byName.containsKey(candidate)) {
                return candidate;
            }
        }
    }

    public List<IrNode> leaves() {
        List<IrNode> out = new ArrayList<>();
        for (IrNode n : nodes) {
            if (n.isLeaf()) {
                out.add(n);
            }
        }
        return out;
    }

    public List<IrNode> consumersOf(IrNode n) {
        return n.consumers;
    }

    /** Leaves first, then creation order: consumers always follow their producers. */
    public List<IrNode> topological() {
        return nodes;
    }

    /** Human-readable IR annotation dump (op, shape, dtype, placement, deps). */
    public String dump() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-4s %-22s %-10s %-9s %-11s %-10s %-6s %s%n",
                "id", "name", "op", "shape", "store/acc", "placement", "group", "inputs -> consumers"));
        for (IrNode n : nodes) {
            List<String> ins = new ArrayList<>();
            for (IrNode i : n.inputs) {
                ins.add(i.name);
            }
            List<String> outs = new ArrayList<>();
            for (IrNode c : n.consumers) {
                outs.add(c.name);
            }
            String flag = n.materialized ? "" : " (fused, no DRAM buffer)";
            sb.append(String.format("%-4d %-22s %-10s %-9s %-11s %-10s %-6s %s -> %s%s%n",
                    n.id, n.name, n.op, n.shape,
                    n.store.irName + "/" + n.accum.irName,
                    n.placement, n.group == -1 ? "-" : Integer.toString(n.group),
                    ins.isEmpty() ? "(none)" : String.join(",", ins),
                    outs.isEmpty() ? "(dead)" : String.join(",", outs),
                    flag));
        }
        return sb.toString();
    }
}
