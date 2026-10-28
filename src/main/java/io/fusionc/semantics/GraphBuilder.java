package io.fusionc.semantics;

import io.fusionc.FrontendException;
import io.fusionc.ast.Ast;
import io.fusionc.ir.DType;
import io.fusionc.ir.IrGraph;
import io.fusionc.ir.IrNode;
import io.fusionc.ir.OpKind;
import io.fusionc.ir.Shape;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Lowers the parsed AST into the primitive IR DAG.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>declaration processing (constants, numeric policy, host tensors),</li>
 *   <li>composite macro expansion ({@code attention}, {@code MLP}) into primitives,</li>
 *   <li>shape and dtype inference, with hard errors on mismatch,</li>
 *   <li>hard errors on unsupported ops (the compiler must fail loudly, never miscompile).</li>
 * </ul>
 */
public final class GraphBuilder {

    private final Map<String, Long> consts = new LinkedHashMap<>();
    private final Map<String, DType> hostStore = new LinkedHashMap<>();
    private final Map<String, Shape> hostShape = new LinkedHashMap<>();
    private final Map<String, Boolean> hostIsInput = new LinkedHashMap<>();
    private final Map<String, IrNode> visible = new LinkedHashMap<>();

    private final IrGraph graph = new IrGraph();
    private DType defaultStore = DType.FP16;
    private DType defaultAccum = DType.FP32;

    private GraphBuilder() {
    }

    public static IrGraph build(Ast.Program program) {
        GraphBuilder builder = new GraphBuilder();
        return builder.run(program);
    }

    private IrGraph run(Ast.Program program) {
        for (Ast.Decl decl : program.decls) {
            collect(decl);
        }
        if (hostShape.isEmpty() && program.pipelines.isEmpty()) {
            throw new FrontendException("input contains no tensors and no pipelines");
        }
        for (Ast.Pipeline pipeline : program.pipelines) {
            lowerPipeline(pipeline);
        }
        for (IrNode node : graph.nodes) {
            for (IrNode consumer : node.consumers) {
                if (!consumer.inputs.contains(node)) {
                    throw new IllegalStateException("inconsistent DAG edge " + node.name + " -> " + consumer.name);
                }
            }
        }
        return graph;
    }

    // ------------------------------------------------------------ declarations

    private void collect(Ast.Decl decl) {
        if (decl instanceof Ast.ConstDecl) {
            Ast.ConstDecl c = (Ast.ConstDecl) decl;
            if (consts.containsKey(c.name)) {
                throw new FrontendException("duplicate constant '" + c.name + "'");
            }
            if (c.value <= 0) {
                throw new FrontendException("constant '" + c.name + "' must be positive, got " + c.value);
            }
            consts.put(c.name, c.value);
        } else if (decl instanceof Ast.DtypeDecl) {
            Ast.DtypeDecl d = (Ast.DtypeDecl) decl;
            defaultStore = DType.fromDsl(d.store);
            defaultAccum = DType.fromDsl(d.accum);
            if (defaultAccum != DType.FP32) {
                throw new FrontendException(
                    "unsupported numeric policy: accumulation must be fp32 on Pascal (FP16 arithmetic runs at "
                    + "1/64 of the FP32 rate on GP10x); got accum " + d.accum);
            }
        } else {
            Ast.TensorDecl t = (Ast.TensorDecl) decl;
            if (hostShape.containsKey(t.name) || visible.containsKey(t.name)) {
                throw new FrontendException("duplicate tensor '" + t.name + "'");
            }
            if (t.dims.size() < 1 || t.dims.size() > 2) {
                throw new FrontendException("tensor '" + t.name + "' must be 1-D (bias/gain vector) or 2-D "
                        + "(fold the batch into the rows), got " + t.dims.size() + " dimensions");
            }
            int rows = t.dims.size() == 1 ? 1 : (int) evalDim(t.dims.get(0), t.name);
            int cols = (int) evalDim(t.dims.get(t.dims.size() - 1), t.name);
            DType store = DType.fromDsl(t.dtype);
            Shape shape = new Shape(rows, cols);
            hostShape.put(t.name, shape);
            hostStore.put(t.name, store);
            hostIsInput.put(t.name, t.input);

            IrNode node = graph.create(t.name, t.input ? OpKind.INPUT : OpKind.PARAM);
            node.shape = shape;
            node.store = store;
            node.accum = DType.FP32;
            node.attrs.put(IrNode.ATTR_HOST_TENSOR, t.name);
            visible.put(t.name, node);
            graph.namedOutputs.put(t.name, node);
        }
    }

    private long evalDim(Ast.Expr expr, String tensor) {
        if (expr instanceof Ast.IntLit) {
            return ((Ast.IntLit) expr).value;
        }
        if (expr instanceof Ast.Ref) {
            Long value = consts.get(((Ast.Ref) expr).name);
            if (value == null) {
                throw new FrontendException("unknown constant '" + ((Ast.Ref) expr).name
                        + "' in shape of tensor '" + tensor + "'");
            }
            return value;
        }
        throw new FrontendException("tensor '" + tensor + "' has a non-integer dimension");
    }

    // --------------------------------------------------------------- pipelines

    private void lowerPipeline(Ast.Pipeline pipeline) {
        String prefix = pipeline.output;
        IrNode current = null;
        for (Ast.Stage stage : pipeline.stages) {
            if (stage instanceof Ast.Name) {
                current = requireTensor(((Ast.Name) stage).name);
                continue;
            }
            current = lowerCall((Ast.Call) stage, current, prefix);
        }
        if (current == null) {
            throw new FrontendException("pipeline '" + prefix + "' has no stages");
        }
        graph.outputs.add(current);
        graph.namedOutputs.put(prefix, current);
        visible.put(prefix, current);
        current.attrs.put(IrNode.ATTR_PIPELINE, prefix);
    }

    private IrNode lowerCall(Ast.Call call, IrNode subject, String prefix) {
        switch (call.op) {
            case "attention":
                return lowerAttention(call, subject, prefix);
            case "MLP":
                return lowerMlp(call, subject, prefix);
            case "matmul":
                return lowerMatmul(call, subject, prefix, false);
            case "matmul_nt":
                return lowerMatmul(call, subject, prefix, true);
            case "bias":
                return lowerBias(call, subject, prefix);
            case "relu":
                return lowerActivation(call, subject, prefix, OpKind.RELU);
            case "gelu":
                return lowerActivation(call, subject, prefix, OpKind.GELU);
            case "softmax":
                return lowerUnary(call, subject, prefix, OpKind.SOFTMAX);
            case "add":
                return lowerAdd(call, subject, prefix);
            case "scale":
                return lowerScale(call, subject, prefix);
            case "layernorm":
                return lowerLayerNorm(call, subject, prefix);
            default:
                throw new FrontendException("unsupported op '" + call.op + "'"
                        + " -- no emitter exists for it (the compiler refuses to guess); supported ops are: "
                        + "attention, MLP, matmul, matmul_nt, bias, relu, gelu, softmax, add, scale, layernorm");
        }
    }

    // ------------------------------------------------------------------- ops

    /**
     * attention(Q, K, V) on [B, D] x [B, D] x [B, D] expands to
     * S = Q x K^T * (1/sqrt(D)); P = softmax(S); O = P x V.
     */
    private IrNode lowerAttention(Ast.Call call, IrNode subject, String prefix) {
        List<IrNode> args = resolveArgs(call, subject, 3, prefix);
        IrNode q = args.get(0);
        IrNode k = args.get(1);
        IrNode v = args.get(2);
        require2d(q, "attention(Q,K,V)");
        require2d(k, "attention(Q,K,V)");
        require2d(v, "attention(Q,K,V)");
        if (q.shape.cols != k.shape.cols) {
            throw new FrontendException("attention: Q has head dim " + q.shape.cols
                    + " but K has " + k.shape.cols);
        }
        if (k.shape.rows != v.shape.rows) {
            throw new FrontendException("attention: K has " + k.shape.rows + " keys but V has "
                    + v.shape.rows + " rows");
        }
        double scale = 1.0 / Math.sqrt(q.shape.cols);

        IrNode scores = create(prefix, "attn_scores", OpKind.MATMUL, new Shape(q.shape.rows, k.shape.rows));
        scores.attrs.put(IrNode.ATTR_TRANS_B, Boolean.TRUE);
        scores.attrs.put(IrNode.ATTR_SCALE, scale);
        scores.attrs.put(IrNode.ATTR_ORIGIN, "attention");
        graph.connect(scores, q, k);

        IrNode probs = create(prefix, "attn_softmax", OpKind.SOFTMAX, scores.shape);
        probs.attrs.put(IrNode.ATTR_ORIGIN, "attention");
        graph.connect(probs, scores);

        IrNode out = create(prefix, "attn_pv", OpKind.MATMUL, new Shape(q.shape.rows, v.shape.cols));
        out.attrs.put(IrNode.ATTR_TRANS_B, Boolean.FALSE);
        out.attrs.put(IrNode.ATTR_ORIGIN, "attention");
        out.attrs.put(IrNode.ATTR_SCALE, 1.0);
        graph.connect(out, probs, v);
        return out;
    }

    /**
     * MLP(W1, act, W2 [, b1 [, b2]]) applied to the pipeline subject:
     * H = act(X x W1 + b1); Y = H x W2 + b2.
     */
    private IrNode lowerMlp(Ast.Call call, IrNode subject, String prefix) {
        if (subject == null) {
            throw new FrontendException("MLP(...) needs an input: use '<value> -> MLP(W1, relu, W2)'");
        }
        if (call.args.size() < 3 || call.args.size() > 5) {
            throw new FrontendException("MLP(W1, act, W2 [, b1 [, b2]]) takes 3..5 arguments, got "
                    + call.args.size());
        }
        IrNode w1 = requireTensor(refName(call.args.get(0), "MLP weight"));
        String actName = refName(call.args.get(1), "MLP activation");
        IrNode w2 = requireTensor(refName(call.args.get(2), "MLP weight"));
        OpKind act = activationKind(actName);

        require2d(subject, "MLP input");
        require2d(w1, "MLP W1");
        require2d(w2, "MLP W2");
        if (subject.shape.cols != w1.shape.rows) {
            throw new FrontendException("MLP: input has " + subject.shape.cols + " features but W1 expects "
                    + w1.shape.rows);
        }
        if (w1.shape.cols != w2.shape.rows) {
            throw new FrontendException("MLP: hidden dim mismatch between W1 (" + w1.shape.cols
                    + ") and W2 (" + w2.shape.rows + ")");
        }

        IrNode fc1 = create(prefix, "mlp_fc1", OpKind.MATMUL, new Shape(subject.shape.rows, w1.shape.cols));
        fc1.attrs.put(IrNode.ATTR_TRANS_B, Boolean.FALSE);
        fc1.attrs.put(IrNode.ATTR_SCALE, 1.0);
        fc1.attrs.put(IrNode.ATTR_ORIGIN, "MLP");
        graph.connect(fc1, subject, w1);

        IrNode hidden = fc1;
        if (call.args.size() >= 4) {
            IrNode b1 = requireTensor(refName(call.args.get(3), "MLP bias"));
            requireBias(b1, hidden.shape.cols, "MLP b1");
            IrNode add = create(prefix, "mlp_b1", OpKind.BIAS_ADD, hidden.shape);
            add.attrs.put(IrNode.ATTR_ORIGIN, "MLP");
            graph.connect(add, hidden, b1);
            hidden = add;
        }

        IrNode activated = create(prefix, "mlp_act", act, hidden.shape);
        activated.attrs.put(IrNode.ATTR_ORIGIN, "MLP");
        graph.connect(activated, hidden);
        hidden = activated;

        IrNode fc2 = create(prefix, "mlp_fc2", OpKind.MATMUL, new Shape(subject.shape.rows, w2.shape.cols));
        fc2.attrs.put(IrNode.ATTR_TRANS_B, Boolean.FALSE);
        fc2.attrs.put(IrNode.ATTR_SCALE, 1.0);
        fc2.attrs.put(IrNode.ATTR_ORIGIN, "MLP");
        graph.connect(fc2, hidden, w2);

        IrNode out = fc2;
        if (call.args.size() == 5) {
            IrNode b2 = requireTensor(refName(call.args.get(4), "MLP bias"));
            requireBias(b2, out.shape.cols, "MLP b2");
            IrNode add = create(prefix, "mlp_b2", OpKind.BIAS_ADD, out.shape);
            add.attrs.put(IrNode.ATTR_ORIGIN, "MLP");
            graph.connect(add, out, b2);
            out = add;
        }
        return out;
    }

    private IrNode lowerMatmul(Ast.Call call, IrNode subject, String prefix, boolean transB) {
        List<IrNode> args = resolveArgs(call, subject, 2, prefix);
        IrNode a = args.get(0);
        IrNode b = args.get(1);
        require2d(a, "matmul A");
        require2d(b, "matmul B");
        Shape result;
        if (transB) {
            if (a.shape.cols != b.shape.cols) {
                throw new FrontendException("matmul_nt: A is " + a.shape + " and B is " + b.shape
                        + "; A.cols must equal B.cols");
            }
            result = new Shape(a.shape.rows, b.shape.rows);
        } else {
            if (a.shape.cols != b.shape.rows) {
                throw new FrontendException("matmul: A is " + a.shape + " and B is " + b.shape
                        + "; A.cols must equal B.rows");
            }
            result = new Shape(a.shape.rows, b.shape.cols);
        }
        IrNode node = create(prefix, transB ? "matmul_nt" : "matmul", OpKind.MATMUL, result);
        node.attrs.put(IrNode.ATTR_TRANS_B, transB);
        node.attrs.put(IrNode.ATTR_SCALE, 1.0);
        graph.connect(node, a, b);
        return node;
    }

    private IrNode lowerBias(Ast.Call call, IrNode subject, String prefix) {
        List<IrNode> args = resolveArgs(call, subject, 2, prefix);
        IrNode x = args.get(0);
        IrNode b = args.get(1);
        require2d(x, "bias");
        requireBias(b, x.shape.cols, "bias");
        IrNode node = create(prefix, "bias", OpKind.BIAS_ADD, x.shape);
        graph.connect(node, x, b);
        return node;
    }

    private IrNode lowerActivation(Ast.Call call, IrNode subject, String prefix, OpKind kind) {
        return lowerUnary(call, subject, prefix, kind);
    }

    private IrNode lowerUnary(Ast.Call call, IrNode subject, String prefix, OpKind kind) {
        List<IrNode> args = resolveArgs(call, subject, 1, prefix);
        IrNode x = args.get(0);
        require2d(x, kind.name().toLowerCase());
        IrNode node = create(prefix, kind.name().toLowerCase(), kind, x.shape);
        graph.connect(node, x);
        return node;
    }

    private IrNode lowerAdd(Ast.Call call, IrNode subject, String prefix) {
        List<IrNode> args = resolveArgs(call, subject, 2, prefix);
        IrNode a = args.get(0);
        IrNode b = args.get(1);
        require2d(a, "add");
        require2d(b, "add");
        Shape shape = broadcastShape(a.shape, b.shape, "add");
        IrNode node = create(prefix, "add", OpKind.ADD, shape);
        graph.connect(node, a, b);
        return node;
    }

    private IrNode lowerScale(Ast.Call call, IrNode subject, String prefix) {
        if (call.args.isEmpty()) {
            throw new FrontendException("scale(x, k) needs a scalar argument");
        }
        Ast.Expr last = call.args.get(call.args.size() - 1);
        double factor;
        if (last instanceof Ast.FloatLit) {
            factor = ((Ast.FloatLit) last).value;
        } else if (last instanceof Ast.IntLit) {
            factor = ((Ast.IntLit) last).value;
        } else {
            throw new FrontendException("scale(x, k): k must be a numeric literal");
        }
        List<IrNode> args = resolveArgs(call, subject, 1, prefix);
        IrNode x = args.get(0);
        require2d(x, "scale");
        IrNode node = create(prefix, "scale", OpKind.SCALE, x.shape);
        node.attrs.put(IrNode.ATTR_SCALE, factor);
        graph.connect(node, x);
        return node;
    }

    private IrNode lowerLayerNorm(Ast.Call call, IrNode subject, String prefix) {
        List<IrNode> args = resolveArgs(call, subject, 3, prefix);
        IrNode x = args.get(0);
        IrNode gain = args.get(1);
        IrNode bias = args.get(2);
        require2d(x, "layernorm");
        requireBias(gain, x.shape.cols, "layernorm gain");
        requireBias(bias, x.shape.cols, "layernorm bias");
        IrNode node = create(prefix, "layernorm", OpKind.LAYERNORM, x.shape);
        node.attrs.put(IrNode.ATTR_SCALE, 1e-5);
        graph.connect(node, x, gain, bias);
        return node;
    }

    // --------------------------------------------------------------- helpers

    private IrNode create(String prefix, String suffix, OpKind op, Shape shape) {
        String base = prefix.isEmpty() ? suffix : prefix + "." + suffix;
        IrNode node = graph.create(graph.uniqueName(base), op);
        node.shape = shape;
        node.store = defaultStore;
        node.accum = defaultAccum;
        node.attrs.put(IrNode.ATTR_PIPELINE, prefix);
        return node;
    }

    /**
     * Resolves the tensor arguments of a call. When only {@code arity - 1} arguments are
     * given, the pipeline subject is prepended: {@code X -> bias(b1)} means {@code bias(X, b1)}.
     */
    private List<IrNode> resolveArgs(Ast.Call call, IrNode subject, int arity, String prefix) {
        List<Ast.Expr> exprs = new ArrayList<>(call.args);
        List<IrNode> out = new ArrayList<>();
        if (exprs.size() == arity - 1 && subject != null) {
            out.add(subject);
        } else if (exprs.size() != arity) {
            throw new FrontendException(call.op + "(...) takes " + arity + " tensor argument(s), got "
                    + exprs.size() + (subject != null ? " (a pipeline subject is available and may replace one)" : ""));
        }
        for (Ast.Expr e : exprs) {
            out.add(requireTensor(refName(e, call.op + " argument")));
        }
        return out;
    }

    private String refName(Ast.Expr expr, String context) {
        if (expr instanceof Ast.Ref) {
            return ((Ast.Ref) expr).name;
        }
        throw new FrontendException(context + " must be a tensor or activation name");
    }

    private IrNode requireTensor(String name) {
        IrNode node = visible.get(name);
        if (node == null) {
            throw new FrontendException("unknown tensor '" + name + "'");
        }
        return node;
    }

    private OpKind activationKind(String name) {
        if ("relu".equals(name)) {
            return OpKind.RELU;
        }
        if ("gelu".equals(name)) {
            return OpKind.GELU;
        }
        throw new FrontendException("unsupported activation '" + name + "' (expected relu or gelu)");
    }

    private void require2d(IrNode node, String context) {
        if (node.shape == null) {
            throw new FrontendException(context + ": tensor '" + node.name + "' has no shape");
        }
    }

    private void requireBias(IrNode b, int cols, String context) {
        if (b.shape.rows != 1 && b.shape.rows != cols) {
            throw new FrontendException(context + ": bias tensor '" + b.name + "' has shape " + b.shape
                    + ", expected [" + cols + "] or [1," + cols + "]");
        }
        if (b.shape.rows == cols && b.shape.cols != 1 && b.shape.cols != cols) {
            throw new FrontendException(context + ": bias tensor '" + b.name + "' has shape " + b.shape
                    + ", expected [" + cols + "]");
        }
        int elements = (int) b.shape.elements();
        if (elements != cols) {
            throw new FrontendException(context + ": bias tensor '" + b.name + "' has " + elements
                    + " elements, expected " + cols);
        }
    }

    private Shape broadcastShape(Shape a, Shape b, String context) {
        if (a.equals(b)) {
            return a;
        }
        throw new FrontendException(context + ": shape mismatch " + a + " vs " + b
                + " (residual adds must be element-wise; broadcasting a bias is 'bias(x, b)')");
    }
}
