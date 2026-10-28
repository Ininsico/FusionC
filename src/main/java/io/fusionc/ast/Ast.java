package io.fusionc.ast;

import java.util.ArrayList;
import java.util.List;

/**
 * AST produced by the CUP-generated LALR parser. Everything here is syntax only:
 * no shapes, no dtypes, no semantics (those live in the IR).
 */
public final class Ast {

    private Ast() {
    }

    // ------------------------------------------------------------------ program

    public static final class Program {
        public final List<Decl> decls;
        public final List<Pipeline> pipelines;

        public Program(List<Decl> decls, List<Pipeline> pipelines) {
            this.decls = decls;
            this.pipelines = pipelines;
        }
    }

    // ------------------------------------------------------------- declarations

    public abstract static class Decl {
    }

    public static final class ConstDecl extends Decl {
        public final String name;
        public final long value;

        public ConstDecl(String name, long value, int line) {
            this.name = name;
            this.value = value;
        }
    }

    public static final class DtypeDecl extends Decl {
        public final String store;
        public final String accum;

        public DtypeDecl(String store, String accum) {
            this.store = store;
            this.accum = accum;
        }
    }

    public static final class TensorDecl extends Decl {
        public final String name;
        public final String dtype;
        public final List<Expr> dims;
        public final boolean input;

        private TensorDecl(String name, String dtype, List<Expr> dims, boolean input) {
            this.name = name;
            this.dtype = dtype;
            this.dims = dims;
            this.input = input;
        }

        public static TensorDecl host(String name, String dtype, List<Expr> dims, boolean input) {
            return new TensorDecl(name, dtype, dims, input);
        }
    }

    // ------------------------------------------------------------------ exprs

    public abstract static class Expr {
    }

    public static final class IntLit extends Expr {
        public final long value;

        public IntLit(long value) {
            this.value = value;
        }
    }

    public static final class FloatLit extends Expr {
        public final double value;

        public FloatLit(double value) {
            this.value = value;
        }
    }

    public static final class Ref extends Expr {
        public final String name;

        public Ref(String name) {
            this.name = name;
        }
    }

    // ------------------------------------------------------------ pipelines

    public abstract static class Stage {
    }

    public static final class Name extends Stage {
        public final String name;

        public Name(String name) {
            this.name = name;
        }
    }

    public static final class Call extends Stage {
        public final String op;
        public final List<Expr> args;

        public Call(String op, List<Expr> args, int line, int column) {
            this.op = op;
            this.args = args;
        }
    }

    public static final class Pipeline {
        public final String output;
        public final List<Stage> stages;

        public Pipeline(String output, List<Stage> stages, int line) {
            this.output = output;
            this.stages = stages;
        }
    }

    // ------------------------------------------------------------------ dumps

    public static String dump(Program program) {
        StringBuilder sb = new StringBuilder();
        sb.append("Program\n");
        for (Decl d : program.decls) {
            if (d instanceof ConstDecl) {
                ConstDecl c = (ConstDecl) d;
                sb.append("  const ").append(c.name).append(" = ").append(c.value).append('\n');
            } else if (d instanceof DtypeDecl) {
                DtypeDecl t = (DtypeDecl) d;
                sb.append("  dtype store=").append(t.store).append(" accum=").append(t.accum).append('\n');
            } else {
                TensorDecl t = (TensorDecl) d;
                sb.append("  ").append(t.input ? "input " : "param ").append(t.name)
                  .append(" : ").append(t.dtype).append(' ')
                  .append(dimsToString(t.dims)).append('\n');
            }
        }
        for (Pipeline p : program.pipelines) {
            sb.append("  pipeline ").append(p.output).append(" = ");
            List<String> parts = new ArrayList<>();
            for (Stage s : p.stages) {
                parts.add(s instanceof Call ? dumpCall((Call) s) : ((Name) s).name);
            }
            sb.append(String.join(" -> ", parts)).append('\n');
        }
        return sb.toString();
    }

    private static String dumpCall(Call c) {
        List<String> args = new ArrayList<>();
        for (Expr e : c.args) {
            if (e instanceof IntLit) {
                args.add(Long.toString(((IntLit) e).value));
            } else if (e instanceof FloatLit) {
                args.add(Double.toString(((FloatLit) e).value));
            } else {
                args.add(((Ref) e).name);
            }
        }
        return c.op + "(" + String.join(", ", args) + ")";
    }

    private static String dimsToString(List<Expr> dims) {
        List<String> parts = new ArrayList<>();
        for (Expr e : dims) {
            if (e instanceof IntLit) {
                parts.add(Long.toString(((IntLit) e).value));
            } else if (e instanceof FloatLit) {
                parts.add(Double.toString(((FloatLit) e).value));
            } else {
                parts.add(((Ref) e).name);
            }
        }
        return "[" + String.join(", ", parts) + "]";
    }
}
