package io.fusionc.frontend;

import java.util.ArrayList;
import java.util.List;

/**
 * One node of the derivation tree the CUP actions build while parsing: a grammar symbol, its
 * children, and -- for terminals -- the token spelling. The GUI draws this tree top-down.
 */
public final class Deriv {

    public final String symbol;
    /** Terminal spelling (e.g. "SEQ", "->", ";"); null for nonterminal nodes. */
    public final String text;
    public final List<Deriv> children = new ArrayList<>();

    private Deriv(String symbol, String text) {
        this.symbol = symbol;
        this.text = text;
    }

    public static Deriv rule(String symbol, Deriv... kids) {
        Deriv node = new Deriv(symbol, null);
        for (Deriv kid : kids) {
            node.children.add(kid);
        }
        return node;
    }

    public static Deriv terminal(String symbol, String text) {
        return new Deriv(symbol, text);
    }

    public boolean isTerminal() {
        return text != null;
    }

    public int count() {
        int n = 1;
        for (Deriv child : children) {
            n += child.count();
        }
        return n;
    }

    public int depth() {
        int d = 0;
        for (Deriv child : children) {
            d = Math.max(d, 1 + child.depth());
        }
        return d;
    }

    /** Indented text form (used by the headless selftest). */
    public String dump() {
        StringBuilder sb = new StringBuilder();
        dump(sb, 0);
        return sb.toString();
    }

    private void dump(StringBuilder sb, int level) {
        for (int i = 0; i < level; i++) {
            sb.append("  ");
        }
        sb.append(symbol);
        if (text != null) {
            sb.append(": ").append(text);
        }
        sb.append('\n');
        for (Deriv child : children) {
            child.dump(sb, level + 1);
        }
    }

    @Override
    public String toString() {
        return isTerminal() ? symbol + ": " + text : symbol;
    }
}
