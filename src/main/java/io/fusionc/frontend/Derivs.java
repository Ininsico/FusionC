package io.fusionc.frontend;

import java.util.IdentityHashMap;

/**
 * Identity map from a semantic value produced by a CUP action to the derivation node for that
 * reduction. The actions put their result here, so a parent production can find the child
 * nodes when it builds its own. Reset before every parse.
 */
public final class Derivs {

    private static final IdentityHashMap<Object, Deriv> NODES = new IdentityHashMap<>();

    private Derivs() {
    }

    public static void reset() {
        NODES.clear();
    }

    public static Deriv of(Object value) {
        Deriv node = NODES.get(value);
        if (node == null) {
            return Deriv.terminal("(unknown)", String.valueOf(value));
        }
        return node;
    }

    public static void put(Object value, Deriv node) {
        NODES.put(value, node);
    }
}
