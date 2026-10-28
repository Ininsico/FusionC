package io.fusionc.frontend;

/**
 * Parse-time announcements, in the style of the CyberScript exercise: the grammar's
 * semantic actions print what they just recognised. The compiler keeps this off; the
 * ParseTree demo turns it on so the parse can be watched as it happens.
 */
public final class ParseTrace {

    public static boolean on = false;

    private ParseTrace() {
    }

    public static void say(String message) {
        if (on) {
            System.out.println("  " + message);
        }
    }
}
