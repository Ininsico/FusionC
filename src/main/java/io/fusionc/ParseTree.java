package io.fusionc;

import io.fusionc.frontend.DslParser;
import io.fusionc.frontend.ParseTrace;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * Front-end demo, in the style of the CyberScript exercise: the CUP grammar's semantic
 * actions announce each construct while the input is parsed, and the parse result is then
 * printed directly with {@code System.out.println(result)} (the tree object's toString()).
 *
 * <pre>
 *   java -cp build/fusionc.jar io.fusionc.ParseTree examples/attention_mlp.dsl
 *   .\scripts\parse-tree.ps1 examples\syntax_error.dsl
 * </pre>
 *
 * Exit status: 0 = parsed, 1 = lexical/syntax error, 2 = usage/file error.
 */
public final class ParseTree {

    private ParseTree() {
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("usage: java -cp build/fusionc.jar io.fusionc.ParseTree <file.dsl>");
            System.exit(2);
        }
        File file = new File(args[0]);
        if (!file.isFile()) {
            System.err.println("error: no such file: " + file.getPath());
            System.exit(2);
        }

        try {
            System.out.println("=== token stream (" + file.getName() + ") ===");
            printTokens(file);
            System.out.println();
            System.out.println("=== parse (the grammar actions announce each construct) ===");
            ParseTrace.on = true;
            Object result = DslParser.parseFile(file);
            ParseTrace.on = false;
            System.out.println();
            System.out.println("=== parse tree ===");
            System.out.println(result);
            System.out.println("parsed OK");
        } catch (FrontendException e) {
            ParseTrace.on = false;
            System.err.println("error: " + e.getMessage());
            System.exit(1);
        } catch (IOException e) {
            System.err.println("error: cannot read " + file.getPath() + ": " + e.getMessage());
            System.exit(2);
        }
    }

    /** One line per token: index, line:column, token type, spelling. */
    private static void printTokens(File file) throws IOException {
        io.fusionc.frontend.Scanner scanner =
                new io.fusionc.frontend.Scanner(new FileReader(file, StandardCharsets.UTF_8));
        java_cup.runtime.Symbol tok;
        int index = 0;
        while ((tok = scanner.next_token()).sym != io.fusionc.frontend.sym.EOF) {
            String text = tok.value != null ? tok.value.toString() : "";
            System.out.printf("%3d  line %2d col %2d  %-10s %s%n",
                    ++index, tok.left, tok.right,
                    io.fusionc.frontend.sym.terminalNames[tok.sym], text);
        }
        System.out.printf("(%d tokens)%n", index);
    }
}
