package io.fusionc;

import io.fusionc.ast.Ast;
import io.fusionc.backend.CudaEmitter;
import io.fusionc.frontend.DslParser;
import io.fusionc.ir.IrGraph;
import io.fusionc.pass.FusionDecision;
import io.fusionc.pass.FusionGroup;
import io.fusionc.pass.FusionPass;
import io.fusionc.pass.PascalHeuristics;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;

/**
 * FusionC command line driver: DSL -> JFlex/CUP front end -> IR -> fusion -> CUDA C.
 */
public final class Main {

    private static final String USAGE =
        "FusionC -- source-to-source transformer-inference compiler for Pascal (sm_61)\n"
      + "\n"
      + "usage: fusionc --input <graph.dsl> [--fuse | --no-fusion] -o <out.cu> [options]\n"
      + "\n"
      + "  --input <file>        DSL source (required)\n"
      + "  --fuse                fuse operations under the Pascal resource heuristics (default)\n"
      + "  --no-fusion           baseline: exactly one kernel per operation\n"
      + "  -o <file>             write the generated CUDA C here (required)\n"
      + "  --decisions <file>    write the fusion decision log as JSON\n"
      + "  --ir <file>           write the annotated IR (op, shape, dtype, placement, deps)\n"
      + "  --sm-count <n>        SM count used by occupancy heuristics (default 6 = GTX 1050 Ti)\n"
      + "  --force-fusion        ignore profitability guards (resource caps still enforced)\n"
      + "  --dump-ast            print the parsed AST\n"
      + "  --dump-tokens         print the JFlex token stream (lexer output)\n"
      + "  --quiet               do not print the decision log to stdout\n";

    public static void main(String[] args) {
        String input = null;
        String output = null;
        String decisionsFile = null;
        String irFile = null;
        boolean fused = true;
        boolean force = false;
        boolean dumpAst = false;
        boolean dumpTokens = false;
        boolean quiet = false;

        try {
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--input":
                    case "-i":
                        input = require(args, ++i, "--input");
                        break;
                    case "--fuse":
                        fused = true;
                        break;
                    case "--no-fusion":
                        fused = false;
                        break;
                    case "-o":
                    case "--output":
                        output = require(args, ++i, "-o");
                        break;
                    case "--decisions":
                        decisionsFile = require(args, ++i, "--decisions");
                        break;
                    case "--ir":
                        irFile = require(args, ++i, "--ir");
                        break;
                    case "--sm-count":
                        PascalHeuristics.targetSmCount = Integer.parseInt(require(args, ++i, "--sm-count"));
                        break;
                    case "--force-fusion":
                        force = true;
                        break;
                    case "--dump-ast":
                        dumpAst = true;
                        break;
                    case "--dump-tokens":
                        dumpTokens = true;
                        break;
                    case "--quiet":
                        quiet = true;
                        break;
                    case "-h":
                    case "--help":
                        System.out.print(USAGE);
                        return;
                    default:
                        throw new FrontendException("unknown argument '" + args[i] + "'\n\n" + USAGE);
                }
            }
            if (input == null || output == null) {
                throw new FrontendException("--input and -o are required\n\n" + USAGE);
            }

            File inputFile = new File(input);
            if (dumpTokens) {
                try {
                    dumpTokens(inputFile, System.out);
                } catch (IOException e) {
                    throw new FrontendException("cannot read " + inputFile.getName() + ": " + e.getMessage());
                }
            }
            Ast.Program program = DslParser.parseFile(inputFile);
            if (dumpAst) {
                System.out.print(Ast.dump(program));
            }
            IrGraph graph = io.fusionc.semantics.GraphBuilder.build(program);
            FusionPass pass = new FusionPass(graph, fused, force);
            List<FusionGroup> groups = pass.run();

            if (irFile != null) {
                write(new File(irFile), graph.dump());
            }

            CudaEmitter emitter = new CudaEmitter(graph, groups, fused);
            String source = emitter.emit(inputFile.getName());
            write(new File(output), source);

            String json = decisionJson(inputFile.getName(), fused, graph, groups, pass.decisions());
            if (decisionsFile != null) {
                write(new File(decisionsFile), json);
            }

            if (!quiet) {
                logDecisions(System.out, groups, pass.decisions(), fused, graph);
            }
            System.out.printf("emitted %s: %d kernel(s), %d IR node(s), %d decision(s)%n",
                    output, countKernels(groups), graph.nodes.size(), pass.decisions().size());
        } catch (FrontendException e) {
            System.err.println("error: " + e.getMessage());
            System.exit(1);
        } catch (IOException e) {
            System.err.println("error: cannot write output: " + e.getMessage());
            System.exit(1);
        }
    }

    /** Print the JFlex token stream: index, position, token type, semantic value. */
    private static void dumpTokens(File input, PrintStream out) throws IOException {
        io.fusionc.frontend.Scanner scanner =
                new io.fusionc.frontend.Scanner(new java.io.FileReader(input, StandardCharsets.UTF_8));
        java_cup.runtime.Symbol tok;
        int index = 0;
        while ((tok = scanner.next_token()).sym != io.fusionc.frontend.sym.EOF) {
            String text = tok.value != null ? tok.value.toString() : "";
            out.printf("%3d  line %2d col %2d  %-10s %s%n", ++index, tok.left, tok.right,
                    io.fusionc.frontend.sym.terminalNames[tok.sym], text);
        }
        out.printf("(%d tokens)%n", index);
    }

    private static int countKernels(List<FusionGroup> groups) {
        int n = 0;
        for (FusionGroup g : groups) {
            if (!g.members.get(0).isLeaf()) {
                n++;
            }
        }
        return n;
    }

    private static void logDecisions(PrintStream out, List<FusionGroup> groups,
                                     List<FusionDecision> decisions, boolean fused, IrGraph graph) {
        out.println("=== fusion decisions (" + (fused ? "--fuse" : "--no-fusion")
                + ", sm_61, target " + PascalHeuristics.targetSmCount + " SMs) ===");
        for (FusionDecision d : decisions) {
            out.printf("%-8s %-18s %s%n", d.accepted ? "FUSED" : "REJECT", d.kind,
                    String.join(" + ", d.members));
            StringBuilder pred = new StringBuilder();
            if (d.predicted.containsKey("smemBytes")) {
                pred.append(String.format("smem=%.1fKB ", d.predicted.get("smemBytes") / 1024.0));
            }
            if (d.predicted.containsKey("regsPerThread")) {
                pred.append("regs=").append(d.predicted.get("regsPerThread")).append(' ');
            }
            if (d.predicted.containsKey("occupancyMilli")) {
                pred.append("occupancy=")
                    .append(String.format("%.2f", d.predicted.get("occupancyMilli") / 1000.0)).append(' ');
            }
            if (d.gridBlocks > 0) {
                pred.append("grid=").append(d.gridBlocks).append(' ');
            }
            if (d.bytesSaved > 0) {
                pred.append(String.format("dram_saved=%.1fKB", d.bytesSaved / 1024.0));
            }
            if (pred.length() > 0) {
                out.println("         predicted: " + pred.toString().trim());
            }
            out.println("         reason   : " + d.reason);
        }
    }

    private static String decisionJson(String source, boolean fused, IrGraph graph,
                                       List<FusionGroup> groups, List<FusionDecision> decisions) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"source\": \"").append(source).append("\",\n");
        sb.append("  \"mode\": \"").append(fused ? "fuse" : "no-fusion").append("\",\n");
        sb.append("  \"target\": \"sm_61\",\n");
        sb.append("  \"smCount\": ").append(PascalHeuristics.targetSmCount).append(",\n");
        sb.append("  \"irNodes\": ").append(graph.nodes.size()).append(",\n");
        sb.append("  \"regions\": [\n");
        List<FusionGroup> emitted = new java.util.ArrayList<>();
        for (FusionGroup g : groups) {
            if (!g.members.get(0).isLeaf()) {
                emitted.add(g);
            }
        }
        for (int i = 0; i < emitted.size(); i++) {
            FusionGroup g = emitted.get(i);
            sb.append("    { \"region\": \"").append(g.regionName()).append("\", \"kernel\": \"")
              .append(FusionGroup.kernelName(g.kind, g.id, g.members)).append("\", \"kind\": \"")
              .append(g.kind.label).append("\", \"members\": [");
            for (int j = 0; j < g.members.size(); j++) {
                sb.append(j > 0 ? ", " : "").append('"').append(g.members.get(j).name).append('"');
            }
            sb.append("] }").append(i + 1 < emitted.size() ? "," : "").append('\n');
        }
        sb.append("  ],\n");
        sb.append("  \"decisions\": [\n");
        for (int i = 0; i < decisions.size(); i++) {
            sb.append(decisions.get(i).toJson()).append(i + 1 < decisions.size() ? "," : "").append('\n');
        }
        sb.append("  ]\n");
        sb.append("}\n");
        return sb.toString();
    }

    private static String require(String[] args, int index, String flag) {
        if (index >= args.length) {
            throw new FrontendException(flag + " requires a value");
        }
        return args[index];
    }

    private static void write(File file, String content) throws IOException {
        File parent = file.getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("cannot create directory " + parent);
        }
        Files.write(file.toPath(), content.getBytes(StandardCharsets.UTF_8));
    }
}
