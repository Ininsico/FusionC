package io.fusionc.pass;

import io.fusionc.ir.IrGraph;
import io.fusionc.ir.IrNode;
import io.fusionc.ir.MemSpace;
import io.fusionc.ir.OpKind;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The fusion pass: merges connected subgraphs of the primitive DAG into single kernels under
 * Pascal resource heuristics.
 *
 * <p>Three patterns are recognised, in priority order:
 * <ol>
 *   <li>{@code QK^T -> softmax -> PV}   (fused attention, flash-style, no cp.async),</li>
 *   <li>{@code GEMM -> activation -> GEMM} (fused MLP, intermediate tile resident in smem),</li>
 *   <li>{@code GEMM -> elementwise epilogue} (bias / activation / scale, resident in registers).</li>
 * </ol>
 *
 * <p>Every candidate is logged with predicted shared memory, registers, occupancy, grid size and a
 * human-readable reason, whether it was accepted <em>or</em> rejected. Rejections are as much a
 * compiler decision as fusions are, so they are reported the same way.
 */
public final class FusionPass {

    private final IrGraph graph;
    private final int smCount;
    private final boolean force;
    private final boolean fused;

    private final List<FusionGroup> groups = new ArrayList<>();
    private final List<FusionDecision> decisions = new ArrayList<>();
    private final Map<IrNode, FusionGroup> owner = new IdentityHashMap<>();
    private final Set<String> attempted = new HashSet<>();

    private int nextGroupId;

    public FusionPass(IrGraph graph, boolean fused, boolean force) {
        this.graph = graph;
        this.fused = fused;
        this.smCount = PascalHeuristics.targetSmCount;
        this.force = force;
    }

    public List<FusionGroup> run() {
        if (fused) {
            for (int iteration = 0; iteration < PascalHeuristics.FUSION_FIXPOINT_ITERATIONS; iteration++) {
                boolean changed = false;
                changed |= fuseAttention();
                changed |= fuseProducerConsumerGemm();
                changed |= fuseGemmEpilogue();
                changed |= fuseElementwiseChains();
                if (!changed) {
                    break;
                }
            }
        }
        finaliseSingletons();
        for (FusionGroup group : groups) {
            group.setGraphOutputs(graph.outputs);
            group.classify();
            applyPlacement(group);
        }
        applyNumericPolicy();
        return groups;
    }

    public List<FusionDecision> decisions() {
        return decisions;
    }

    // ------------------------------------------------------------------ pattern 1

    private boolean fuseAttention() {
        boolean changed = false;
        for (IrNode scores : graph.nodes) {
            if (owner.containsKey(scores) || scores.op != OpKind.MATMUL || !scores.transB()) {
                continue;
            }
            if (scores.shape.rows != scores.shape.cols || scores.consumers.size() != 1) {
                continue;
            }
            IrNode probs = scores.consumers.get(0);
            if (probs.op != OpKind.SOFTMAX || owner.containsKey(probs) || probs.consumers.size() != 1) {
                continue;
            }
            IrNode pv = probs.consumers.get(0);
            if (pv.op != OpKind.MATMUL || pv.transB() || owner.containsKey(pv) || pv.inputs.get(0) != probs) {
                continue;
            }
            IrNode q = scores.inputs.get(0);
            IrNode k = scores.inputs.get(1);
            IrNode v = pv.inputs.get(1);
            int b = q.shape.rows;
            int d = q.shape.cols;
            if (k.shape.rows != b || k.shape.cols != d || v.shape.rows != b || pv.shape.cols != d) {
                continue;
            }
            if (!attempted.add("attn:" + scores.name)) {
                continue;
            }
            if (acceptAttention(scores, probs, pv, b, d)) {
                changed = true;
            }
        }
        return changed;
    }

    private boolean acceptAttention(IrNode scores, IrNode probs, IrNode pv, int b, int d) {
        int minGrid = PascalHeuristics.MIN_GRID_BLOCKS_PER_SM * smCount;
        long saved = 2 * scores.bytes() + 2 * probs.bytes();
        int bestGrid = 0;
        int bestRpb = 0;

        if (b > MAX_ATTENTION_KEY_BAND) {
            FusionDecision decision = new FusionDecision(-1, GroupKind.ATTENTION_FUSED.label, false);
            decision.members.add(scores.name);
            decision.members.add(probs.name);
            decision.members.add(pv.name);
            decision.bytesSaved = saved;
            decision.reason = "rejected: " + b + " keys exceed the " + MAX_ATTENTION_KEY_BAND
                    + "-key band the Pascal fused-attention emitter supports (wider rows would need online "
                    + "softmax rescaling across bands); keeping 3 unfused kernels.";
            decisions.add(decision);
            return false;
        }

        for (int rpb : new int[]{8, 4, 2, 1}) {
            int grid = ceilDiv(b, rpb);
            if (grid > bestGrid) {
                bestGrid = grid;
                bestRpb = rpb;
            }
            for (int dc : new int[]{PascalHeuristics.ATTN_DIM_CHUNK, 32}) {
                if (dc > d || d % dc != 0) {
                    continue;
                }
                ResourceModel.Est est = ResourceModel.attention(b, d, rpb, dc);
                if (!ResourceModel.fits(est)) {
                    continue;
                }
                if (grid < minGrid && !force) {
                    break;
                }
                FusionGroup group = new FusionGroup(nextGroupId++, GroupKind.ATTENTION_FUSED);
                group.members.addAll(Arrays.asList(scores, probs, pv));
                group.params.put("rowsPerBlock", rpb);
                group.params.put("dimChunk", dc);
                group.params.put("seq", b);
                group.params.put("headDim", d);
                group.params.put("threads", rpb * PascalHeuristics.WARP_SIZE);
                group.params.put("smemBytes", est.smemBytes);
                FusionDecision decision = new FusionDecision(group.id, GroupKind.ATTENTION_FUSED.label, true);
                decision.members.add(scores.name);
                decision.members.add(probs.name);
                decision.members.add(pv.name);
                decision.predicted.put("smemBytes", est.smemBytes);
                decision.predicted.put("regsPerThread", est.regsPerThread);
                decision.predicted.put("occupancyMilli", (int) Math.round(est.occupancy * 1000));
                decision.predicted.put("gridBlocks", est.gridBlocks);
                decision.predicted.put("rowsPerBlock", rpb);
                decision.predicted.put("dimChunk", dc);
                decision.metrics.put("bytesSaved", (double) saved);
                decision.metrics.put("launchesSaved", 2.0);
                decision.bytesSaved = saved;
                decision.gridBlocks = est.gridBlocks;
                decision.reason = String.format(java.util.Locale.ROOT,
                        "QK^T+softmax+PV is a reduction-free chain: fusing removes a [%dx%d] fp32 score tile "
                        + "and a [%dx%d] fp16 probability tile from DRAM (%.1f KB saved) plus 2 launches; "
                        + "%s; %d blocks >= %d floor (%d/SM x %d SMs); %d query rows/block (one warp each), "
                        + "K/V staged in %d-wide dimension chunks through a doubled shared buffer "
                        + "(no cp.async on Pascal -> next chunk issued before the current chunk's FMAs)",
                        b, b, b, b, saved / 1024.0, ResourceModel.describe(est), est.gridBlocks, minGrid,
                        PascalHeuristics.MIN_GRID_BLOCKS_PER_SM, smCount, rpb, dc);
                group.decision = decision;
                register(group);
                decisions.add(decision);
                return true;
            }
        }

        FusionDecision decision = new FusionDecision(-1, GroupKind.ATTENTION_FUSED.label, false);
        decision.members.add(scores.name);
        decision.members.add(probs.name);
        decision.members.add(pv.name);
        decision.bytesSaved = saved;
        decision.gridBlocks = bestGrid;
        decision.reason = String.format(java.util.Locale.ROOT,
                "rejected: no feasible attention configuration. Best grid %d blocks (rowsPerBlock=%d) is below "
                + "the %d-block occupancy floor (%d/SM x %d SMs), so the fused kernel would idle the device; "
                + "smaller dimension chunks cannot recover the missing parallelism. Keeping 3 unfused kernels.",
                bestGrid, bestRpb, minGrid, PascalHeuristics.MIN_GRID_BLOCKS_PER_SM, smCount);
        decision.metrics.put("bytesSaved", (double) saved);
        decisions.add(decision);
        return false;
    }

    /** Widest key band the fused-attention emitter handles without online-softmax rescaling. */
    private static final int MAX_ATTENTION_KEY_BAND = 64;

    // ------------------------------------------------------------------ pattern 2

    private boolean fuseProducerConsumerGemm() {
        boolean changed = false;
        for (IrNode producer : graph.nodes) {
            if (owner.containsKey(producer) || producer.op != OpKind.MATMUL || producer.transB()) {
                continue;
            }
            List<IrNode> chain = new ArrayList<>();
            IrNode cursor = producer;
            while (cursor.consumers.size() == 1) {
                IrNode next = cursor.consumers.get(0);
                if (owner.containsKey(next) || next.inputs.get(0) != cursor) {
                    break;
                }
                if (isActivation(next.op) || next.op == OpKind.BIAS_ADD) {
                    if (!leafOnlyExtras(next, cursor)) {
                        break;
                    }
                    chain.add(next);
                    cursor = next;
                    continue;
                }
                if (next.op == OpKind.MATMUL && !next.transB() && next.shape.rows == producer.shape.rows) {
                    int hidden = producer.shape.cols;
                    if (next.inputs.get(0) != cursor || next.inputs.get(1).shape.rows != hidden) {
                        break;
                    }
                    if (chain.stream().noneMatch(n -> isActivation(n.op))) {
                        break;
                    }
                    String key = "mlp:" + producer.name;
                    if (!attempted.add(key)) {
                        break;
                    }
                    if (acceptMlp(producer, chain, next)) {
                        changed = true;
                    }
                    break;
                }
                break;
            }
        }
        return changed;
    }

    private boolean acceptMlp(IrNode producer, List<IrNode> chain, IrNode consumer) {
        int m = producer.shape.rows;
        int d = producer.shape.cols;
        int f = consumer.shape.cols;
        int hidden = producer.shape.cols;
        long hiddenBytes = (long) m * hidden * producer.store.sizeBytes;
        long saved = 2 * hiddenBytes + hiddenBytes * chain.size();
        int minGrid = PascalHeuristics.MIN_GRID_BLOCKS_PER_SM * smCount;

        int[][] candidates = {
            {PascalHeuristics.MLP_RM, PascalHeuristics.MLP_CN, PascalHeuristics.MLP_HF},
            {PascalHeuristics.MLP_RM, PascalHeuristics.MLP_CN / 2, PascalHeuristics.MLP_HF},
            {PascalHeuristics.MLP_RM / 2, PascalHeuristics.MLP_CN, PascalHeuristics.MLP_HF},
            {PascalHeuristics.MLP_RM / 2, PascalHeuristics.MLP_CN / 2, PascalHeuristics.MLP_HF},
            {PascalHeuristics.MLP_RM / 2, PascalHeuristics.MLP_CN, 2 * PascalHeuristics.MLP_HF},
            {PascalHeuristics.MLP_RM / 4, PascalHeuristics.MLP_CN, PascalHeuristics.MLP_HF},
        };
        String rejectWhy = null;
        ResourceModel.Est best = null;
        int[] bestCfg = null;
        for (int[] cfg : candidates) {
            if (m % cfg[0] != 0 || f % cfg[1] != 0 || hidden % cfg[2] != 0
                    || d % PascalHeuristics.MLP_BK != 0) {
                rejectWhy = "candidate tile " + cfg[0] + "x" + cfg[1] + " hidden chunk " + cfg[2]
                        + " does not divide the problem (M=" + m + ", N=" + f + ", hidden=" + hidden
                        + ", K=" + d + ") exactly, and the emitter does not generate partial tiles";
                continue;
            }
            ResourceModel.Est est = ResourceModel.mlp(m, f, cfg[0], cfg[1], PascalHeuristics.MLP_BK, cfg[2]);
            if (!ResourceModel.fits(est)) {
                rejectWhy = "candidate tile " + cfg[0] + "x" + cfg[1] + " (hidden chunk " + cfg[2] + ") needs "
                        + String.format(java.util.Locale.ROOT, "%.1f KB smem", est.smemBytes / 1024.0)
                        + ", over the " + (PascalHeuristics.MAX_SMEM_BYTES_PER_BLOCK * PascalHeuristics.SMEM_SAFETY_MARGIN / 1024)
                        + " KB usable cap";
                continue;
            }
            if (best == null || est.occupancy > best.occupancy) {
                best = est;
                bestCfg = cfg;
            }
            int accum = ResourceModel.accumRegsForMlp(cfg[0], cfg[1]);
            if (accum < MIN_ACCUM_REGS_PER_THREAD) {
                rejectWhy = "candidate tile " + cfg[0] + "x" + cfg[1] + " leaves only " + accum
                        + " accumulator registers per thread (< " + MIN_ACCUM_REGS_PER_THREAD
                        + "): the inner loop would be shared-memory bound";
                continue;
            }
            // The fused kernel keeps the hidden tile in shared memory, but it still has to stream
            // both weight matrices from L2 once per row-block. If its row tile is smaller than the
            // unfused GEMM tile, the weights are re-read more often than the baseline would read
            // them, and the extra L2 traffic can outweigh the DRAM traffic the fusion removes.
            int amplifiedFused = ceilDiv(m, cfg[0]);
            int amplifiedUnfused = ceilDiv(m, PascalHeuristics.GEMM_BM);
            if (amplifiedFused > amplifiedUnfused * PascalHeuristics.MAX_PRODUCER_REREAD_AMPLIFICATION
                    && !force) {
                rejectWhy = "row tile " + cfg[0] + " splits " + m + " rows into " + amplifiedFused
                        + " row-blocks that each stream W1 and W2 again (" + amplifiedFused
                        + "x weight traffic) whereas the unfused GEMM tiling pays "
                        + amplifiedUnfused + "x; over the "
                        + PascalHeuristics.MAX_PRODUCER_REREAD_AMPLIFICATION + "x amplification cap";
                continue;
            }
            if (est.gridBlocks < minGrid && !force) {
                rejectWhy = "best feasible tile " + cfg[0] + "x" + cfg[1] + " gives " + est.gridBlocks
                        + " blocks, below the " + minGrid + "-block occupancy floor ("
                        + PascalHeuristics.MIN_GRID_BLOCKS_PER_SM + "/SM x " + smCount
                        + " SMs); the fused kernel could not fill the device";
                continue;
            }
            FusionGroup group = new FusionGroup(nextGroupId++, GroupKind.GEMM_GEMM_FUSED);
            group.members.add(producer);
            group.members.addAll(chain);
            group.members.add(consumer);
            group.params.put("rm", cfg[0]);
            group.params.put("cn", cfg[1]);
            group.params.put("bk", PascalHeuristics.MLP_BK);
            group.params.put("hf", cfg[2]);
            group.params.put("threads", PascalHeuristics.MLP_THREADS);
            group.params.put("smemBytes", est.smemBytes);
            group.params.put("hidden", hidden);
            chooseMlpEpilogue(group, consumer);
            FusionDecision decision = new FusionDecision(group.id, GroupKind.GEMM_GEMM_FUSED.label, true);
            for (IrNode n : group.members) {
                decision.members.add(n.name);
            }
            decision.predicted.put("smemBytes", est.smemBytes);
            decision.predicted.put("regsPerThread", est.regsPerThread);
            decision.predicted.put("occupancyMilli", (int) Math.round(est.occupancy * 1000));
            decision.predicted.put("gridBlocks", est.gridBlocks);
            decision.predicted.put("tileRM", cfg[0]);
            decision.predicted.put("tileCN", cfg[1]);
            decision.predicted.put("hiddenChunk", cfg[2]);
            decision.metrics.put("bytesSaved", (double) saved);
            decision.metrics.put("launchesSaved", 1.0);
            decision.bytesSaved = saved;
            decision.gridBlocks = est.gridBlocks;
            decision.reason = String.format(java.util.Locale.ROOT,
                    "GEMM->%s->GEMM (%dx%d x %dx%d, hidden %d) fused: the [%dx%d] hidden tile stays in "
                    + "shared memory instead of round-tripping through DRAM (%.1f KB saved) and the second "
                    + "kernel launch disappears; tile %dx%d hidden-chunk %d, %s; %d blocks >= %d floor",
                    chain.stream().anyMatch(n -> isActivation(n.op)) ? "activation" : "bias",
                    m, d, hidden, f, hidden, m, hidden, saved / 1024.0, cfg[0], cfg[1], cfg[2],
                    ResourceModel.describe(est), est.gridBlocks, minGrid);
            group.decision = decision;
            register(group);
            decisions.add(decision);
            return true;
        }

        FusionDecision decision = new FusionDecision(-1, GroupKind.GEMM_GEMM_FUSED.label, false);
        decision.members.add(producer.name);
        for (IrNode n : chain) {
            decision.members.add(n.name);
        }
        decision.members.add(consumer.name);
        decision.bytesSaved = saved;
        if (best != null) {
            decision.predicted.put("smemBytes", best.smemBytes);
            decision.predicted.put("regsPerThread", best.regsPerThread);
            decision.predicted.put("occupancyMilli", (int) Math.round(best.occupancy * 1000));
            decision.gridBlocks = best.gridBlocks;
        }
        decision.metrics.put("bytesSaved", (double) saved);
        decision.reason = "rejected: " + rejectWhy + "; keeping " + (chain.size() + 2)
                + " unfused kernels (the " + String.format(java.util.Locale.ROOT, "%.1f KB",
                saved / 1024.0) + " of DRAM traffic saved does not compensate for the lost parallelism)";
        decisions.add(decision);
        return false;
    }

    private static final int MIN_ACCUM_REGS_PER_THREAD = 4;

    private void chooseMlpEpilogue(FusionGroup group, IrNode consumer) {
        if (consumer.consumers.size() == 1) {
            IrNode next = consumer.consumers.get(0);
            if (!owner.containsKey(next) && (isActivation(next.op) || next.op == OpKind.BIAS_ADD)
                    && leafOnlyExtras(next, consumer)) {
                group.members.add(next);
            }
        }
    }

    // ------------------------------------------------------------------ pattern 3

    private boolean fuseGemmEpilogue() {
        boolean changed = false;
        // (a) fresh matmuls, (b) existing groups that can still absorb a trailing epilogue
        List<IrNode> heads = new ArrayList<>();
        for (IrNode n : graph.nodes) {
            if (n.op == OpKind.MATMUL && !owner.containsKey(n)) {
                heads.add(n);
            } else if (owner.containsKey(n)) {
                FusionGroup group = owner.get(n);
                if ((group.kind == GroupKind.GEMM_EPILOGUE || group.kind == GroupKind.GEMM_GEMM_FUSED)
                        && group.members.get(group.members.size() - 1) == n) {
                    heads.add(n);
                }
            }
        }
        for (IrNode head : heads) {
            FusionGroup existing = owner.get(head);
            IrNode cursor = head;
            List<IrNode> chain = new ArrayList<>();
            while (cursor.consumers.size() == 1 && chain.size() < PascalHeuristics.MAX_EPILOGUE_CHAIN) {
                IrNode next = cursor.consumers.get(0);
                if (owner.containsKey(next) || !isEpilogueOp(next.op)) {
                    break;
                }
                if (next.inputs.get(0) != cursor || !leafOnlyExtras(next, cursor)) {
                    break;
                }
                chain.add(next);
                cursor = next;
            }
            if (chain.isEmpty()) {
                continue;
            }
            String key = "epi:" + head.name + ":" + chain.size();
            if (!attempted.add(key)) {
                continue;
            }
            long saved = 2 * head.bytes();
            for (int i = 0; i + 1 < chain.size(); i++) {
                saved += 2 * chain.get(i).bytes();
            }
            int[] tile = ResourceModel.chooseGemmTile(head.shape.rows, head.shape.cols);
            ResourceModel.Est est = ResourceModel.gemm(head.shape.rows, head.shape.cols, chain.size(), tile);
            if (!ResourceModel.fits(est)) {
                reject("gemm_epilogue", head, chain, saved, est,
                        "predicted resources do not fit the sm_61 caps");
                continue;
            }
            if (saved < PascalHeuristics.MIN_GLOBAL_BYTES_SAVED && !force) {
                reject("gemm_epilogue", head, chain, saved, est,
                        "only " + saved + " bytes of DRAM traffic removed, below the "
                        + PascalHeuristics.MIN_GLOBAL_BYTES_SAVED + "-byte floor");
                continue;
            }
            if (!PascalHeuristics.FUSE_ELEMENTWISE_EPILOGUE && !force) {
                reject("gemm_epilogue", head, chain, saved, est, "epilogue fusion disabled by policy switch");
                continue;
            }
            if (existing == null) {
                FusionGroup group = new FusionGroup(nextGroupId++, GroupKind.GEMM_EPILOGUE);
                group.members.add(head);
                group.members.addAll(chain);
                group.params.put("bm", tile[0]);
                group.params.put("bn", tile[1]);
                group.params.put("bk", PascalHeuristics.GEMM_BK);
                group.params.put("threads", ResourceModel.gemmThreads(tile));
                group.params.put("smemBytes", est.smemBytes);
                group.decision = acceptDecision(group, chain, saved, est, String.format(java.util.Locale.ROOT,
                        "%s consumes the %s accumulator tile straight out of registers: the [%dx%d] tile never "
                        + "reaches DRAM (%.1f KB saved) and %d launch(es) disappear; %s vs unfused %s",
                        describeChain(chain), head.op.name().toLowerCase(), head.shape.rows, head.shape.cols,
                        saved / 1024.0, chain.size(), ResourceModel.describe(est),
                        ResourceModel.describe(ResourceModel.gemm(head.shape.rows, head.shape.cols, 0))));
                register(group);
            } else {
                for (IrNode n : chain) {
                    existing.members.add(n);
                }
                existing.decision = acceptDecision(existing, chain, saved, est, String.format(java.util.Locale.ROOT,
                        "extended existing %s group with %s: a further %.1f KB of DRAM traffic and %d launch(es) removed",
                        existing.kind.label, describeChain(chain), saved / 1024.0, chain.size()));
            }
            decisions.add(existing == null ? owner.get(cursor).decision : existing.decision);
            changed = true;
            if (existing != null) {
                break;
            }
        }
        return changed;
    }

    private void reject(String kind, IrNode head, List<IrNode> chain, long saved, ResourceModel.Est est,
                        String why) {
        FusionDecision decision = new FusionDecision(-1, kind, false);
        decision.members.add(head.name);
        for (IrNode n : chain) {
            decision.members.add(n.name);
        }
        decision.bytesSaved = saved;
        decision.gridBlocks = est.gridBlocks;
        decision.predicted.put("smemBytes", est.smemBytes);
        decision.predicted.put("regsPerThread", est.regsPerThread);
        decision.predicted.put("occupancyMilli", (int) Math.round(est.occupancy * 1000));
        decision.reason = "rejected: " + why;
        decisions.add(decision);
    }

    private FusionDecision acceptDecision(FusionGroup group, List<IrNode> chain, long saved,
                                          ResourceModel.Est est, String reason) {
        FusionDecision decision = new FusionDecision(group.id, group.kind.label, true);
        for (IrNode n : group.members) {
            decision.members.add(n.name);
        }
        decision.predicted.put("smemBytes", est.smemBytes);
        decision.predicted.put("regsPerThread", est.regsPerThread);
        decision.predicted.put("occupancyMilli", (int) Math.round(est.occupancy * 1000));
        decision.predicted.put("gridBlocks", est.gridBlocks);
        decision.metrics.put("bytesSaved", (double) saved);
        decision.metrics.put("launchesSaved", (double) chain.size());
        decision.bytesSaved = saved;
        decision.gridBlocks = est.gridBlocks;
        decision.reason = reason;
        return decision;
    }

    // ------------------------------------------------------------------ pattern 4

    private boolean fuseElementwiseChains() {
        boolean changed = false;
        for (IrNode head : new ArrayList<>(graph.nodes)) {
            if (owner.containsKey(head) || head.isLeaf() || !isEpilogueOp(head.op) || head.inputs.isEmpty()) {
                continue;
            }
            if (!isFusableSeed(head)) {
                continue;
            }
            List<IrNode> members = new ArrayList<>();
            IrNode cursor = head;
            members.add(cursor);
            while (cursor.consumers.size() == 1) {
                IrNode next = cursor.consumers.get(0);
                if (owner.containsKey(next) || !isEpilogueOp(next.op) || next.inputs.get(0) != cursor
                        || !leafOnlyExtras(next, cursor)
                        || members.size() >= PascalHeuristics.MAX_EPILOGUE_CHAIN) {
                    break;
                }
                members.add(next);
                cursor = next;
            }
            if (members.size() < 2) {
                continue;
            }
            long saved = 0;
            for (int i = 0; i + 1 < members.size(); i++) {
                saved += 2 * members.get(i).bytes();
            }
            ResourceModel.Est est = ResourceModel.elementwise(members.size());
            if (saved < PascalHeuristics.MIN_GLOBAL_BYTES_SAVED && !force) {
                reject("elementwise_fused", head, members.subList(1, members.size()), saved, est,
                        "only " + saved + " bytes removed between " + members.size()
                        + " elementwise ops, below the " + PascalHeuristics.MIN_GLOBAL_BYTES_SAVED + "-byte floor");
                continue;
            }
            FusionGroup group = new FusionGroup(nextGroupId++, GroupKind.ELEMENTWISE_FUSED);
            group.members.addAll(members);
            group.params.put("threads", PascalHeuristics.ELEMENTWISE_THREADS);
            group.params.put("smemBytes", 0);
            group.decision = acceptDecision(group, members.subList(1, members.size()), saved, est,
                    "chained " + members.size() + " elementwise ops (" + describeChain(members) + "): "
                    + String.format(java.util.Locale.ROOT, "%.1f KB", saved / 1024.0)
                    + " of intermediate DRAM traffic removed");
            register(group);
            decisions.add(group.decision);
            changed = true;
        }
        return changed;
    }

    /** An elementwise node may start a fused chain only if its own producers are leaves/other ops. */
    private boolean isFusableSeed(IrNode node) {
        if (node.op == OpKind.SCALE) {
            return node.inputs.size() >= 1;
        }
        for (IrNode in : node.inputs) {
            if (in.isLeaf()) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ helpers

    private void finaliseSingletons() {
        for (IrNode node : graph.nodes) {
            if (owner.containsKey(node) || node.isLeaf()) {
                continue;
            }
            FusionGroup group = new FusionGroup(nextGroupId++, GroupKind.SINGLE);
            group.members.add(node);
            ResourceModel.Est est = resourcesForSingle(node);
            group.params.put("threads", est.threads);
            group.params.put("smemBytes", est.smemBytes);
            if (node.op == OpKind.MATMUL) {
                int[] tile = ResourceModel.chooseGemmTile(node.shape.rows, node.shape.cols);
                group.params.put("bm", tile[0]);
                group.params.put("bn", tile[1]);
                group.params.put("bk", PascalHeuristics.GEMM_BK);
            }
            group.decision = new FusionDecision(group.id, "single", true);
            group.decision.members.add(node.name);
            group.decision.predicted.put("smemBytes", est.smemBytes);
            group.decision.predicted.put("regsPerThread", est.regsPerThread);
            group.decision.predicted.put("occupancyMilli", (int) Math.round(est.occupancy * 1000));
            group.decision.predicted.put("gridBlocks", est.gridBlocks);
            group.decision.reason = "standalone op: no profitable fusion partner found for " + node.name;
            register(group);
        }
    }

    private ResourceModel.Est resourcesForSingle(IrNode node) {
        switch (node.op) {
            case MATMUL:
                return ResourceModel.gemm(node.shape.rows, node.shape.cols, node.scale() == 1.0 ? 0 : 1,
                        ResourceModel.chooseGemmTile(node.shape.rows, node.shape.cols));
            case SOFTMAX:
            case LAYERNORM:
                return ResourceModel.rowReduce(node.shape.rows, node.shape.cols);
            default:
                return ResourceModel.elementwise(1);
        }
    }

    private void register(FusionGroup group) {
        groups.add(group);
        for (IrNode n : group.members) {
            owner.put(n, group);
        }
    }

    private void applyPlacement(FusionGroup group) {
        switch (group.kind) {
            case ATTENTION_FUSED:
                // scores live in registers, probabilities are exchanged through shared memory
                group.members.get(0).placement = MemSpace.REGISTER;
                group.members.get(1).placement = MemSpace.SHARED;
                group.members.get(2).placement = MemSpace.REGISTER;
                break;
            case GEMM_GEMM_FUSED:
                group.members.get(0).placement = MemSpace.SHARED;   // hidden tile stays in smem
                for (int i = 1; i < group.members.size() - 1; i++) {
                    group.members.get(i).placement = MemSpace.REGISTER;
                }
                group.members.get(group.members.size() - 1).placement = MemSpace.REGISTER;
                break;
            default:
                for (IrNode n : group.internal) {
                    n.placement = MemSpace.REGISTER;
                }
                break;
        }
    }

    /** Softmax/LayerNorm logits keep FP32 storage even under an FP16 graph policy. */
    private void applyNumericPolicy() {
        if (!PascalHeuristics.PROMOTE_REDUCTION_INPUTS_TO_FP32) {
            return;
        }
        for (IrNode node : graph.nodes) {
            if (node.op != OpKind.SOFTMAX && node.op != OpKind.LAYERNORM) {
                continue;
            }
            for (IrNode in : node.inputs) {
                if (!in.isLeaf() && in.store.sizeBytes < 4) {
                    in.store = io.fusionc.ir.DType.FP32;
                    in.attrs.put("promoted", "fp32 (reduction input: 10-bit mantissa would cost ~1e-2 "
                            + "of probability mass after exp)");
                }
            }
        }
    }

    private boolean leafOnlyExtras(IrNode node, IrNode mainInput) {
        for (IrNode in : node.inputs) {
            if (in != mainInput && !in.isLeaf()) {
                return false;
            }
        }
        return true;
    }

    private static boolean isEpilogueOp(OpKind op) {
        return op == OpKind.BIAS_ADD || op == OpKind.RELU || op == OpKind.GELU || op == OpKind.SCALE
                || op == OpKind.ADD;
    }

    private static boolean isActivation(OpKind op) {
        return op == OpKind.RELU || op == OpKind.GELU;
    }

    private static String describeChain(List<IrNode> chain) {
        Set<String> parts = new LinkedHashSet<>();
        for (IrNode n : chain) {
            parts.add(n.op.name().toLowerCase());
        }
        return String.join("+", parts);
    }

    private static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }
}
