package io.fusionc.pass;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Structured record of one fusion decision, emitted as JSON for the evaluation driver. */
public final class FusionDecision {

    public final int groupId;
    public final String kind;
    public final boolean accepted;
    public final List<String> members = new ArrayList<>();
    public final Map<String, Integer> predicted = new LinkedHashMap<>();
    public final Map<String, Double> metrics = new LinkedHashMap<>();
    public String reason = "";
    public long bytesSaved;
    public int gridBlocks;

    public FusionDecision(int groupId, String kind, boolean accepted) {
        this.groupId = groupId;
        this.kind = kind;
        this.accepted = accepted;
    }

    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("  {\n");
        sb.append("    \"group\": ").append(groupId).append(",\n");
        sb.append("    \"kind\": \"").append(kind).append("\",\n");
        sb.append("    \"accepted\": ").append(accepted).append(",\n");
        sb.append("    \"members\": [");
        for (int i = 0; i < members.size(); i++) {
            sb.append(i > 0 ? ", " : "").append('"').append(members.get(i)).append('"');
        }
        sb.append("],\n");
        sb.append("    \"predicted\": {");
        boolean first = true;
        for (Map.Entry<String, Integer> e : predicted.entrySet()) {
            sb.append(first ? " " : ", ").append('"').append(e.getKey()).append("\": ").append(e.getValue());
            first = false;
        }
        sb.append(first ? "}" : " }").append(",\n");
        sb.append("    \"metrics\": {");
        first = true;
        for (Map.Entry<String, Double> e : metrics.entrySet()) {
            sb.append(first ? " " : ", ").append('"').append(e.getKey()).append("\": ")
              .append(String.format(java.util.Locale.ROOT, "%.4f", e.getValue()));
            first = false;
        }
        sb.append(first ? "}" : " }").append(",\n");
        sb.append("    \"gridBlocks\": ").append(gridBlocks).append(",\n");
        sb.append("    \"bytesSaved\": ").append(bytesSaved).append(",\n");
        sb.append("    \"reason\": \"").append(reason.replace("\\", "\\\\").replace("\"", "\\\"")).append("\"\n");
        sb.append("  }");
        return sb.toString();
    }
}
