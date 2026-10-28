package io.fusionc;

import io.fusionc.frontend.Deriv;
import io.fusionc.frontend.DslParser;

import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextArea;
import javax.swing.JToolBar;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Swing viewer for the <b>derivation tree</b>: the tree the CUP parser builds from the start
 * symbol down to every terminal. The source on the right is editable — press
 * "Parse (Ctrl+Enter)" and the tree on the left is rebuilt live; a syntax error pops up a
 * dialog with the exact position and the offending line is selected.
 *
 * <pre>
 *   .\scripts\parse-tree-gui.ps1
 *   java -cp build/fusionc.jar io.fusionc.ParseTreeGui examples/attention_mlp.dsl
 * </pre>
 *
 * {@code --selftest} parses, builds the tree and runs the layout maths without opening a window.
 */
public final class ParseTreeGui {

    private ParseTreeGui() {
    }

    public static void main(String[] args) {
        if (args.length > 0 && "--selftest".equals(args[0])) {
            selfTest(new File(args.length > 1 ? args[1] : "examples/attention_mlp.dsl"));
            return;
        }
        String initial = args.length > 0
                ? readFile(new File(args[0]))
                : readFile(new File("examples/attention_mlp.dsl"));
        SwingUtilities.invokeLater(() -> new TreeWindow(initial).setVisible(true));
    }

    private static String readFile(File file) {
        try {
            return Files.readString(file.toPath(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "";
        }
    }

    private static void selfTest(File file) {
        Deriv tree = DslParser.parseTree(file);
        int layoutChecksum = 0;
        TreePanel panel = new TreePanel();
        panel.setRoot(TreeNode.from(tree));
        System.out.print(tree.dump());
        System.out.printf("%nselftest ok: nodes=%d depth=%d canvas=%dx%d%n",
                tree.count(), tree.depth(),
                panel.getPreferredSize().width, panel.getPreferredSize().height);
    }

    // ------------------------------------------------------------------ tree data

    static final class TreeNode {
        final String label;
        final boolean terminal;
        final List<TreeNode> children = new ArrayList<>();

        TreeNode(String label, boolean terminal) {
            this.label = label;
            this.terminal = terminal;
        }

        static TreeNode from(Deriv node) {
            TreeNode tree = new TreeNode(node.toString(), node.isTerminal());
            for (Deriv child : node.children) {
                tree.children.add(from(child));
            }
            return tree;
        }
    }

    // ------------------------------------------------------------------ canvas

    static final class TreePanel extends JPanel {
        private static final int NODE_HEIGHT = 28;
        private static final int LEVEL_GAP = 58;
        private static final int SIBLING_GAP = 16;
        private static final int PAD_X = 12;
        private static final int MARGIN = 26;
        private static final Font NODE_FONT = new Font(Font.MONOSPACED, Font.PLAIN, 11);

        private TreeNode root;
        private final Map<TreeNode, Integer> subtreeWidth = new HashMap<>();
        private final Map<TreeNode, Point> position = new HashMap<>();

        TreePanel() {
            setBackground(Color.WHITE);
        }

        void setRoot(TreeNode root) {
            this.root = root;
            subtreeWidth.clear();
            position.clear();
            if (root != null) {
                int width = computeWidths(root);
                assign(root, MARGIN, 0, width);
                int depth = maxDepth(root);
                setPreferredSize(new Dimension(width + 2 * MARGIN, MARGIN * 2 + (depth + 1) * LEVEL_GAP));
            } else {
                setPreferredSize(new Dimension(420, 220));
            }
            revalidate();
            repaint();
        }

        private int nodeWidth(TreeNode node) {
            return Math.max(70, getFontMetrics(NODE_FONT).stringWidth(node.label) + 2 * PAD_X);
        }

        private int computeWidths(TreeNode node) {
            int width = nodeWidth(node);
            if (!node.children.isEmpty()) {
                int childrenWidth = SIBLING_GAP * (node.children.size() - 1);
                for (TreeNode child : node.children) {
                    childrenWidth += computeWidths(child);
                }
                width = Math.max(width, childrenWidth);
            }
            subtreeWidth.put(node, width);
            return width;
        }

        private void assign(TreeNode node, int left, int depth, int width) {
            position.put(node, new Point(left + width / 2, MARGIN + depth * LEVEL_GAP));
            if (node.children.isEmpty()) {
                return;
            }
            int childrenWidth = SIBLING_GAP * (node.children.size() - 1);
            for (TreeNode child : node.children) {
                childrenWidth += subtreeWidth.get(child);
            }
            int x = left + (width - childrenWidth) / 2;
            for (TreeNode child : node.children) {
                int childWidth = subtreeWidth.get(child);
                assign(child, x, depth + 1, childWidth);
                x += childWidth + SIBLING_GAP;
            }
        }

        private int maxDepth(TreeNode node) {
            int max = 0;
            for (TreeNode child : node.children) {
                max = Math.max(max, 1 + maxDepth(child));
            }
            return max;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (root == null) {
                return;
            }
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setFont(NODE_FONT);
            g2.setStroke(new BasicStroke(1.4f));
            paintEdges(g2, root);
            paintNodes(g2, root);
            g2.dispose();
        }

        private void paintEdges(Graphics2D g2, TreeNode node) {
            Point parent = position.get(node);
            for (TreeNode child : node.children) {
                Point point = position.get(child);
                g2.setColor(new Color(0xC3, 0xCB, 0xDB));
                g2.drawLine(parent.x, parent.y + NODE_HEIGHT / 2, point.x, point.y - NODE_HEIGHT / 2);
                paintEdges(g2, child);
            }
        }

        private void paintNodes(Graphics2D g2, TreeNode node) {
            Point point = position.get(node);
            int width = nodeWidth(node);
            int x = point.x - width / 2;
            int y = point.y - NODE_HEIGHT / 2;

            g2.setColor(node.terminal ? new Color(0x6E, 0x8B, 0x6A) : kindColor(node.label));
            g2.fillRoundRect(x, y, width, NODE_HEIGHT, 10, 10);
            g2.setColor(new Color(0x2F, 0x38, 0x47));
            g2.drawRoundRect(x, y, width, NODE_HEIGHT, 10, 10);

            g2.setColor(Color.WHITE);
            FontMetrics fm = g2.getFontMetrics();
            g2.drawString(node.label,
                    point.x - fm.stringWidth(node.label) / 2,
                    point.y + (fm.getAscent() - fm.getDescent()) / 2);

            for (TreeNode child : node.children) {
                paintNodes(g2, child);
            }
        }

        private Color kindColor(String label) {
            switch (label) {
                case "program":
                    return new Color(0x3E, 0x5C, 0x8E);
                case "decl":
                case "decl_list":
                    return new Color(0x55, 0x64, 0x7E);
                case "stmt":
                case "stmt_list":
                case "stage_list":
                case "stage":
                    return new Color(0x4C, 0x64, 0x94);
                case "dims":
                case "expr":
                case "expr_list":
                case "arglist":
                case "datatype":
                    return new Color(0x7A, 0x6A, 0x92);
                default:
                    return new Color(0x5B, 0x6B, 0x84);
            }
        }
    }

    // ------------------------------------------------------------------ window

    static final class TreeWindow extends JFrame {
        private final TreePanel canvas = new TreePanel();
        private final JTextArea editor = new JTextArea();
        private final JLabel status = new JLabel(" edit the program on the right, then press Parse");
        private final JLabel fileLabel = new JLabel(" ");

        TreeWindow(String initialSource) {
            super("FusionC — parse tree (JFlex + CUP)");
            setDefaultCloseOperation(EXIT_ON_CLOSE);
            setSize(1280, 800);
            setLocationRelativeTo(null);
            setLayout(new BorderLayout());

            JToolBar bar = new JToolBar();
            bar.setFloatable(false);
            JButton open = new JButton("Open .dsl ...");
            open.addActionListener(e -> chooseFile());
            JButton parse = new JButton("Parse  (Ctrl+Enter)");
            parse.addActionListener(e -> parseEditor());
            bar.add(open);
            bar.add(parse);
            bar.addSeparator();
            bar.add(fileLabel);
            add(bar, BorderLayout.NORTH);

            editor.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            editor.setBackground(new Color(0xFA, 0xFB, 0xFD));
            editor.setText(initialSource);
            editor.getInputMap().put(KeyStroke.getKeyStroke("control ENTER"), "parse");
            editor.getActionMap().put("parse", new AbstractAction() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    parseEditor();
                }
            });

            JScrollPane treeScroll = new JScrollPane(canvas);
            treeScroll.setBorder(BorderFactory.createTitledBorder(
                    "derivation tree — from the start symbol down to every terminal"));
            JScrollPane editorScroll = new JScrollPane(editor);
            editorScroll.setBorder(BorderFactory.createTitledBorder("program (editable)"));

            JSplitPane split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScroll, editorScroll);
            split.setResizeWeight(0.65);
            split.setBorder(null);
            add(split, BorderLayout.CENTER);
            add(status, BorderLayout.SOUTH);

            SwingUtilities.invokeLater(this::parseEditor);
        }

        private void chooseFile() {
            JFileChooser chooser = new JFileChooser(new File("examples"));
            chooser.setFileFilter(new FileNameExtensionFilter("FusionC DSL (*.dsl)", "dsl"));
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                File file = chooser.getSelectedFile();
                fileLabel.setText(file.getPath());
                editor.setText(readFile(file));
                parseEditor();
            }
        }

        private void parseEditor() {
            String source = editor.getText();
            try {
                Deriv tree = DslParser.parseTree(source);
                canvas.setRoot(TreeNode.from(tree));
                status.setText(" parsed OK — " + tree.count() + " nodes, depth " + tree.depth()
                        + "  ·  green = terminal (a word of the program), blue = grammar symbol");
            } catch (FrontendException e) {
                canvas.setRoot(null);
                status.setText(" syntax error — " + e.getMessage());
                highlight(e.getMessage());
                JOptionPane.showMessageDialog(this, e.getMessage(), "Syntax error",
                        JOptionPane.ERROR_MESSAGE);
            } catch (Exception e) {
                canvas.setRoot(null);
                status.setText(" error — " + e.getMessage());
                JOptionPane.showMessageDialog(this, e.toString(), "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
        }

        /** Select the line the parser stopped on, so the error is visible in the editor. */
        private void highlight(String message) {
            Matcher matcher = Pattern.compile("line (\\d+), column (\\d+)").matcher(message);
            if (!matcher.find()) {
                return;
            }
            int line = Integer.parseInt(matcher.group(1)) - 1;
            int column = Integer.parseInt(matcher.group(2)) - 1;
            try {
                int start = editor.getLineStartOffset(line);
                int end = editor.getLineEndOffset(line);
                editor.requestFocusInWindow();
                editor.setCaretPosition(Math.min(start + column, Math.max(start, end - 1)));
                editor.select(start, Math.max(start, end - 1));
            } catch (Exception ignored) {
                // line out of range -- leave the caret alone
            }
        }
    }
}
