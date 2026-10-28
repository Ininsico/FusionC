import java.io.FileReader;

public class Main {
    public static void main(String[] args) {
        try {
            Lexer CyberLexer = new Lexer(new FileReader(args[0]));
            parser p = new parser(CyberLexer);

            Object result = p.parse();
            System.out.println(result);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}