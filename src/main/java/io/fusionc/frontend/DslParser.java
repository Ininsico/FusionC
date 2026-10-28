package io.fusionc.frontend;

import io.fusionc.FrontendException;
import io.fusionc.ast.Ast;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;

/**
 * Front-end driver: JFlex {@link Scanner} (generated from {@code Scanner.flex}) feeding the
 * CUP-generated {@code parser} (generated from {@code Parser.cup}).
 */
public final class DslParser {

    private DslParser() {
    }

    public static Ast.Program parseFile(File file) {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)) {
            return parse(reader);
        } catch (FrontendException e) {
            throw new FrontendException(file.getName() + ": " + e.getMessage(), e);
        } catch (Exception e) {
            throw new FrontendException(file.getName() + ": parse failure: " + e.getMessage(), e);
        }
    }

    public static Ast.Program parseString(String source) {
        return parse(new InputStreamReader(new ByteArrayInputStream(source.getBytes(StandardCharsets.UTF_8)),
                StandardCharsets.UTF_8));
    }

    private static Ast.Program parse(Reader reader) {
        try {
            Scanner scanner = new Scanner(reader);
            parser cupParser = new parser(scanner);
            Object result = cupParser.parse().value;
            return (Ast.Program) result;
        } catch (FrontendException e) {
            throw e;
        } catch (Exception e) {
            throw new FrontendException("parse failure: " + e.getMessage(), e);
        }
    }
}
