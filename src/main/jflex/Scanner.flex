/* Scanner.flex -- JFlex lexer for the FusionC tensor-pipeline DSL.
 *
 * Run:  java -jar tools/jflex.jar -d build/generated src/main/jflex/Scanner.flex
 *
 * %cup compatibility mode makes the generated Scanner implement
 * java_cup.runtime.Scanner: next_token() returns java_cup.runtime.Symbol
 * objects whose {sym, left=line, right=column, value} fields are consumed by
 * the CUP-generated parser (build/generated/io/fusionc/frontend/parser.java).
 */
package io.fusionc.frontend;

import java_cup.runtime.Symbol;

%%

%class Scanner
%public
%unicode
%line
%column
%cup

%{
  /** Build a valueless token carrying 1-based line/column for diagnostics. */
  private Symbol token(int type) {
    return new Symbol(type, yyline + 1, yycolumn + 1);
  }

  /** Build a token carrying its semantic value (ident text / integer / float). */
  private Symbol token(int type, Object value) {
    return new Symbol(type, yyline + 1, yycolumn + 1, value);
  }

  private void fail(String message) {
    throw new io.fusionc.FrontendException(
        "lexical error at line " + (yyline + 1) + ", column " + (yycolumn + 1) + ": " + message);
  }
%}

LineTerminator = \r|\n|\r\n
InputCharacter = [^\r\n]
WhiteSpace     = {LineTerminator} | [ \t\f]
LineComment    = "//" {InputCharacter}*
HashComment    = "#" {InputCharacter}*
BlockComment   = "/*" ~"*/" "*/"
Identifier     = [A-Za-z_][A-Za-z0-9_]*
IntegerLiteral = [0-9]+
FloatLiteral   = {IntegerLiteral} "." [0-9]+ ([eE] [+-]? [0-9]+)? | {IntegerLiteral} [eE] [+-]? [0-9]+

%%

<YYINITIAL> {
  {WhiteSpace}    { /* skip */ }
  {LineComment}   { /* skip */ }
  {HashComment}   { /* skip */ }
  {BlockComment}  { /* skip */ }

  "const"         { return token(sym.CONST); }
  "dtype"         { return token(sym.DTYPE); }
  "store"         { return token(sym.STORE); }
  "accum"         { return token(sym.ACCUM); }
  "input"         { return token(sym.INPUT); }
  "param"         { return token(sym.PARAM); }
  "fp16"          { return token(sym.FP16); }
  "fp32"          { return token(sym.FP32); }

  "->"            { return token(sym.ARROW); }
  "="             { return token(sym.EQ); }
  ";"             { return token(sym.SEMI); }
  ","             { return token(sym.COMMA); }
  ":"             { return token(sym.COLON); }
  "("             { return token(sym.LPAREN); }
  ")"             { return token(sym.RPAREN); }
  "["             { return token(sym.LBRACKET); }
  "]"             { return token(sym.RBRACKET); }

  {FloatLiteral}  { return token(sym.FLOAT, Double.valueOf(yytext())); }
  {IntegerLiteral} { return token(sym.INT, Long.valueOf(yytext())); }
  {Identifier}    { return token(sym.IDENT, yytext()); }

  [^]             { fail("illegal character '" + yytext() + "'"); }
}
