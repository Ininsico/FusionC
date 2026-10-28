import java_cup.runtime.*;

%%

%class Lexer
%cup
%unicode
%line
%column

/* Valid Identifiers */
smallAlphabets=[a-z]
capitalAlphabets=[A-Z]
digits=[0-9]
specialChar=[_]

/* Declaration Keywords */
kNETWORK = "NETWORK"
kDEVICE = "DEVICE"

/* Command Keywords */
kSET = "SET"
kSCAN = "SCAN"
kALERT = "ALERT"
kLOG = "LOG"
kBLOCK = "BLOCK"
kREPORT = "REPORT"

/* Control Flow Keywords */
kIF = "IF"
kEND = "END"

/* Condition Keywords */
kPORT = "PORT"
kOPEN = "OPEN"
kCLOSED = "CLOSED"
kTRAFFIC = "TRAFFIC"
kATTACK = "ATTACK"
kDETECTED = "DETECTED"

/* Target Keywords */
kSOURCE = "SOURCE"
kADMIN = "ADMIN"

/* Attack Type Keywords */
kDDOS = "DDOS"
kBRUTEFORCE = "BRUTEFORCE"
kPORTSCAN = "PORTSCAN"

/* Numbers */
integerNumbers = [0-9]+
floatNumbers = [0-9]+\.[0-9]+
scientificNumbers = ([0-9]+\.[0-9]+|[0-9]+)[eE][+-]?[0-9]+

/* Strings */
stringLiteral = \"([^\"\\\n]|\\.)*\"

/* Comments */
singleLineComment = "//"[^\n]*

/* Relational Operators */
opLTE = "<="
opGTE = ">="
opEQ = "=="
opNE = "!="
opLT = "<"
opGT = ">"

/* Assignment Operator */
opASS = "="

/* Arithmetic Operators */
opPLUS = "+"
opSUB = "-"
opMUL = "*"
opDIV = "/"

/* Parentheses */
leftParen = "("
rightParen = ")"

%%

/* Ignore Whitespace */
[ \t\r\n]+ {}

/* Declaration Keywords */
{kNETWORK}   { return new Symbol(sym.NETWORK); }
{kDEVICE}    { return new Symbol(sym.DEVICE); }

/* Command Keywords */
{kSET}       { return new Symbol(sym.SET); }
{kSCAN}      { return new Symbol(sym.SCAN); }
{kALERT}     { return new Symbol(sym.ALERT); }
{kLOG}       { return new Symbol(sym.LOG); }
{kBLOCK}     { return new Symbol(sym.BLOCK); }
{kREPORT}    { return new Symbol(sym.REPORT); }

/* Control Flow Keywords */
{kIF}        { return new Symbol(sym.IF); }
{kEND}       { return new Symbol(sym.END); }

/* Condition Keywords */
{kPORT}      { return new Symbol(sym.PORT); }
{kOPEN}      { return new Symbol(sym.OPEN); }
{kCLOSED}    { return new Symbol(sym.CLOSED); }
{kTRAFFIC}   { return new Symbol(sym.TRAFFIC); }
{kATTACK}    { return new Symbol(sym.ATTACK); }
{kDETECTED}  { return new Symbol(sym.DETECTED); }

/* Target Keywords */
{kSOURCE}    { return new Symbol(sym.SOURCE); }
{kADMIN}     { return new Symbol(sym.ADMIN); }

/* Attack Type Keywords */
{kDDOS}        { return new Symbol(sym.DDOS); }
{kBRUTEFORCE}  { return new Symbol(sym.BRUTEFORCE); }
{kPORTSCAN}    { return new Symbol(sym.PORTSCAN); }

/* Numbers (integer, float and scientific all reach the parser as NUMBER) */
{scientificNumbers} { return new Symbol(sym.NUMBER, yytext()); }
{floatNumbers}      { return new Symbol(sym.NUMBER, yytext()); }
{integerNumbers}    { return new Symbol(sym.NUMBER, yytext()); }

/* Strings */
{stringLiteral} { return new Symbol(sym.STRING, yytext()); }

/* Comments */
{singleLineComment} {}


/* Relational Operators */
{opLTE} { return new Symbol(sym.LE); }
{opGTE} { return new Symbol(sym.GE); }
{opEQ}  { return new Symbol(sym.EQ); }
{opNE}  { return new Symbol(sym.NE); }
{opLT}  { return new Symbol(sym.LT); }
{opGT}  { return new Symbol(sym.GT); }

/* Assignment Operator */
{opASS} { return new Symbol(sym.ASSIGN); }

/* Arithmetic Operators */
{opPLUS} { return new Symbol(sym.PLUS); }
{opSUB}  { return new Symbol(sym.MINUS); }
{opMUL}  { return new Symbol(sym.MULT); }
{opDIV}  { return new Symbol(sym.DIV); }

/* Parentheses */
{leftParen}  { return new Symbol(sym.LPAREN); }
{rightParen} { return new Symbol(sym.RPAREN); }

/* Valid Identifier */
({smallAlphabets}|{capitalAlphabets}|{specialChar})({smallAlphabets}|{capitalAlphabets}|{digits}|{specialChar})* {
    return new Symbol(sym.ID, yytext());
}

/* Invalid Character */
. {
    System.out.printf("Lexical Error: Invalid character '%s'. Line %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}
