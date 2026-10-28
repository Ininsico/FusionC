%%

%class Lexer
%standalone
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
unterminatedString = \"([^\"\\\n]|\\.)*

/* Comments */
singleLineComment = "//"[^\n]*
multiLineComment = "/*" ~"*/"
unterminatedComment = "/*"

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
{kNETWORK} {
    System.out.printf("Valid Network: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kDEVICE} {
    System.out.printf("Valid Device: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Command Keywords */
{kSET} {
    System.out.printf("Valid Set: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kSCAN} {
    System.out.printf("Valid Scan: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kALERT} {
    System.out.printf("Valid Alert: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kLOG} {
    System.out.printf("Valid Log: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kBLOCK} {
    System.out.printf("Valid Block: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kREPORT} {
    System.out.printf("Valid Report: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Control Flow Keywords */
{kIF} {
    System.out.printf("Valid If: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kEND} {
    System.out.printf("Valid End: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Condition Keywords */
{kPORT} {
    System.out.printf("Valid Port: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kOPEN} {
    System.out.printf("Valid Open: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kCLOSED} {
    System.out.printf("Valid Closed: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kTRAFFIC} {
    System.out.printf("Valid Traffic: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kATTACK} {
    System.out.printf("Valid Attack: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kDETECTED} {
    System.out.printf("Valid Detected: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Target Keywords */
{kSOURCE} {
    System.out.printf("Valid Source: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kADMIN} {
    System.out.printf("Valid Admin: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Attack Type Keywords */
{kDDOS} {
    System.out.printf("Valid DDoS: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kBRUTEFORCE} {
    System.out.printf("Valid BruteForce: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{kPORTSCAN} {
    System.out.printf("Valid PortScan: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Numbers */
{scientificNumbers} {
    System.out.printf("Valid Scientific Number: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{floatNumbers} {
    System.out.printf("Valid Float Number: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{integerNumbers} {
    System.out.printf("Valid Integer: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Strings */
{stringLiteral} {
    System.out.printf("Valid String: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{unterminatedString} {
    System.out.printf("Lexical Error: Unterminated string %s. Line %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Comments */
{singleLineComment} {}

{multiLineComment} {}

{unterminatedComment} {
    System.out.printf("Lexical Error: Unterminated comment. Line %d, Column: %d\n", yyline + 1, yycolumn + 1);
}

/* Relational Operators */
{opLTE} {
    System.out.printf("Valid Less Than or Equal: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{opGTE} {
    System.out.printf("Valid Greater Than or Equal: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{opEQ} {
    System.out.printf("Valid Equal: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{opNE} {
    System.out.printf("Valid Not Equal: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{opLT} {
    System.out.printf("Valid Less Than: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{opGT} {
    System.out.printf("Valid Greater Than: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Assignment Operator */
{opASS} {
    System.out.printf("Valid Assignment: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Arithmetic Operators */
{opPLUS} {
    System.out.printf("Valid Plus: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{opSUB} {
    System.out.printf("Valid Minus: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{opMUL} {
    System.out.printf("Valid Multiply: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{opDIV} {
    System.out.printf("Valid Divide: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Parentheses */
{leftParen} {
    System.out.printf("Valid Left Parenthesis: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

{rightParen} {
    System.out.printf("Valid Right Parenthesis: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Valid Identifier */
({smallAlphabets}|{capitalAlphabets}|{specialChar})({smallAlphabets}|{capitalAlphabets}|{digits}|{specialChar})* {
    System.out.printf("Valid Identifier: %s at Line: %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}

/* Invalid Character */
. {
    System.out.printf("Lexical Error: Invalid character '%s'. Line %d, Column: %d\n", yytext(), yyline + 1, yycolumn + 1);
}
