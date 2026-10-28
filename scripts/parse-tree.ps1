# Run the JFlex lexer + CUP-generated LALR parser on one DSL file and print the
# token stream and the parse tree. Runs only when you call it.
#
#   .\scripts\parse-tree.ps1                        # the reference workload
#   .\scripts\parse-tree.ps1 examples\residual_block.dsl
#   .\scripts\parse-tree.ps1 build\demo\parse_lab\invalid.dsl   # example syntax error
param([string]$DslFile = 'examples\attention_mlp.dsl')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (-not (Test-Path 'build\fusionc.jar')) {
    throw 'build/fusionc.jar is missing -- run .\scripts\build.ps1 first'
}

java -cp 'build\fusionc.jar' io.fusionc.ParseTree $DslFile
exit $LASTEXITCODE
