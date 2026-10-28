# Open the parse-tree GUI (Swing window): source on the right, the tree the CUP parser
# built drawn as a graph on the left, and a pop-up dialog on a syntax error.
#
#   .\scripts\parse-tree-gui.ps1
#   .\scripts\parse-tree-gui.ps1 examples\residual_block.dsl
#   .\scripts\parse-tree-gui.ps1 examples\syntax_error.dsl      # shows the error dialog
param([string]$DslFile = 'examples\attention_mlp.dsl')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (-not (Test-Path 'build\fusionc.jar')) {
    throw 'build\fusionc.jar is missing -- run .\scripts\build.ps1 first'
}

java -cp 'build\fusionc.jar' io.fusionc.ParseTreeGui $DslFile
exit $LASTEXITCODE
