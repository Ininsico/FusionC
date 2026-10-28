# Build FusionC: generate the JFlex scanner and the CUP parser, then compile and package.
#   .\scripts\build.ps1
#
# Requires a JDK 11+ on PATH (or JAVA_HOME set) and the jars in tools/:
#   tools/jflex.jar, tools/java-cup.jar, tools/java-cup-runtime.jar
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

function Resolve-Java {
    if ($env:JAVA_HOME -and (Test-Path (Join-Path $env:JAVA_HOME 'bin/java.exe'))) {
        return (Join-Path $env:JAVA_HOME 'bin/java.exe')
    }
    $cmd = Get-Command java -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = Get-ChildItem 'C:\Program Files\Eclipse Adoptium\*\bin\java.exe' -ErrorAction SilentlyContinue
    if ($candidates) { return $candidates[0].FullName }
    throw 'no JDK found: set JAVA_HOME or put java.exe on PATH'
}

$java  = Resolve-Java
$javac = Join-Path (Split-Path -Parent $java) 'javac.exe'

$pkg     = 'src/main/java/io/fusionc/frontend'
$classes = 'build/classes'
if (Test-Path $classes) { Remove-Item -Recurse -Force $classes }
New-Item -ItemType Directory -Force -Path $pkg, $classes | Out-Null

Write-Host '[1/4] JFlex: Scanner.flex -> src/main/java/io/fusionc/frontend/Scanner.java'
# The generated scanner/parser/sym live in the source tree (standard for JFlex/CUP
# projects): the build regenerates them, and IDEs resolve them without extra config.
# %cup mode makes the generated scanner implement java_cup.runtime.Scanner, so JFlex itself
# needs the CUP runtime on its classpath.
& $java -cp 'tools/jflex.jar;tools/java-cup-runtime.jar' jflex.Main -d $pkg 'src/main/jflex/Scanner.flex'
if ($LASTEXITCODE -ne 0) { throw 'jflex failed' }

Write-Host '[2/4] CUP: Parser.cup -> sym.java + parser.java (same package)'
# the vbmacher java-cup jar ships no Main-Class attribute, so the entry point is named explicitly
& $java -cp 'tools/java-cup.jar;tools/java-cup-runtime.jar' java_cup.Main `
    -package io.fusionc.frontend -parser parser -symbols sym -destdir $pkg 'src/main/cup/Parser.cup'
if ($LASTEXITCODE -ne 0) { throw 'cup failed' }

Write-Host '[3/4] javac'
$sources = Get-ChildItem -Recurse -Filter *.java 'src/main/java' | ForEach-Object { $_.FullName }
& $javac -encoding UTF-8 -cp 'tools/java-cup-runtime.jar' -d $classes @sources
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '[4/4] package build/fusionc.jar (with the CUP runtime bundled)'
Push-Location $classes
& (Join-Path (Split-Path -Parent $java) 'jar.exe') xf (Resolve-Path '../../tools/java-cup-runtime.jar')
Pop-Location
& (Join-Path (Split-Path -Parent $java) 'jar.exe') cfe 'build/fusionc.jar' 'io.fusionc.Main' -C $classes .
if ($LASTEXITCODE -ne 0) { throw 'jar failed' }

Write-Host ''
Write-Host 'build ok -> build/fusionc.jar'
Write-Host 'try:  java -jar build/fusionc.jar --input examples/attention_mlp.dsl --fuse -o build/out_fused.cu'
