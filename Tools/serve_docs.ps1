#!/usr/bin/env pwsh
# Script para iniciar el servidor de documentación

Write-Host "🚀 Iniciando servidor de documentación..." -ForegroundColor Green
Write-Host ""

# Cambiar al directorio Tools
Set-Location $PSScriptRoot

# Verificar que existe mkdocs.yml
if (-not (Test-Path "mkdocs.yml")) {
    Write-Host "❌ Error: No se encuentra mkdocs.yml" -ForegroundColor Red
    exit 1
}

# Verificar que mkdocs está instalado
try {
    $null = Get-Command mkdocs -ErrorAction Stop
} catch {
    Write-Host "❌ Error: MkDocs no está instalado" -ForegroundColor Red
    Write-Host "Instala las dependencias con: pip install -r docs-requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "Servidor de documentacion iniciandose en http://localhost:8080" -ForegroundColor Cyan
Write-Host "Presiona Ctrl+C para detener el servidor" -ForegroundColor Yellow
Write-Host ""

# Iniciar servidor
mkdocs serve --dev-addr localhost:8080
