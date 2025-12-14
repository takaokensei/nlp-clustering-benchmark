# Script para criar ambiente virtual com uv e Python 3.12
$ErrorActionPreference = "Stop"

# Adicionar uv ao PATH
$uvPath = "$env:USERPROFILE\.local\bin"
if (Test-Path $uvPath) {
    $env:Path = "$uvPath;$env:Path"
}

# Verificar se uv esta disponivel
try {
    $uvVersion = & uv --version 2>&1
    Write-Host "UV encontrado: $uvVersion"
} catch {
    Write-Host "UV nao encontrado. Instalando..."
    powershell -Command "Invoke-WebRequest -UseBasicParsing https://astral.sh/uv/install.ps1 | Invoke-Expression"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# Instalar Python 3.12 se necessario
Write-Host "Instalando Python 3.12..."
& uv python install 3.12

# Remover .venv antigo (se possivel)
if (Test-Path ".venv") {
    Write-Host "Tentando remover .venv antigo..."
    Get-Process | Where-Object {$_.Path -like "*\.venv\*"} | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
}

# Remover .venv antigo completamente
if (Test-Path ".venv") {
    Write-Host "Removendo .venv antigo completamente..."
    Get-Process | Where-Object {$_.Path -like "*\.venv\*"} | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
}

# Criar novo ambiente virtual com Python 3.12
Write-Host "Criando novo ambiente virtual com Python 3.12..."
& uv venv .venv --python 3.12

# Instalar PyTorch com CUDA usando uv no ambiente virtual
Write-Host "Instalando PyTorch com suporte CUDA usando uv..."
& uv pip install --python .venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar outras dependencias usando uv no ambiente virtual
Write-Host "Instalando outras dependencias usando uv..."
& uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# Verificar CUDA
Write-Host ""
Write-Host "Verificando instalacao..."
& .venv\Scripts\python.exe check_cuda.py

Write-Host ""
Write-Host "Ambiente virtual criado com sucesso usando uv!"
Write-Host "Para ativar: .venv\Scripts\Activate.ps1"
