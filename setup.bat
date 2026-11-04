@echo off
REM Cria o ambiente virtual do projeto e instala as dependencias.
REM Basta dar duplo clique neste arquivo (ou rodar "setup.bat" no terminal)
REM em qualquer computador novo com Python instalado.

cd /d "%~dp0"

if not exist ".venv" (
    echo [*] Criando ambiente virtual em .venv ...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo [!] Falha ao criar o venv. Verifique se o Python esta instalado e no PATH.
        pause
        exit /b 1
    )
) else (
    echo [*] Ambiente virtual .venv ja existe, pulando criacao.
)

echo [*] Instalando dependencias de requirements.txt ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo [OK] Ambiente pronto.
echo Para usar no terminal:      .venv\Scripts\activate
echo Para usar no VSCode/Jupyter: selecione o kernel ".venv" (Python Environments)
pause
