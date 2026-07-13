@echo off
REM Retrieval Playground - Quick Start Script for Windows

echo.
echo ####################################################
echo # Retrieval Playground - Docker Workshop Setup   #
echo ####################################################
echo.

REM Prefer Docker Compose V2 (`docker compose`); fall back to legacy `docker-compose`
docker compose version >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "COMPOSE=docker compose"
) else (
    where docker-compose >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Docker Compose is not installed!
        echo Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/
        pause
        exit /b 1
    )
    set "COMPOSE=docker-compose"
)

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed!
    echo.
    echo Please install Docker Desktop from:
    echo   https://docs.docker.com/desktop/install/windows-install/
    echo.
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not running!
    echo.
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

echo [OK] Docker is installed and running
echo.

REM Check for .env file
if not exist .env (
    echo [WARNING] .env file not found!
    echo.
    echo Creating .env from template...
    if exist .env.example (
        copy .env.example .env >nul
        echo [OK] .env file created
        echo.
        echo [IMPORTANT] Edit the .env file and add your API keys!
        echo    Open .env in Notepad and replace the placeholder values.
        echo.
        echo Press any key after you've updated the .env file...
        pause >nul
    ) else (
        echo [ERROR] .env.example not found. Creating basic template...
        (
            echo GOOGLE_API_KEY=your_gemini_api_key
        ) > .env
        echo [OK] .env file created
        echo.
        echo [IMPORTANT] Edit the .env file and add your API keys!
        echo.
        pause
        exit /b 1
    )
)

echo Building Docker image (this may take 5-10 minutes on first run)...
echo.
%COMPOSE% build

echo.
echo [OK] Build complete!
echo.
echo Starting Jupyter Notebook server...
echo.
%COMPOSE% up -d

echo.
echo [OK] Jupyter Notebook is running!
echo.
echo ####################################################
echo # Access your notebooks at:                      #
echo #                                                 #
echo #   http://localhost:8888                        #
echo #                                                 #
echo ####################################################
echo.
echo Navigate to: retrieval_playground/tutorial/
echo.
echo Useful commands:
echo   Stop:     %COMPOSE% down
echo   Restart:  %COMPOSE% restart
echo   Logs:     %COMPOSE% logs -f
echo.
pause
