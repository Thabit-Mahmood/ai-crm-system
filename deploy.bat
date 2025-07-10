@echo off
REM AI-CRM System Deployment Script for Windows

echo ====================================
echo AI-CRM System Deployment Script
echo ====================================
echo.

if "%1"=="" goto menu
goto %1

:menu
echo Available commands:
echo   1. build    - Build all Docker images
echo   2. start    - Start all services
echo   3. stop     - Stop all services
echo   4. restart  - Restart all services
echo   5. logs     - View logs from all services
echo   6. clean    - Remove all containers and volumes
echo   7. status   - Check system status
echo   8. exit     - Exit this script
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto start
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto restart
if "%choice%"=="5" goto logs
if "%choice%"=="6" goto clean
if "%choice%"=="7" goto status
if "%choice%"=="8" goto end

echo Invalid choice. Please try again.
goto menu

:build
echo Building all services...
docker-compose build --parallel
echo Build complete!
pause
goto menu

:start
echo Starting AI-CRM System...
docker-compose up -d
echo.
echo System is starting up...
echo Dashboard will be available at http://localhost
echo.
timeout /t 5
echo Checking system health...
curl -s http://localhost/health
echo.
pause
goto menu

:stop
echo Stopping all services...
docker-compose down
echo Services stopped.
pause
goto menu

:restart
echo Restarting all services...
docker-compose down
docker-compose up -d
echo Services restarted.
pause
goto menu

:logs
echo Showing logs (Press Ctrl+C to stop)...
docker-compose logs -f
pause
goto menu

:clean
echo WARNING: This will remove all containers and volumes!
set /p confirm="Are you sure? (y/N): "
if /i "%confirm%"=="y" (
    docker-compose down -v --remove-orphans
    docker system prune -f
    echo System cleaned.
) else (
    echo Operation cancelled.
)
pause
goto menu

:status
echo Checking system status...
echo.
echo Container Status:
docker-compose ps
echo.
echo System Health:
curl -s http://localhost/health || echo System is not responding
echo.
pause
goto menu

:end
echo Goodbye!
exit /b
