#!/bin/bash

# AI-CRM System Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}AI-CRM System Deployment Script${NC}"
echo -e "${BLUE}====================================${NC}"
echo

# Function to show menu
show_menu() {
    echo "Available commands:"
    echo "  1. build    - Build all Docker images"
    echo "  2. start    - Start all services"
    echo "  3. stop     - Stop all services"
    echo "  4. restart  - Restart all services"
    echo "  5. logs     - View logs from all services"
    echo "  6. clean    - Remove all containers and volumes"
    echo "  7. status   - Check system status"
    echo "  8. backup   - Backup database"
    echo "  9. monitor  - Monitor system performance"
    echo "  0. exit     - Exit this script"
    echo
}

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed!${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Prerequisites check passed!${NC}"
}

# Build function
build_system() {
    echo -e "${YELLOW}Building all services...${NC}"
    docker-compose build --parallel
    echo -e "${GREEN}Build complete!${NC}"
}

# Start function
start_system() {
    echo -e "${YELLOW}Starting AI-CRM System...${NC}"
    docker-compose up -d
    echo
    echo -e "${GREEN}System is starting up...${NC}"
    echo "Dashboard will be available at http://localhost"
    echo
    sleep 5
    echo "Checking system health..."
    curl -s http://localhost/health || echo -e "${RED}System is not responding${NC}"
}

# Stop function
stop_system() {
    echo -e "${YELLOW}Stopping all services...${NC}"
    docker-compose down
    echo -e "${GREEN}Services stopped.${NC}"
}

# Restart function
restart_system() {
    echo -e "${YELLOW}Restarting all services...${NC}"
    docker-compose down
    docker-compose up -d
    echo -e "${GREEN}Services restarted.${NC}"
}

# Logs function
show_logs() {
    echo -e "${YELLOW}Showing logs (Press Ctrl+C to stop)...${NC}"
    docker-compose logs -f
}

# Clean function
clean_system() {
    echo -e "${RED}WARNING: This will remove all containers and volumes!${NC}"
    read -p "Are you sure? (y/N): " confirm
    if [[ $confirm == [yY] ]]; then
        docker-compose down -v --remove-orphans
        docker system prune -f
        echo -e "${GREEN}System cleaned.${NC}"
    else
        echo "Operation cancelled."
    fi
}

# Status function
check_status() {
    echo -e "${YELLOW}Checking system status...${NC}"
    echo
    echo "Container Status:"
    docker-compose ps
    echo
    echo "System Health:"
    curl -s http://localhost/health || echo -e "${RED}System is not responding${NC}"
    echo
    echo "Resource Usage:"
    docker stats --no-stream
}

# Backup function
backup_database() {
    echo -e "${YELLOW}Backing up database...${NC}"
    timestamp=$(date +%Y%m%d_%H%M%S)
    docker-compose exec -T postgres pg_dump -U admin ai_crm > "backup_${timestamp}.sql"
    echo -e "${GREEN}Database backed up to backup_${timestamp}.sql${NC}"
}

# Monitor function
monitor_system() {
    echo -e "${YELLOW}Starting system monitoring...${NC}"
    echo "Press Ctrl+C to stop"
    docker stats
}

# Main script
check_prerequisites

# If command line argument provided, execute it
if [ $# -eq 1 ]; then
    case $1 in
        build) build_system ;;
        start) start_system ;;
        stop) stop_system ;;
        restart) restart_system ;;
        logs) show_logs ;;
        clean) clean_system ;;
        status) check_status ;;
        backup) backup_database ;;
        monitor) monitor_system ;;
        *) echo -e "${RED}Unknown command: $1${NC}" ;;
    esac
    exit 0
fi

# Interactive mode
while true; do
    show_menu
    read -p "Enter your choice (0-9): " choice
    
    case $choice in
        1) build_system ;;
        2) start_system ;;
        3) stop_system ;;
        4) restart_system ;;
        5) show_logs ;;
        6) clean_system ;;
        7) check_status ;;
        8) backup_database ;;
        9) monitor_system ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) echo -e "${RED}Invalid choice. Please try again.${NC}" ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
    clear
done
