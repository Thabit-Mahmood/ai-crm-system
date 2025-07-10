# AI-CRM System Makefile

.PHONY: help build up down restart logs clean test lint

# Default target
help:
	@echo "AI-CRM System Management Commands:"
	@echo "  make build    - Build all Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make restart  - Restart all services"
	@echo "  make logs     - View logs from all services"
	@echo "  make clean    - Remove all containers and volumes"
	@echo "  make test     - Run system tests"
	@echo "  make lint     - Run code linting"
	@echo "  make dev      - Start in development mode"
	@echo "  make prod     - Start in production mode"

# Build all services
build:
	docker compose build --parallel

# Start all services
up:
	docker compose up -d
	@echo "AI-CRM System is starting..."
	@echo "Dashboard will be available at http://localhost"
	@echo "Use 'make logs' to view service logs"

# Stop all services
down:
	docker compose down

# Restart all services
restart: down up

# View logs
logs:
	docker compose logs -f

# View logs for specific service
logs-%:
	docker compose logs -f $*

# Clean everything
clean:
	docker compose down -v --remove-orphans
	docker system prune -f

# Development mode with live reload
dev:
	docker compose -f docker compose.yml -f docker compose.dev.yml up

# Production mode with optimizations
prod:
	docker compose -f docker compose.yml -f docker compose.prod.yml up -d

# Run tests
test:
	@echo "Running system tests..."
	@docker compose exec data-ingestion npm test
	@docker compose exec language-detector pytest
	@docker compose exec nlp-processor pytest
	@docker compose exec alert-manager pytest
	@docker compose exec analytics-engine pytest

# Lint code
lint:
	@echo "Linting code..."
	@cd services/data-ingestion && npm run lint
	@cd services/language-detector && pylint *.py
	@cd services/nlp-processor && pylint *.py
	@cd services/alert-manager && pylint *.py
	@cd services/analytics-engine && pylint *.py

# Database operations
db-backup:
	docker compose exec postgres pg_dump -U admin ai_crm > backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore:
	@read -p "Enter backup filename: " backup_file; \
	docker compose exec -T postgres psql -U admin ai_crm < $$backup_file

# Service-specific commands
ingestion-shell:
	docker compose exec data-ingestion sh

nlp-shell:
	docker compose exec nlp-processor bash

postgres-shell:
	docker compose exec postgres psql -U admin ai_crm

redis-cli:
	docker compose exec redis redis-cli

# Health check
health:
	@curl -s http://localhost/health || echo "System is not responding"

# Performance monitoring
monitor:
	@echo "Opening system monitoring..."
	@docker stats

# GPU monitoring (if available)
gpu-monitor:
	@docker compose exec nlp-processor nvidia-smi -l 1
