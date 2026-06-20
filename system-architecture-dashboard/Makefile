.PHONY: help install dev prod test docker docker-up docker-down clean lint format

help:
	@echo "System Architecture Dashboard - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       - Install dependencies"
	@echo "  make dev           - Setup development environment"
	@echo ""
	@echo "Running:"
	@echo "  make run           - Run development server"
	@echo "  make run-prod      - Run production server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker        - Build Docker image"
	@echo "  make docker-up     - Start Docker containers"
	@echo "  make docker-down   - Stop Docker containers"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linter (flake8)"
	@echo "  make format        - Format code (black)"
	@echo "  make test          - Run tests (pytest)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove generated files"
	@echo "  make clean-all     - Remove all generated files + venv"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install -e .
	cp .env.example .env
	@echo "✓ Development environment ready"
	@echo "  Edit .env to customize settings"
	@echo "  Run 'make run' to start"

run:
	export FLASK_ENV=development && export FLASK_DEBUG=True && python3 -m backend.app

run-prod:
	export FLASK_ENV=production && export FLASK_DEBUG=False && gunicorn -w 4 -b 0.0.0.0:5000 backend.app:create_app

test:
	pytest tests/ -v --cov=backend

lint:
	flake8 backend/ tests/

format:
	black backend/ tests/

docker:
	docker build -t system-architecture-dashboard:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f dashboard

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

clean-all: clean
	rm -rf venv/
	rm -f .env

.DEFAULT_GOAL := help
