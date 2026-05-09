.PHONY: help install dev up down logs

help:
	@echo "Available targets:"
	@echo "  install   Install Python dependencies with uv"
	@echo "  dev       Run the LangGraph dev server"
	@echo "  up        Start docker-compose services (postgres + mlflow)"
	@echo "  down      Stop docker-compose services"
	@echo "  logs      Tail docker-compose logs"

install:
	uv sync

dev:
	uv run langgraph dev

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f
