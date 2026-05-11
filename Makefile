.PHONY: help install dev up down logs migrate seed mcp-server rag-pipeline

help:
	@echo "Available targets:"
	@echo "  install        Install Python dependencies with uv"
	@echo "  dev            Run the LangGraph dev server"
	@echo "  up             Start docker-compose services (postgres + mlflow)"
	@echo "  down           Stop docker-compose services"
	@echo "  logs           Tail docker-compose logs"
	@echo "  migrate        Apply pending alembic migrations (upgrade head)"
	@echo "  seed           Seed itsm tickets into Postgres"
	@echo "  mcp-server     Run the ITSM ticket MCP server (HTTP)"
	@echo "  rag-pipeline   Index every PDF under static/knowledge_base into Qdrant"

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

migrate:
	uv run alembic upgrade head

seed:
	uv run seed-tickets

mcp-server:
	uv run run-ticket-mcp-server

rag-pipeline:
	uv run run-rag-pipeline
