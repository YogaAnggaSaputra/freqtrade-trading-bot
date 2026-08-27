.PHONY: up down restart logs ps build test lint clean db-migrate db-revision

up:
	docker compose up -d

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f

ps:
	docker compose ps

build:
	docker compose build

test:
	pytest tests/ -v

lint:
	ruff check services/ shared/ tests/
	mypy services/ shared/ tests/ --ignore-missing-imports

clean:
	docker compose down -v
	-rm -rf data/parquet/*

db-migrate:
	docker compose exec postgres alembic upgrade head

db-revision:
	docker compose exec postgres alembic revision --autogenerate -m "$(message)"

shell-db:
	docker compose exec postgres psql -U botbinance -d botbinance

shell-redis:
	docker compose exec redis redis-cli
