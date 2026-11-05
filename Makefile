.PHONY: install test check quality db export index sync

UV_RUN := uv run $(if $(wildcard .env),--env-file .env)

install:
	uv sync

test: check
	$(UV_RUN) pytest || test $$? -eq 5  # 5: no tests collected

check:
	uv run ruff check
	uv run ruff format --check
	uv run lint-imports

quality:
	uv run ruff check --fix
	uv run ruff format

db:
	docker compose up -d --wait db

export:
	$(UV_RUN) selecta export

index: db
	$(UV_RUN) selecta index

sync: db
	$(UV_RUN) selecta sync
