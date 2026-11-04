.PHONY: install test check quality

install:
	uv sync

test: check
	uv run pytest || test $$? -eq 5  # 5: no tests collected

check:
	uv run ruff check
	uv run ruff format --check
	uv run lint-imports

quality:
	uv run ruff check --fix
	uv run ruff format
