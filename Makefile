.PHONY: install data train serve test lint drift docker-build docker-run bench

install:
	pip install -r requirements-dev.txt

data:
	python scripts/download_data.py

train:            ## train + quality gates -> artifacts/
	python -m training.train

serve:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

drift:            ## offline drift study -> reports/
	python -m monitoring.drift_report

bench:            ## latency benchmark against a running server
	python -m monitoring.benchmark --url http://localhost:8000

docker-build:
	docker build -t credit-risk-api:latest .

docker-run:
	docker run --rm -p 8000:8000 credit-risk-api:latest
