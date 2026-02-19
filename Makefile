.PHONY: install ingest run test eval clean docker-build docker-run

# Install dependencies
install:
	pip install -r requirements.txt

# Run ingestion to build index
ingest:
	python -m ingestion.ingest_texts

# Run the API server
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v

# Run evaluation
eval:
	python -m eval.run_eval

# Clean generated files
clean:
	rm -rf index/*.json index/*.npy
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache

# Build Docker image
docker-build:
	docker build -t fin-filings-rag .

# Run Docker container
docker-run:
	docker run -p 8000:8000 fin-filings-rag

# Full pipeline: install, ingest, test
all: install ingest test eval
