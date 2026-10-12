.PHONY: install train test api mlflow docker-up docker-down

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

train:
	python -m training.train

test:
	python -m pytest

api:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	python -m mlflow ui --host 0.0.0.0 --port 5000

docker-up:
	docker compose up --build

docker-down:
	docker compose down

