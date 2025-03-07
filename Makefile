# Makefile для проекта Face Embeddings

# Базовые настройки
PYTHON := python3
PROJECT_NAME := Face Embeddings
DOCKER_IMAGE_NAME := face-embeddings
DOCKER_CONTAINER_NAME := face-embeddings-container

# Пути
SRC_DIR := src
DATA_DIR := data
CHECKPOINTS_DIR := checkpoints

# Переменные окружения
export PYTHONPATH := /app

.PHONY: build run train inference clean docker-clean help

# Сборка Docker-образа
build:
	docker build -t $(DOCKER_IMAGE_NAME) .

# Запуск Docker-контейнера с монтированием данных
run:
	docker run --shm-size=16G -it --user root --name $(DOCKER_CONTAINER_NAME) \
		--gpus all \
		-v $(PWD)/$(DATA_DIR):/app/data \
		-v $(PWD)/$(CHECKPOINTS_DIR):/app/checkpoints \
		$(DOCKER_IMAGE_NAME) bash

# Запуск обучения внутри контейнера
train:
	uv run  $(PYTHON) -m $(SRC_DIR).train

# Запуск инференса внутри контейнера
inference:
	uv run  $(PYTHON) -m $(SRC_DIR).inference

# Очистка
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Cleaned up cache files"

# Очистка Docker ресурсов
docker-clean:
	docker stop $(DOCKER_CONTAINER_NAME) 2>/dev/null || true
	docker rm $(DOCKER_CONTAINER_NAME) 2>/dev/null || true
	docker rmi $(DOCKER_IMAGE_NAME) 2>/dev/null || true
	@echo "Docker resources cleaned"

# Показать справку
help:
	@echo "Доступные команды:"
	@echo "  make build         - Собрать Docker-образ"
	@echo "  make run           - Запустить Docker-контейнер"
	@echo "  make train         - Запустить обучение модели"
	@echo "  make inference     - Запустить инференс модели"
	@echo "  make clean         - Удалить файлы кеша"
	@echo "  make docker-clean  - Очистить Docker ресурсы"
