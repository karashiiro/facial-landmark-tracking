SHELL := /bin/bash
.DEFAULT_GOAL := help 

help: ## Show this help
	@echo Dependencies: python
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	python -m pip install -r ./requirements.txt

preprocess: ## Build the training metadata (requires dataset from README)
	python ./preprocess_landmarks.py

train: ## Train the regressor
	python ./train_shape_predictor.py

demo: ## Run the demo script
	python ./demo.py