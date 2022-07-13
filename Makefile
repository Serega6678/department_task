SHELL = /bin/bash

.PHONY: bootstrap split_data extract_data train predict

.DEFAULT_GOAL = help

bootstrap:
	poetry install

split_data:
	poetry run python -m src.split_data data/train.json train_initial valid_initial

extract_data:
	poetry run python -m src.prepare_train_data data/train_initial.json data/valid_initial.json

train:
	poetry run python -m src.train data/train_initial_embeddings.npy data/valid_initial_embeddings.npy

predict:
	poetry run python -m src.predict data/test_example.json 0.5