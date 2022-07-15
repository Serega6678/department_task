SHELL = /bin/bash

.PHONY: bootstrap split_data extract_data train predict estimate_thr

.DEFAULT_GOAL = help

bootstrap:
	poetry install

split_data:
	poetry run python -m src.split_data data/train.json train_initial valid_initial test_initial

extract_data:
	poetry run python -m src.prepare_train_data data/train_initial.json data/valid_initial.json data/test_initial.json

train:
	poetry run python -m src.train data/train_initial_embeddings.npy data/valid_initial_embeddings.npy

predict_valid:
	poetry run python -m src.predict data/valid_initial.json 0 data/checkpoints/best.ckpt

estimate_thr: predict_valid
	rm data/valid_initial_answer.json
	poetry run python -m src.find_best_threshold data/valid_initial.json

predict_test:
	poetry run python -m src.predict data/test_initial.json 0.69 data/checkpoints/best.ckpt

measure_metric:
	poetry run python -m src.find_best_threshold data/test_initial.json 0.69
