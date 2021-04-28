O_DIR=

train:
	rm -v /experiments/checkpoint/*
	rm -v /experiments/log/*
	python train.py

evaluation:
	python evaluation.py

colab:
	%load_ext tensorboard
	%tensorboard --logdir ./$(O_DIR)/experiments/log

tensorboard:
	%load_ext tensorboard
	%tensorboard --logdir ./$(O_DIR)/experiments/log
