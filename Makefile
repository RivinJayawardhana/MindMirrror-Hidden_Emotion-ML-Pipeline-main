.PHONY: install data train all clean

install:
	pip install -r requirements.txt

data:
	python pipelines/emotion_data_pipeline.py

train:
	python pipelines/emotion_train_pipeline.py

all: data train

clean:
	rm -rf artifacts/
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf utils/__pycache__
	rm -rf pipelines/__pycache__
