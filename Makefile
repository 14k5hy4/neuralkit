.PHONY: test install clean

install:
	pip install -e .

test:
	python -m unittest discover tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
