.PHONY: clean test

all: test

test:
	poetry run pytest -s -v -x

coverage.xml:
	poetry run pytest --junitxml=junit.xml --cov=tensordata --cov-report xml:coverage.xml

clean:
	rm -rf coverage.xml

output/figure%.svg: tensordata/figures/figure%.py
	mkdir -p output
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 poetry run fbuild $*