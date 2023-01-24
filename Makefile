VENV = venv

ifeq ($(OS),Windows_NT)
	VENV_BIN = $(VENV)/Scripts
else
	VENV_BIN = $(VENV)/bin
endif

PYTHON = $(VENV_BIN)/python
PIP = $(VENV_BIN)/python -m pip

SYSTEM_PYTHON = python3

VENV_TEMP = venv_temp

ARGS =

IMAGE_NAME = de4code/layered-vision_unstable
IMAGE_TAG = develop


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	$(SYSTEM_PYTHON) -m venv $(VENV)


dev-install:
	$(PIP) install --requirement=requirements.build.txt
	$(PIP) install \
		--constraint=constraints.txt \
		--requirement=requirements.dev.txt \
		--requirement=requirements.txt


dev-venv: venv-create dev-install


dev-flake8:
	$(PYTHON) -m flake8 layered_vision tests setup.py


dev-pylint:
	$(PYTHON) -m pylint layered_vision tests setup.py


dev-mypy:
	$(PYTHON) -m mypy --ignore-missing-imports --show-error-codes \
		layered_vision tests setup.py


dev-lint: dev-flake8 dev-pylint dev-mypy


dev-pytest:
	$(PYTHON) -m pytest -v -p no:cacheprovider $(ARGS)


dev-watch:
	$(PYTHON) -m pytest_watch -- -p no:cacheprovider -p no:warnings $(ARGS)


dev-test: dev-lint dev-pytest


dev-remove-dist:
	rm -rf ./dist


dev-build-dist:
	$(PYTHON) setup.py sdist bdist_wheel


dev-list-dist-contents:
	tar -ztvf dist/layered-*.tar.gz


dev-get-version:
	$(PYTHON) setup.py --version


dev-test-install-dist:
	$(MAKE) VENV=$(VENV_TEMP) venv-create
	$(VENV_TEMP)/bin/pip install --requirement=requirements.build.txt
	$(VENV_TEMP)/bin/pip install --force-reinstall ./dist/*.tar.gz
	$(VENV_TEMP)/bin/pip install --force-reinstall ./dist/*.whl


start:
	$(PYTHON) -m layered_vision start $(ARGS)


docker-build:
	docker build . -t $(IMAGE_NAME):$(IMAGE_TAG)


docker-run:
	docker run \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=unix$$DISPLAY \
		-v /dev/shm:/dev/shm \
		--rm $(IMAGE_NAME):$(IMAGE_TAG) $(ARGS)
