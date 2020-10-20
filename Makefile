VENV = venv
PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python

VENV_TEMP = venv_temp

ARGS =


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	python3 -m venv $(VENV)


dev-install:
	$(PIP) install -r requirements.build.txt
	$(PIP) install -r requirements.dev.txt
	$(PIP) install -r requirements.txt


dev-venv: venv-create dev-install


dev-flake8:
	$(PYTHON) -m flake8 layered_vision tests setup.py


dev-pylint:
	$(PYTHON) -m pylint layered_vision tests setup.py


dev-lint: dev-flake8 dev-pylint


dev-pytest:
	$(PYTHON) -m pytest -p no:cacheprovider $(ARGS)


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
	$(VENV_TEMP)/bin/pip install -r requirements.build.txt
	$(VENV_TEMP)/bin/pip install --force-reinstall ./dist/*.tar.gz
	$(VENV_TEMP)/bin/pip install --force-reinstall ./dist/*.whl


start:
	$(PYTHON) -m layered_vision start $(ARGS)
