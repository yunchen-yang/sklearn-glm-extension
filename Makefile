# base shell
SHELL := /bin/bash

# export env variables
-include .env
export

# STANDARD COLORS
ifneq (,$(findstring xterm,${TERM}))
	BLACK        := $(shell tput -Txterm setaf 0)
	RED          := $(shell tput -Txterm setaf 1)
	GREEN        := $(shell tput -Txterm setaf 2)
	YELLOW       := $(shell tput -Txterm setaf 3)
	LIGHTPURPLE  := $(shell tput -Txterm setaf 4)
	PURPLE       := $(shell tput -Txterm setaf 5)
	BLUE         := $(shell tput -Txterm setaf 6)
	WHITE        := $(shell tput -Txterm setaf 7)
	RESET        := $(shell tput -Txterm sgr0)
else
	BLACK        := ""
	RED          := ""
	GREEN        := ""
	YELLOW       := ""
	LIGHTPURPLE  := ""
	PURPLE       := ""
	BLUE         := ""
	WHITE        := ""
	RESET        := ""
endif

TARGET_COLOR := $(PURPLE)
ARCHITECTURE := $(shell uname -p)
WORK_DIR := $(shell pwd)

# VENV variables
VENV_NAME ?= ./.venv
VENV_BIN_PATH ?= $(VENV_NAME)/bin
VENV_ACTIVATE = $(VENV_BIN_PATH)/activate
VENV_PYTHON = $(VENV_BIN_PATH)/python

# PYTHON
PYTHON = $(shell which python3)
PYTHON_VERSION := $(shell ${PYTHON} --version | cut -d " " -f 2 | cut -d "." -f 1-2)

clean: ## Removes virtual environment.
	rm -fr $(VENV_NAME)

all: venv dev-install ## Creates a new virtual environment and installs the dev dependency.

refresh-env: clean all ## Regenerates the working environment and installs the dependencies from scratch.

venv: ## Creates a new virtual environment using venv, with the latest version of pip.
	(\
		test -d $(VENV_NAME) || $(PYTHON) -m venv $(VENV_NAME) ;\
		. $(VENV_ACTIVATE) ;\
		pip install --upgrade pip ;\
		pip -V ;\
		which python ;\
	)

install: venv ## Creates a new virtual environment and install base the dependencies on it.
	. $(VENV_ACTIVATE) && (\
			pip install ./ ;\
			pip install pip-tools ;\
	)

dev-install: venv ## Creates a new virtual environment and install the development dependencies on it.
	. $(VENV_ACTIVATE) && (\
		pip install --editable .[dev] ;\
		pip install pip-tools ;\
		python -m ipykernel install --user --name glmext --display-name "glmext" ;\
	)
