COLOR ?= auto
PYTEST_FLAGS ?= -v
PYTEST_PARALLEL ?= auto # overwritten in CI

IMAGE_TAG ?= latest

.PHONY: build
build:
	docker build -t apolo-deploy:$(IMAGE_TAG) .

.PHONY: push
push:
	@if [ -z "$(TARGET_IMAGE)" ]; then \
		echo "Error: TARGET_IMAGE is not set"; \
		exit 1; \
	fi
	docker tag apolo-deploy:$(IMAGE_TAG) $(TARGET_IMAGE)
	docker push $(TARGET_IMAGE)

.PHONY: apolo-push
apolo-push:
	@if [ -z "$(TARGET_IMAGE)" ]; then \
		echo "Error: TARGET_IMAGE is not set"; \
		exit 1; \
	fi
	apolo push apolo-deploy:$(IMAGE_TAG) image:$(TARGET_IMAGE)

.PHONY: setup
setup:
	pip install -r requirements/python-test.txt
	pre-commit install

.PHONY: lint
lint: format
	python3 -m pip install types-PyYAML types-requests
	mypy modules

.PHONY: format
format:
	pre-commit run --all-files --show-diff-on-failure
