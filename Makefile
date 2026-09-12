# Convenience wrappers.  `make run IN=clip.mp4 OUT=clip_norm.mp4` is the
# one-command entry point section 7 asks for.

UID  := $(shell id -u)
GID  := $(shell id -g)
IMAGE      ?= uwnorm:latest
GPU_IMAGE  ?= uwnorm:gpu
IN         ?=
OUT        ?=
ARGS       ?=
PRESET     ?=
DOCKER_RUN  = docker run --rm -u $(UID):$(GID) \
                -v $(CURDIR)/in:/work/in:ro \
                -v $(CURDIR)/out:/work/out \
                -v $(CURDIR)/cache:/work/cache

ifneq ($(PRESET),)
CONFIG_ARG = --config $(PRESET)
endif

.PHONY: help build build-gpu run run-gpu analyze report shell test lint dirs clean size

help:
	@printf 'targets:\n'
	@printf '  build       build the CPU image\n'
	@printf '  build-gpu   build the CUDA image\n'
	@printf '  run         IN=clip.mp4 OUT=clip_norm.mp4 [PRESET=deep_blue] [ARGS=...]\n'
	@printf '  run-gpu     same, on the GPU image with NVDEC/NVENC\n'
	@printf '  analyze     IN=clip.mp4 - analysis + report only\n'
	@printf '  test        run the test suite locally\n'
	@printf '  size        report the CPU image size against the 1.5 GB target\n'

dirs:
	@mkdir -p in out cache

build:
	docker build --build-arg UID=$(UID) --build-arg GID=$(GID) -t $(IMAGE) .

build-gpu:
	docker build -f Dockerfile.gpu --build-arg UID=$(UID) --build-arg GID=$(GID) -t $(GPU_IMAGE) .

check-io:
	@test -n "$(IN)"  || { echo 'set IN=<file in ./in>'; exit 1; }
	@test -n "$(OUT)" || { echo 'set OUT=<file in ./out>'; exit 1; }

run: dirs check-io
	$(DOCKER_RUN) $(IMAGE) run /work/in/$(IN) -o /work/out/$(OUT) \
		--cache-dir /work/cache --report /work/out/$(basename $(OUT)).report.html \
		--verify $(CONFIG_ARG) $(ARGS)

run-gpu: dirs check-io
	$(DOCKER_RUN) --gpus all $(GPU_IMAGE) run /work/in/$(IN) -o /work/out/$(OUT) \
		--cache-dir /work/cache --report /work/out/$(basename $(OUT)).report.html \
		--device cuda --hwaccel cuda --verify $(CONFIG_ARG) $(ARGS)

analyze: dirs
	@test -n "$(IN)" || { echo 'set IN=<file in ./in>'; exit 1; }
	$(DOCKER_RUN) $(IMAGE) run /work/in/$(IN) --dry-run \
		--cache-dir /work/cache --report /work/out/$(basename $(IN)).report.html \
		$(CONFIG_ARG) $(ARGS)

shell: dirs
	$(DOCKER_RUN) --entrypoint /bin/bash -it $(IMAGE)

test:
	python3 -m pytest -q

size:
	@docker image inspect $(IMAGE) --format '{{.Size}}' \
		| awk '{printf "%s: %.2f GB (target <= 1.50 GB)\n", "$(IMAGE)", $$1/1e9}'

clean:
	rm -rf out/*.mp4 out/*.mov out/*.mkv out/*.html out/*.png cache/*.json
