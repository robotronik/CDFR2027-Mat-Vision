# Raccourcis de compilation autour de CMake.

BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc 2>/dev/null || echo 2)
CMAKE ?= cmake
CTEST ?= ctest

.DEFAULT_GOAL := build
.PHONY: build configure test run run-match calibrate clean

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

test: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

run: build
	./$(BUILD_DIR)/matvision-server

run-match: build
	./$(BUILD_DIR)/matvision-server --match

calibrate: build
	./$(BUILD_DIR)/matvision-calibrate

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
