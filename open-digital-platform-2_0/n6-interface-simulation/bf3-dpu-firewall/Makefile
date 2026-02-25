#
# Makefile for NVIDIA BlueField-3 N6 Interface Firewall
# Production-grade build system for DOCA applications
#
# Copyright (c) 2024 NVIDIA Corporation
# SPDX-License-Identifier: Proprietary
#

# ===============================================
# Build Configuration
# ===============================================

# Application Information
APP_NAME        := n6_firewall
APP_VERSION     := 2.6.0
APP_DESCRIPTION := "N6 Interface Firewall for NVIDIA BlueField-3 DPU"

# Build Directories
SRCDIR          := src
INCDIR          := include
LIBDIR          := lib
OBJDIR          := build/obj
BINDIR          := build/bin
DEPDIR          := build/deps

# Source Files
SOURCES         := $(wildcard $(SRCDIR)/*.c)
OBJECTS         := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DEPENDENCIES    := $(SOURCES:$(SRCDIR)/%.c=$(DEPDIR)/%.d)

# Target Binary
TARGET          := $(BINDIR)/$(APP_NAME)

# ===============================================
# DOCA SDK Configuration
# ===============================================

# DOCA SDK Installation Path
DOCA_INSTALL_DIR ?= /opt/mellanox/doca

# Check for custom DOCA installation
ifdef DOCA_HOME
    DOCA_INSTALL_DIR := $(DOCA_HOME)
endif

# DOCA SDK Paths
DOCA_INC_DIR    := $(DOCA_INSTALL_DIR)/include
DOCA_LIB_DIR    := $(DOCA_INSTALL_DIR)/lib/aarch64-linux-gnu
DOCA_EXAMPLES   := $(DOCA_INSTALL_DIR)/share/doca/examples

# Check for x86_64 host build
ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
    DOCA_LIB_DIR := $(DOCA_INSTALL_DIR)/lib/x86_64-linux-gnu
endif

# PKG-Config Setup
PKG_CONFIG_PATH := $(DOCA_LIB_DIR)/pkgconfig:$(PKG_CONFIG_PATH)
export PKG_CONFIG_PATH

# ===============================================
# DPDK Configuration
# ===============================================

# DPDK Installation (usually bundled with DOCA)
DPDK_INSTALL_DIR ?= $(DOCA_INSTALL_DIR)
DPDK_INC_DIR     := $(DPDK_INSTALL_DIR)/include/dpdk
DPDK_LIB_DIR     := $(DOCA_LIB_DIR)

# ===============================================
# Compiler Configuration
# ===============================================

# Compiler Selection
CC              := gcc
CXX             := g++

# Cross-compilation for BlueField DPU (if needed)
ifdef CROSS_COMPILE
    CC := $(CROSS_COMPILE)gcc
    CXX := $(CROSS_COMPILE)g++
endif

# Compiler Standard
CSTD            := -std=gnu11
CXXSTD          := -std=gnu++17

# ===============================================
# Compiler Flags
# ===============================================

# Base Flags
CFLAGS_BASE     := -Wall -Wextra -Werror -Wno-unused-parameter
CFLAGS_BASE     += -fPIC -pthread
CFLAGS_BASE     += $(CSTD)

# Debug/Release Configuration
ifdef DEBUG
    CFLAGS_BASE += -O0 -g3 -DDEBUG -fno-omit-frame-pointer
    CFLAGS_BASE += -fsanitize=address -fsanitize=undefined
    BUILD_TYPE  := debug
else ifdef PROFILE
    CFLAGS_BASE += -O2 -g1 -DNDEBUG -pg
    BUILD_TYPE  := profile
else
    CFLAGS_BASE += -O3 -DNDEBUG -flto
    CFLAGS_BASE += -march=native -mtune=native
    BUILD_TYPE  := release
endif

# Architecture-specific optimizations
ifeq ($(ARCH),aarch64)
    CFLAGS_BASE += -mcpu=native
else ifeq ($(ARCH),x86_64)
    CFLAGS_BASE += -mavx2 -mfma
endif

# Include Directories
CFLAGS_INCLUDES := -I$(INCDIR)
CFLAGS_INCLUDES += -I$(DOCA_INC_DIR)
CFLAGS_INCLUDES += -I$(DPDK_INC_DIR)

# Feature Flags
CFLAGS_FEATURES := -DN6_FIREWALL_VERSION=\"$(APP_VERSION)\"
CFLAGS_FEATURES += -D_GNU_SOURCE
CFLAGS_FEATURES += -DALLOW_EXPERIMENTAL_API

# Final CFLAGS
CFLAGS          := $(CFLAGS_BASE) $(CFLAGS_INCLUDES) $(CFLAGS_FEATURES)

# ===============================================
# Linker Configuration
# ===============================================

# Base Linker Flags
LDFLAGS_BASE    := -pthread

# Debug/Release Linker Flags
ifdef DEBUG
    LDFLAGS_BASE += -fsanitize=address -fsanitize=undefined
else ifdef PROFILE
    LDFLAGS_BASE += -pg
else
    LDFLAGS_BASE += -flto -Wl,-O2
endif

# Library Directories
LDFLAGS_DIRS    := -L$(DOCA_LIB_DIR) -L$(DPDK_LIB_DIR)

# Runtime Library Path
LDFLAGS_RPATH   := -Wl,-rpath,$(DOCA_LIB_DIR)
LDFLAGS_RPATH   += -Wl,-rpath,$(DPDK_LIB_DIR)

# Final LDFLAGS
LDFLAGS         := $(LDFLAGS_BASE) $(LDFLAGS_DIRS) $(LDFLAGS_RPATH)

# ===============================================
# Library Linking
# ===============================================

# DOCA Libraries
LIBS_DOCA       := -ldoca_flow -ldoca_devemu -ldoca_ctx -ldoca_dev
LIBS_DOCA       += -ldoca_common -ldoca_utils -ldoca_log

# DPDK Libraries (Core)
LIBS_DPDK_CORE  := -lrte_eal -lrte_ethdev -lrte_mbuf -lrte_mempool
LIBS_DPDK_CORE  += -lrte_ring -lrte_kvargs -lrte_telemetry

# DPDK Libraries (Extensions)
LIBS_DPDK_EXT   := -lrte_flow_classify -lrte_gro -lrte_gso
LIBS_DPDK_EXT   += -lrte_hash -lrte_meter -lrte_acl

# System Libraries
LIBS_SYSTEM     := -lm -ldl -lnuma -lrt

# All Libraries
LIBS            := $(LIBS_DOCA) $(LIBS_DPDK_CORE) $(LIBS_DPDK_EXT) $(LIBS_SYSTEM)

# Try to use pkg-config if available
ifeq ($(shell pkg-config --exists libdoca 2>/dev/null && echo yes),yes)
    CFLAGS  += $(shell pkg-config --cflags libdoca)
    LIBS    := $(shell pkg-config --libs libdoca) $(LIBS_SYSTEM)
    $(info Using pkg-config for DOCA libraries)
else
    $(info Using manual library linking)
endif

# ===============================================
# Build Targets
# ===============================================

.PHONY: all clean install uninstall help check-deps
.DEFAULT_GOAL := all

# Main Build Target
all: check-deps $(TARGET)

# Binary Creation
$(TARGET): $(OBJECTS) | $(BINDIR)
	@echo "Linking $(APP_NAME) ($(BUILD_TYPE) build)..."
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) $(LIBS)
	@echo "✓ Built $(APP_NAME) v$(APP_VERSION)"
	@echo "  Binary: $@"
	@echo "  Build:  $(BUILD_TYPE)"
	@echo "  Size:   $$(du -h $@ | cut -f1)"

# Object File Compilation
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR) $(DEPDIR)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -MMD -MP -MF $(DEPDIR)/$*.d -c $< -o $@

# Directory Creation
$(BINDIR) $(OBJDIR) $(DEPDIR):
	@mkdir -p $@

# Include Dependencies
-include $(DEPENDENCIES)

# ===============================================
# Installation Targets
# ===============================================

PREFIX          ?= /usr/local
BINDIR_INSTALL  := $(PREFIX)/bin
CONFDIR_INSTALL := $(PREFIX)/etc/n6-firewall
SHAREDIR_INSTALL := $(PREFIX)/share/n6-firewall

install: $(TARGET)
	@echo "Installing $(APP_NAME)..."
	install -d $(BINDIR_INSTALL)
	install -m 755 $(TARGET) $(BINDIR_INSTALL)/
	install -d $(CONFDIR_INSTALL)
	install -m 644 config/*.conf $(CONFDIR_INSTALL)/ 2>/dev/null || true
	install -d $(SHAREDIR_INSTALL)
	install -m 644 docs/* $(SHAREDIR_INSTALL)/ 2>/dev/null || true
	@echo "✓ Installed $(APP_NAME) to $(PREFIX)"

uninstall:
	@echo "Uninstalling $(APP_NAME)..."
	rm -f $(BINDIR_INSTALL)/$(APP_NAME)
	rm -rf $(CONFDIR_INSTALL)
	rm -rf $(SHAREDIR_INSTALL)
	@echo "✓ Uninstalled $(APP_NAME)"

# ===============================================
# Development Targets
# ===============================================

# Clean Build Artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	@echo "✓ Clean completed"

# Deep Clean (including dependencies)
distclean: clean
	@echo "Deep cleaning..."
	find . -name "*.orig" -delete 2>/dev/null || true
	find . -name "*.rej" -delete 2>/dev/null || true
	find . -name "*~" -delete 2>/dev/null || true
	@echo "✓ Deep clean completed"

# Check Dependencies
check-deps:
	@echo "Checking build dependencies..."
	@command -v $(CC) >/dev/null || (echo "ERROR: $(CC) not found" && exit 1)
	@test -d $(DOCA_INSTALL_DIR) || (echo "ERROR: DOCA SDK not found at $(DOCA_INSTALL_DIR)" && exit 1)
	@test -f $(DOCA_INC_DIR)/doca_flow.h || (echo "ERROR: DOCA headers not found" && exit 1)
	@test -f $(DOCA_LIB_DIR)/libdoca_flow.so || (echo "ERROR: DOCA libraries not found" && exit 1)
	@echo "✓ Dependencies check passed"

# Static Analysis
analyze:
	@echo "Running static analysis..."
	cppcheck --enable=all --inconclusive --std=c11 $(SRCDIR)/ $(INCDIR)/
	@echo "✓ Static analysis completed"

# Code Formatting
format:
	@echo "Formatting code..."
	find $(SRCDIR) $(INCDIR) -name "*.c" -o -name "*.h" | xargs clang-format -i
	@echo "✓ Code formatting completed"

# ===============================================
# Testing Targets
# ===============================================

# Unit Tests
test: $(TARGET)
	@echo "Running unit tests..."
	./scripts/run_tests.sh
	@echo "✓ Unit tests completed"

# Integration Tests
test-integration: $(TARGET)
	@echo "Running integration tests..."
	./scripts/test_integration.sh
	@echo "✓ Integration tests completed"

# Performance Benchmarks
benchmark: $(TARGET)
	@echo "Running performance benchmarks..."
	./scripts/benchmark.sh
	@echo "✓ Benchmarks completed"

# ===============================================
# Docker and Deployment
# ===============================================

# Docker Build
docker-build:
	@echo "Building Docker image..."
	docker build -t n6-firewall:$(APP_VERSION) .
	docker tag n6-firewall:$(APP_VERSION) n6-firewall:latest
	@echo "✓ Docker image built"

# Docker Run
docker-run:
	@echo "Running Docker container..."
	docker run --rm --privileged --cap-add=ALL \
		-v /dev/hugepages:/dev/hugepages \
		-v /sys/bus/pci:/sys/bus/pci \
		n6-firewall:latest
	@echo "✓ Docker container stopped"

# ===============================================
# Documentation and Help
# ===============================================

# Generate Documentation
docs:
	@echo "Generating documentation..."
	doxygen Doxyfile 2>/dev/null || echo "Doxygen not available"
	@echo "✓ Documentation generated"

# Build Information
info:
	@echo "=== N6 Firewall Build Information ==="
	@echo "Application:      $(APP_NAME) v$(APP_VERSION)"
	@echo "Description:      $(APP_DESCRIPTION)"
	@echo "Build Type:       $(BUILD_TYPE)"
	@echo "Target Arch:      $(ARCH)"
	@echo "Compiler:         $(CC)"
	@echo "DOCA SDK:         $(DOCA_INSTALL_DIR)"
	@echo "Install Prefix:   $(PREFIX)"
	@echo "Source Files:     $(words $(SOURCES))"
	@echo "Object Files:     $(words $(OBJECTS))"
	@echo "====================================="

# Help Target
help:
	@echo "NVIDIA BlueField-3 N6 Firewall Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all           - Build the application (default)"
	@echo "  clean         - Clean build artifacts"
	@echo "  distclean     - Deep clean including temp files"
	@echo "  install       - Install to system (PREFIX=$(PREFIX))"
	@echo "  uninstall     - Remove from system"
	@echo "  check-deps    - Check build dependencies"
	@echo "  analyze       - Run static code analysis"
	@echo "  format        - Format code with clang-format"
	@echo "  test          - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  benchmark     - Run performance benchmarks"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  docs          - Generate documentation"
	@echo "  info          - Show build information"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Build options:"
	@echo "  DEBUG=1       - Enable debug build"
	@echo "  PROFILE=1     - Enable profiling build"
	@echo "  CROSS_COMPILE=<prefix> - Cross-compilation prefix"
	@echo "  DOCA_HOME=<path> - Custom DOCA SDK path"
	@echo "  PREFIX=<path> - Installation prefix (default: /usr/local)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Release build"
	@echo "  make DEBUG=1           # Debug build"
	@echo "  make DOCA_HOME=/opt/doca # Custom DOCA path"
	@echo "  make install PREFIX=/opt # Install to /opt"

# ===============================================
# Development Helpers
# ===============================================

# Show Variables (for debugging Makefile)
show-vars:
	@echo "SOURCES: $(SOURCES)"
	@echo "OBJECTS: $(OBJECTS)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "LIBS: $(LIBS)"
	@echo "DOCA_INSTALL_DIR: $(DOCA_INSTALL_DIR)"
	@echo "PKG_CONFIG_PATH: $(PKG_CONFIG_PATH)"