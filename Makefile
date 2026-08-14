# 浑天 4320D 流形计算 Makefile
# 高维几何视角实践验证

CC = gcc
CFLAGS = -O3 -march=native -Wall
LDFLAGS = -lm

PYTHON = python3

SRC_DIR = tests
BUILD_DIR = build
INCLUDE_DIR = ../ternary-core/include

# 源文件
C_SRCS = $(SRC_DIR)/test_vavx3_manifold.c
PY_SRCS = $(SRC_DIR)/entropy_spin_pure.py $(SRC_DIR)/huntian_quantum_chemistry.py

# 目标文件
C_TARGETS = $(BUILD_DIR)/test_vavx3_manifold

.PHONY: all clean test python

all: $(C_TARGETS)

# 编译 C 程序
$(BUILD_DIR)/test_vavx3_manifold: $(SRC_DIR)/test_vavx3_manifold.c ../ternary-core/include/vavx3_cpu_impl.h
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $< -o $@ $(LDFLAGS)

# 运行所有测试
test: all python
	@echo "\n=== 运行所有测试 ==="
	@$(BUILD_DIR)/test_vavx3_manifold
	@$(PYTHON) $(SRC_DIR)/entropy_spin_pure.py
	@$(PYTHON) $(SRC_DIR)/huntian_quantum_chemistry.py

# 运行 Python 测试
python:
	@$(PYTHON) $(SRC_DIR)/entropy_spin_pure.py
	@$(PYTHON) $(SRC_DIR)/huntian_quantum_chemistry.py

# 清理
clean:
	rm -rf $(BUILD_DIR)/*

# 创建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 帮助
help:
	@echo "浑天 4320D 流形计算构建系统"
	@echo ""
	@echo "目标:"
	@echo "  all    - 编译所有 C 程序"
	@echo "  test   - 运行所有测试 (C + Python)"
	@echo "  python - 运行 Python 测试"
	@echo "  clean  - 清理构建目录"
	@echo "  help   - 显示帮助信息"