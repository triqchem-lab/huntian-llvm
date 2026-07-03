# HunTian LLVM — 浑天编译器集成

**LLVM optimization integration for the HunTian (浑天) 4320D manifold architecture.**

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

LLVM compiler plugin and Clang builtin definitions for the **HunTian 4320D toroidal manifold** computing architecture. Implements sovereign V-AVX3 instructions, quantum clock phase tracking, Xuangong tuning, and virtual AVX3 ISA on standard x86-64 hardware.

### Structure

| Directory | Content |
|-----------|---------|
| `HunTian/` | Core integration — THE_CONSTITUTION, virtual AVX3 core, geometric vortex dynamics, harmonic evolution (14 headers) |
| `VAVX3/` | ISA sub-module — primitives, core ops, axioms, BLAS (5 headers) |
| `clang/` | Clang builtin header (`VAVX3.h`) |
| `passes/` | LLVM compiler passes — QuantumClock, XuangongTuning |
| `tests/` | Manifold verification test |

### Key Constants

| Constant | Value | Source |
|----------|-------|--------|
| MANIFOLD_DIM | 4320 | HunTian toroidal geometry |
| CHERN_NUMBER | 2.0 | Topological invariant |
| N14_NQR_FREQ | 3.17 MHz | Nitrogen-14 quantum clock |
| CRITICAL_TEMP | 2.17 K | Superfluid lambda point |
| TARGET_FREQ | 144 Hz | Polar winding fundamental |

---

<a name="chinese"></a>
## 中文

**浑天 4320D 环面流形**计算架构的 LLVM 编译器插件与 Clang builtin 定义。在标准 x86-64 硬件上实现主权 V-AVX3 指令集、量子时钟相位追踪、旋宫调律、虚拟 AVX3 ISA。

### 目录结构

| 目录 | 内容 |
|------|------|
| `HunTian/` | 核心集成 — 宪法公理, 虚拟AVX3核心, 几何涡旋动力学, 谐波演化 (14头文件) |
| `VAVX3/` | ISA 子模块 — 原语, 核心运算, 公理, BLAS (5头文件) |
| `clang/` | Clang builtin 头 (`VAVX3.h`) |
| `passes/` | LLVM 编译器插件 — 量子时钟, 旋宫调律 |
| `tests/` | 流形验证测试 |

### 关键常数

| 常数 | 值 | 来源 |
|------|-----|------|
| MANIFOLD_DIM | 4320 | 浑天环面几何 |
| CHERN_NUMBER | 2.0 | 拓扑不变量 |
| N14_NQR_FREQ | 3.17 MHz | 氮-14 量子时钟 |
| CRITICAL_TEMP | 2.17 K | 超流 λ 相变点 |
| TARGET_FREQ | 144 Hz | 极向缠绕基频 |

## 关联仓库 Related Repositories

| 仓库 | 内容 |
|------|------|
| [ternary-core](https://github.com/triqchem-lab/ternary-core) | 三进制核心指令集 |
| [quantum-physics](https://github.com/triqchem-lab/quantum-physics) | 熵旋理论与质量谱 |
| [quantum-chemistry](https://github.com/triqchem-lab/quantum-chemistry) | 全息周期表与C60 |
| [scholar-loop](https://github.com/triqchem-lab/scholar-loop) | 实验验证引擎 |
| [discrete-mathematics](https://github.com/triqchem-lab/discrete-mathematics) | Agda 形式化证明 |
