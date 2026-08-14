#ifndef VAVX3_PRIMITIVES_H
#define VAVX3_PRIMITIVES_H

/* ═══════════════════════════════════════════════════════════════
 * vavx3_primitives.h — 兼容包装 (2026-08-16 整合修复)
 *
 * 原实现是 vavx3_core.h 的真子集 (load/geo_rotate/toroidal_inversion/
 * laplacian/phi4/braid/yamabe/entangle/vortex 等 9 个函数与 core 重复,
 * 且重复 typedef vavx3_512i — 同时包含两个头文件触发 C++ 重定义错误)。
 * 现改为转发 vavx3_core.h (超集); vavx3_mask_addr_512 别名已在 core 中提供。
 * ═══════════════════════════════════════════════════════════════ */

#include "vavx3_core.h"

#endif /* VAVX3_PRIMITIVES_H */
