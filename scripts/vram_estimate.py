#!/usr/bin/env python3
"""
vram_estimate.py — 静态估算 LLM/VLM 推理峰值显存（不加载模型权重）

用途：在下载几十 GB 权重之前，先用公式判断这张卡能不能跑。

公式：
    峰值 = 权重 + KV Cache + 激活余量 + 框架开销

    KV Cache = 2(K,V) * layers * kv_heads * head_dim * bytes * tokens * concurrency

用法：
    python vram_estimate.py --csv data/models.csv --gpu-mem 8 --ctx 8192
    python vram_estimate.py --model qwen3.5-2b --ctx 32768 --concurrency 2
    python vram_estimate.py --csv data/models.csv --ctx 8192 --json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# 已知的注意力结构：layers, kv_heads, head_dim
# 注：hybrid 架构（如 Gated DeltaNet）只有部分层需要 KV Cache，
#     这里用 effective_layers 表示"实际缓存 KV 的层数"，避免高估。
ARCH = {
    "qwen3.5-2b":            {"layers": 24, "kv_heads": 2, "head_dim": 256, "effective_layers": 6},
    "minicpm-ocr":           {"layers": 24, "kv_heads": 2, "head_dim": 256, "effective_layers": 6},
    "lfm3-1.7b":             {"layers": 24, "kv_heads": 2, "head_dim": 128, "effective_layers": 6},
    "lfm3-audio-2b":         {"layers": 24, "kv_heads": 2, "head_dim": 128, "effective_layers": 6},
    "qwen3-vl-embedding-2b": {"layers": 28, "kv_heads": 2, "head_dim": 128, "effective_layers": 0},
    "qwen3-asr-0.6b":        {"layers": 28, "kv_heads": 8, "head_dim": 128, "effective_layers": 28},
    "qwen3-asr-1.7b":        {"layers": 28, "kv_heads": 8, "head_dim": 128, "effective_layers": 28},
    "glm-asr-nano":          {"layers": 28, "kv_heads": 8, "head_dim": 128, "effective_layers": 28},
    "cohere-transcribe":     {"layers": 8,  "kv_heads": 8, "head_dim": 128, "effective_layers": 8},
    "granite-speech-4.1-2b": {"layers": 16, "kv_heads": 8, "head_dim": 128, "effective_layers": 16},
    "voxcpm2":               {"layers": 0,  "kv_heads": 0, "head_dim": 0,   "effective_layers": 0},
}

DTYPE_BYTES = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "q8": 1.0,
    "q4": 0.5,
}

# 框架开销：CUDA context + cuBLAS workspace + 碎片
FRAMEWORK_OVERHEAD_GB = 0.45

# vLLM 额外：CUDA graph + 预分配池（默认 gpu_memory_utilization=0.9）
VLLM_RESERVE_FACTOR = 1.15


@dataclass
class Row:
    id: str
    name: str
    task: str
    params_b: float
    weight_gb: float
    kv_gb: float
    activation_gb: float
    total_gb: float
    grade: str
    note: str


def kv_cache_gb(arch: dict, ctx: int, concurrency: int, cache_dtype: str = "bf16") -> float:
    """KV Cache 显存（GB）。effective_layers 为 0 表示无自回归 KV（如 embedding）。"""
    n_layers = arch.get("effective_layers", arch["layers"])
    if n_layers == 0:
        return 0.0
    bytes_per_val = DTYPE_BYTES.get(cache_dtype, 2.0)
    per_token = 2 * n_layers * arch["kv_heads"] * arch["head_dim"] * bytes_per_val
    return per_token * ctx * concurrency / (1024 ** 3)


def activation_gb(task: str, params_b: float) -> float:
    """
    激活余量经验值（GB）。多模态任务按输入尺寸显著放大，这里给保守下限。
    真实值必须在 Phase 1-3 实测校准。
    """
    t = task.upper()
    if t == "OCR":
        return max(0.8, params_b * 0.50)      # 视觉 token 展开，高分辨率时远超此值
    if t in {"ASR", "AST"}:
        return max(0.3, params_b * 0.15)      # 音频编码器前向，随音频长度增长
    if t in {"TTS", "ASR+TTS"}:
        return max(0.5, params_b * 0.40)
    if t == "EMBEDDING":
        return max(0.5, params_b * 0.20)
    return max(0.3, params_b * 0.15)


def grade(total: float, usable: float) -> str:
    """绝对阈值，便于和 README 的人工核对表对齐。"""
    if total <= 4.0:
        return "A"
    if total <= 6.2:
        return "B"
    if total <= usable:
        return "C"
    return "D"


def estimate(row: dict, ctx: int, concurrency: int, precision: str,
             usable_gb: float, runtime: str) -> Row:
    arch = ARCH.get(row["id"])
    params_b = float(row["params_b"])

    # 权重：优先用实测文件体积，没有就按参数量推
    key = {"bf16": "weight_bf16_gb", "q8": "weight_q8_gb", "q4": "weight_q4_gb"}[precision]
    weight = float(row.get(key) or 0) or params_b * DTYPE_BYTES[precision]

    kv = kv_cache_gb(arch, ctx, concurrency) if arch else 0.0
    act = activation_gb(row["task"], params_b)
    total = weight + kv + act + FRAMEWORK_OVERHEAD_GB
    if "vllm" in runtime.lower() or "sglang" in runtime.lower():
        total *= VLLM_RESERVE_FACTOR

    # 厂商硬性最低显存要求（如 OXR 官方要求 24GB）：取 max，超过可用显存即判 D
    hard_min = float(row.get("hard_min_vram_gb") or 0)
    if hard_min:
        total = max(total, hard_min)

    g = grade(total, usable_gb)
    note = row.get("note", "")
    if hard_min:
        note = f"[官方最低 {hard_min:g}GB] " + note
    csv_grade = row.get("grade", "").strip()
    if csv_grade and csv_grade != g:
        note += f"（人工核对档位：{csv_grade}）"

    return Row(
        id=row["id"],
        name=row["name"],
        task=row["task"],
        params_b=params_b,
        weight_gb=round(weight, 2),
        kv_gb=round(kv, 2),
        activation_gb=round(act, 2),
        total_gb=round(total, 2),
        grade=f"{g}/{csv_grade}" if csv_grade else g,
        note=note,
    )


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(description="静态估算推理峰值显存")
    p.add_argument("--csv", default="data/models.csv", help="模型清单 CSV")
    p.add_argument("--model", help="只估算指定 id")
    p.add_argument("--gpu-mem", type=float, default=8.0, help="显卡总显存 GB")
    p.add_argument("--desktop-headroom", type=float, default=0.8,
                   help="桌面/系统常驻占用 GB（Windows 建议 0.8）")
    p.add_argument("--ctx", type=int, default=8192, help="上下文 token 数")
    p.add_argument("--concurrency", type=int, default=1, help="并发序列数")
    p.add_argument("--precision", choices=list(DTYPE_BYTES), default="bf16")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] 找不到 {csv_path}（先 cd 到仓库根目录）", file=sys.stderr)
        return 1

    rows = load_csv(csv_path)
    if args.model:
        rows = [r for r in rows if r["id"] == args.model]
        if not rows:
            print(f"[error] CSV 里没有 id={args.model}", file=sys.stderr)
            return 1

    usable = args.gpu_mem - args.desktop_headroom
    results = [estimate(r, args.ctx, args.concurrency, args.precision, usable, r["runtime"])
               for r in rows]
    results.sort(key=lambda r: r.total_gb)

    if args.json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
        return 0

    print(f"\n目标显卡 {args.gpu_mem:g}GB | 桌面占用 {args.desktop_headroom:g}GB "
          f"| 可支配 {usable:g}GB | 精度 {args.precision} | ctx {args.ctx} | 并发 {args.concurrency}\n")
    print(f"{'分级':<5}{'模型':<30}{'权重':>8}{'KV':>8}{'激活':>8}{'合计':>9}")
    print("-" * 72)
    for r in results:
        print(f"{r.grade:<5}{r.name[:29]:<30}{r.weight_gb:>8.2f}{r.kv_gb:>8.2f}"
              f"{r.activation_gb:>8.2f}{r.total_gb:>9.2f}")
    print("-" * 72)
    print("A 开箱即跑 / B 可行需限制 / C 边缘 / D 不可行")
    print("注意：估算值，非实测。多模态模型的激活项随输入尺寸变化很大，务必实测校准。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
