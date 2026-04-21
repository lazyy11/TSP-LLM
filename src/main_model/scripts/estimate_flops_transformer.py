#!/usr/bin/env python3
"""根据 Transformer FLOPs 公式估算模型计算量"""

import argparse
import json
import math
from dataclasses import dataclass, asdict


@dataclass
class TransformerSpec:
    layers: int
    sequence_length: int
    hidden_size: int
    ff_size: int
    num_heads: int


def estimate_transformer_flops(spec: TransformerSpec) -> float:
    """基于标准公式估算 FLOPs"""
    L = spec.layers
    n = spec.sequence_length
    d = spec.hidden_size
    d_ff = spec.ff_size

    attention_flops = L * (4 * n * d * d + 2 * n * n * d)
    ff_flops = L * 8 * n * d * d_ff
    return attention_flops + ff_flops


def human_readable_flops(flops: float) -> str:
    units = ["", "K", "M", "G", "T", "P"]
    if flops <= 0:
        return "0"
    idx = min(int(math.log(flops, 1000)), len(units) - 1)
    value = flops / (1000 ** idx)
    return f"{value:.2f} {units[idx]}FLOPs"


def main():
    parser = argparse.ArgumentParser(description="估算 Transformer 模型 FLOPs")
    parser.add_argument("layers", type=int, help="Transformer 层数")
    parser.add_argument("sequence_length", type=int, help="序列长度 n")
    parser.add_argument("hidden_size", type=int, help="隐藏维度 d")
    parser.add_argument("ff_size", type=int, help="前馈层宽度 d_ff")
    parser.add_argument("--heads", type=int, default=12, help="注意力头数")
    parser.add_argument("--output", help="输出 JSON 文件路径")
    args = parser.parse_args()

    spec = TransformerSpec(
        layers=args.layers,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        ff_size=args.ff_size,
        num_heads=args.heads,
    )

    total_flops = estimate_transformer_flops(spec)
    readable = human_readable_flops(total_flops)

    result = {
        "spec": asdict(spec),
        "total_flops": total_flops,
        "readable": readable,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

