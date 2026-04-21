#!/usr/bin/env python3
"""测量已训练 Seq2Seq 模型的推理速度（样本/秒与延迟）"""

import argparse
import json
import os
import time
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_dummy_batch(tokenizer, sequence_length: int, batch_size: int, device: torch.device):
    """生成随机输入，近似模拟真实 workload"""
    # 随机生成 token id，范围限定在 vocab size 内
    vocab_size = min(len(tokenizer), tokenizer.vocab_size or len(tokenizer))
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def measure_latency(model, tokenizer, device: torch.device, batch_size: int, sequence_length: int,
                    warmup: int, runs: int, max_new_tokens: int):
    """计算平均延迟（秒）"""
    batch = generate_dummy_batch(tokenizer, sequence_length, batch_size, device)

    # 预热
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model.generate(**batch, max_new_tokens=max_new_tokens)

    latencies: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(**batch, max_new_tokens=max_new_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

    return latencies


def benchmark(model_path: str, batch_sizes: List[int], seq_lengths: List[int], warmup: int,
              runs: int, max_new_tokens: int, device: str, output_dir: str):
    device_obj = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device_obj)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    results = []
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            latencies = measure_latency(
                model=model,
                tokenizer=tokenizer,
                device=device_obj,
                batch_size=batch_size,
                sequence_length=seq_len,
                warmup=warmup,
                runs=runs,
                max_new_tokens=max_new_tokens,
            )

            avg_latency = sum(latencies) / len(latencies)
            if len(latencies) >= 4:
                sorted_latencies = sorted(latencies)
                idx = max(int(0.95 * len(sorted_latencies)) - 1, 0)
                p95_latency = sorted_latencies[idx]
            else:
                p95_latency = max(latencies)
            throughput = batch_size / avg_latency

            results.append({
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "runs": runs,
                "avg_latency_seconds": round(avg_latency, 4),
                "p95_latency_seconds": round(p95_latency, 4),
                "throughput_samples_per_second": round(throughput, 2),
            })

            print(f"batch={batch_size}, L={seq_len}: avg {avg_latency:.4f}s | p95 {p95_latency:.4f}s | {throughput:.2f} samples/s")

    summary_path = os.path.join(output_dir, "inference_benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": model_path,
            "device": device,
            "runs": runs,
            "warmup": warmup,
            "max_new_tokens": max_new_tokens,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"结果已保存至: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="测量 Seq2Seq 模型推理速度")
    parser.add_argument("model_path", help="模型或检查点目录")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 32, 128], help="批量列表")
    parser.add_argument("--seq-lengths", nargs="*", type=int, default=[64, 128, 256], help="输入序列长度列表")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    parser.add_argument("--runs", type=int, default=10, help="统计次数")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="解码时生成的最大新 token 数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="使用的设备")
    parser.add_argument("--output-dir", default="benchmark_outputs", help="结果保存目录")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark(
        model_path=args.model_path,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        warmup=args.warmup,
        runs=args.runs,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

