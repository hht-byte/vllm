#!/usr/bin/env python3
from __future__ import annotations

import argparse

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm.attention_utils.fasa import offline_calibration, save_calibration_result


def build_demo_dataset(num_examples: int, num_steps: int, num_layers: int, num_heads: int, head_dim: int):
    torch.manual_seed(42)
    dataset = []
    dominant_fc = min(2, head_dim // 2 - 1)
    for _ in range(num_examples):
        queries = torch.randn(num_steps, num_layers, num_heads, head_dim)
        keys = torch.randn(num_steps, num_layers, num_heads, head_dim)
        keys[..., dominant_fc * 2 : dominant_fc * 2 + 2] += queries[
            ..., dominant_fc * 2 : dominant_fc * 2 + 2
        ]
        dataset.append({"queries": queries, "keys": keys})
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline calibration for FASA TIP dominant FCs")
    parser.add_argument("--output", type=str, default="/home/user/webapp/fasa_calibration_demo.json")
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--topk-k", type=int, default=2)
    parser.add_argument("--num-dominant-fcs", type=int, default=2)
    args = parser.parse_args()

    dataset = build_demo_dataset(
        args.num_examples,
        args.num_steps,
        args.num_layers,
        args.num_heads,
        args.head_dim,
    )
    result = offline_calibration(
        calibration_examples=dataset,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        topk_k=args.topk_k,
        num_dominant_fcs=args.num_dominant_fcs,
    )
    save_calibration_result(result, args.output)
    print(f"Saved calibration result to {args.output}")
    print(result.to_dict())


if __name__ == "__main__":
    main()
