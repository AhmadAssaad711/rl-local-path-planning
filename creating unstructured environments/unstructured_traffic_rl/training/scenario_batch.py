"""Scenario batch generation utility."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..scenarios.generator import ScenarioGenerator


def generate_manifest(count: int, output_path: Path, seed: int = 0) -> None:
    generator = ScenarioGenerator()
    scenarios = generator.generate_batch(count, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(item) for item in scenarios], indent=2))
    print(f"Saved {count} generated scenarios to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batch of procedural scenario configs.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="datasets/generated_scenarios.json")
    args = parser.parse_args()
    generate_manifest(args.count, Path(args.output), seed=args.seed)


if __name__ == "__main__":
    main()
