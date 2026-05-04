"""Summarize how many dimensions are labeled helpful, harmful, or neutral.

This file was added to compare attribution capacity across embeddings and
benchmarks after dimension attribution scores have been computed.
"""

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    input_dir: Path
    output_path: Path

    @classmethod
    def from_args(cls) -> "Config":
        # Read a directory of attribution-score JSON files and one output path.
        parser = ArgumentParser()
        parser.add_argument(
            "--input-dir",
            type=str,
            default="./output_dim_eval",
            help="Directory containing dimension attribution JSON files.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="./output_dim_eval/dimension_attribution_capacity.json",
            help="Output JSON path for helpful/harmful capacity summary.",
        )
        args = parser.parse_args()
        return cls(input_dir=Path(args.input_dir), output_path=Path(args.output))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_embedding_and_benchmark(path: Path, metadata: dict[str, Any]) -> tuple[str, str]:
    # Prefer explicit metadata, falling back to the <embedding>__<benchmark> filename.
    embedding = metadata.get("embedding")
    benchmark = metadata.get("benchmark")
    if isinstance(embedding, str) and isinstance(benchmark, str):
        return embedding, benchmark

    stem = path.stem
    if "__" in stem:
        inferred_embedding, inferred_benchmark = stem.rsplit("__", 1)
        return (
            embedding if isinstance(embedding, str) else inferred_embedding,
            benchmark if isinstance(benchmark, str) else inferred_benchmark,
        )
    return (
        embedding if isinstance(embedding, str) else stem,
        benchmark if isinstance(benchmark, str) else "unknown",
    )


def get_num_dimensions(raw: dict[str, Any]) -> int:
    metadata = raw.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("num_dimensions"), int):
        return int(metadata["num_dimensions"])
    if isinstance(raw.get("by_dimension"), dict):
        return len(raw["by_dimension"])

    dimension_ids = set()
    for key in ["helpful_dimensions", "harmful_dimensions", "neutral_dimensions"]:
        values = raw.get(key, [])
        if isinstance(values, list):
            dimension_ids.update(int(value) for value in values)
    if dimension_ids:
        return len(dimension_ids)
    raise ValueError("Cannot infer num_dimensions")


def get_labeled_count(raw: dict[str, Any], label: str) -> int:
    metadata_key = f"num_{label}"
    metadata = raw.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get(metadata_key), int):
        return int(metadata[metadata_key])

    dimensions_key = f"{label}_dimensions"
    values = raw.get(dimensions_key, [])
    if isinstance(values, list):
        return len(values)
    raise ValueError(f"Cannot infer {metadata_key}")


def summarize_file(path: Path) -> dict[str, Any] | None:
    # Convert one attribution file into compact count/ratio statistics.
    raw = load_json(path)
    if not isinstance(raw, dict):
        return None
    if "helpful_dimensions" not in raw or "harmful_dimensions" not in raw:
        return None

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    num_dimensions = get_num_dimensions(raw)
    if num_dimensions <= 0:
        raise ValueError(f"{path}: num_dimensions must be positive")

    num_helpful = get_labeled_count(raw, "helpful")
    num_harmful = get_labeled_count(raw, "harmful")
    embedding, benchmark = infer_embedding_and_benchmark(path, metadata)

    return {
        "file": str(path),
        "embedding": embedding,
        "benchmark": benchmark,
        "num_dimensions": num_dimensions,
        "num_helpful": num_helpful,
        "num_harmful": num_harmful,
        "helpful_ratio": num_helpful / num_dimensions,
        "harmful_ratio": num_harmful / num_dimensions,
    }


def find_attribution_files(input_dir: Path, output_path: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.glob("*.json")
        if path.resolve() != output_path.resolve()
    )


def min_record(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return min(records, key=lambda record: (float(record[key]), record["file"]))


def build_summary(input_dir: Path, output_path: Path) -> dict[str, Any]:
    # Aggregate per-file capacity and compute global feasible ratio limits.
    records = []
    skipped_files = []

    for path in find_attribution_files(input_dir, output_path):
        try:
            record = summarize_file(path)
        except ValueError as exc:
            skipped_files.append({"file": str(path), "reason": str(exc)})
            continue
        if record is None:
            skipped_files.append(
                {"file": str(path), "reason": "not a dimension attribution JSON"}
            )
            continue
        records.append(record)

    if not records:
        raise ValueError(f"No dimension attribution JSON files found in {input_dir}")

    helpful_limit = min_record(records, "helpful_ratio")
    harmful_limit = min_record(records, "harmful_ratio")
    common_limit = (
        helpful_limit
        if helpful_limit["helpful_ratio"] <= harmful_limit["harmful_ratio"]
        else harmful_limit
    )
    common_limit_label = (
        "helpful" if common_limit is helpful_limit else "harmful"
    )

    requested_cases = {
        "only_helpful": [0.05, 0.10, 0.20],
        "only_harmful": [0.05, 0.10, 0.20],
        "helpful_harmful_each": [0.025, 0.05, 0.10],
    }
    feasibility = {
        "only_helpful": {
            f"{ratio:g}": ratio <= helpful_limit["helpful_ratio"]
            for ratio in requested_cases["only_helpful"]
        },
        "only_harmful": {
            f"{ratio:g}": ratio <= harmful_limit["harmful_ratio"]
            for ratio in requested_cases["only_harmful"]
        },
        "helpful_harmful_each": {
            f"{ratio:g}": ratio <= min(
                helpful_limit["helpful_ratio"],
                harmful_limit["harmful_ratio"],
            )
            for ratio in requested_cases["helpful_harmful_each"]
        },
    }

    return {
        "metadata": {
            "input_dir": str(input_dir),
            "num_files": len(records),
            "skipped_files": skipped_files,
            "ratio_definition": "count / num_dimensions",
        },
        "global_limits": {
            "max_only_helpful_ratio": helpful_limit["helpful_ratio"],
            "max_only_helpful_limited_by": helpful_limit,
            "max_only_harmful_ratio": harmful_limit["harmful_ratio"],
            "max_only_harmful_limited_by": harmful_limit,
            "max_helpful_harmful_each_ratio": min(
                helpful_limit["helpful_ratio"],
                harmful_limit["harmful_ratio"],
            ),
            "max_helpful_harmful_each_limited_by_label": common_limit_label,
            "max_helpful_harmful_each_limited_by": common_limit,
        },
        "requested_case_feasibility": feasibility,
        "records": records,
    }


def main() -> None:
    config = Config.from_args()
    summary = build_summary(config.input_dir, config.output_path)

    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)

    with config.output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    limits = summary["global_limits"]
    print(
        "Saved capacity summary to "
        f"{config.output_path} "
        f"(helpful max={limits['max_only_helpful_ratio']:.6f}, "
        f"harmful max={limits['max_only_harmful_ratio']:.6f}, "
        f"helpful/harmful each max={limits['max_helpful_harmful_each_ratio']:.6f})"
    )


if __name__ == "__main__":
    main()
