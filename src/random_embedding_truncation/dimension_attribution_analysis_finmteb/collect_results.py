import json
import os
import tomllib
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from random_embedding_truncation.dimension_attribution_analysis_finmteb.select_tasks import (
    DEFAULT_SEED,
    select_tasks,
)

DEFAULT_TASK_LIST = list(select_tasks(DEFAULT_SEED).values())


def read_toml(toml_file: str) -> dict[str, Any]:
    if not os.path.isfile(toml_file):
        raise FileNotFoundError(f"Not Found: {toml_file}")
    with open(toml_file, "rb") as f:
        return tomllib.load(f)


@dataclass
class Config:
    model_name: str
    output_name: str
    result_output_dir: Path
    output_path: Path
    task_list: list[str]
    start_index: int | None
    end_index: int | None

    @classmethod
    def from_args(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Path for the collected JSON file. Defaults to ./output_summary/<config>__finmteb.json.",
        )
        args = parser.parse_args()

        config = read_toml(args.config)
        model_name = config["model_name"]
        output_name = str(config.get("output_name", Path(args.config).stem))
        result_output_dir = Path(config["result_output_dir"])
        output_path = (
            Path(args.output)
            if args.output
            else Path("./output_summary") / f"{output_name}__finmteb.json"
        )

        return cls(
            model_name=model_name,
            output_name=output_name,
            result_output_dir=result_output_dir,
            output_path=output_path,
            task_list=config.get("task_list", DEFAULT_TASK_LIST),
            start_index=config.get("start_index"),
            end_index=config.get("end_index"),
        )


def find_score_jsons(task_output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in task_output_dir.rglob("*.json")
        if not path.name.startswith(".")
        and "metadata" not in path.name.lower()
        and "model_meta" not in path.name.lower()
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}

    if isinstance(value, bool):
        return metrics
    if isinstance(value, int | float):
        metrics[prefix] = float(value)
        return metrics
    if isinstance(value, dict):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            metrics.update(flatten_numeric_metrics(nested_value, next_prefix))
        return metrics
    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            next_prefix = f"{prefix}_{index}" if prefix else str(index)
            metrics.update(flatten_numeric_metrics(nested_value, next_prefix))
        return metrics

    return metrics


def collect_task_metrics(task: str, task_output_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for json_path in find_score_jsons(task_output_dir):
        loaded = load_json(json_path)
        relative_stem = json_path.relative_to(task_output_dir).with_suffix("")
        path_parts = [
            part
            for part in relative_stem.parts
            if part not in {"no_model_name_available", "no_revision_available"}
        ]
        file_prefix = "_".join(path_parts)
        flattened = flatten_numeric_metrics(loaded)

        for metric_name, metric_value in flattened.items():
            if metric_name.startswith("metadata_"):
                continue
            key_parts = [task]
            if file_prefix not in {"", "results", "scores", task}:
                key_parts.append(file_prefix)
            key_parts.append(metric_name)
            metrics["_".join(key_parts)] = metric_value

    return metrics


def add_mean_metrics(
    dimension_metrics: dict[str, float], task_list: list[str]
) -> dict[str, float]:
    grouped_values: dict[str, list[float]] = {}

    for key, value in dimension_metrics.items():
        for task in task_list:
            task_prefix = f"{task}_"
            if key.startswith(task_prefix):
                metric_suffix = key.removeprefix(task_prefix)
                grouped_values.setdefault(metric_suffix, []).append(value)
                break

    for metric_suffix, values in grouped_values.items():
        if values:
            dimension_metrics[f"FinMTEB_mean_{metric_suffix}"] = (
                sum(values) / len(values)
            )

    return dimension_metrics


def get_present_tasks(
    dimension_metrics: dict[str, float], task_list: list[str]
) -> set[str]:
    present_tasks: set[str] = set()

    for key in dimension_metrics:
        for task in task_list:
            if key.startswith(f"{task}_"):
                present_tasks.add(task)
                break

    return present_tasks


def filter_metrics_to_tasks(
    dimension_metrics: dict[str, float], tasks_to_keep: set[str]
) -> dict[str, float]:
    kept_metrics: dict[str, float] = {}

    for key, value in dimension_metrics.items():
        if any(key.startswith(f"{task}_") for task in tasks_to_keep):
            kept_metrics[key] = value

    return kept_metrics


def parse_dimension(task: str, model_slug: str, path: Path) -> int | None:
    prefix = f"{task}_{model_slug}_"
    if not path.name.startswith(prefix):
        return None
    dim_text = path.name.removeprefix(prefix)
    return int(dim_text) if dim_text.isdigit() else None


def discover_dimensions(config: Config) -> list[int]:
    dimensions: set[int] = set()
    model_slug = config.model_name.replace("/", "-")

    for task in config.task_list:
        for path in config.result_output_dir.glob(f"{task}_{model_slug}_*"):
            if not path.is_dir():
                continue
            dimension = parse_dimension(task, model_slug, path)
            if dimension is not None:
                dimensions.add(dimension)

    if config.start_index is not None or config.end_index is not None:
        start_index = config.start_index or 0
        if config.end_index is None:
            if not dimensions:
                return []
            end_index = max(dimensions) + 1
        else:
            end_index = config.end_index
        return [
            dimension
            for dimension in range(start_index, end_index)
            if dimension in dimensions
        ]

    return sorted(dimensions)


def main() -> None:
    config = Config.from_args()
    model_slug = config.model_name.replace("/", "-")
    raw_collected: dict[str, dict[str, float]] = {}
    tasks_by_dimension: dict[str, set[str]] = {}
    dimensions = discover_dimensions(config)

    print(
        f"Discovered {len(dimensions)} dimensions in {config.result_output_dir} "
        f"for {config.model_name}"
    )
    print(f"Collecting {len(config.task_list)} tasks per available dimension...")

    for dimension_index, dimension in enumerate(dimensions, start=1):
        print(f"[{dimension_index}/{len(dimensions)}] Collecting dimension {dimension}")
        dimension_metrics: dict[str, float] = {}
        for task in config.task_list:
            task_output_dir = (
                config.result_output_dir / f"{task}_{model_slug}_{dimension}"
            )
            if not task_output_dir.exists():
                print(f"  - {task}: missing")
                continue
            task_metrics = collect_task_metrics(task, task_output_dir)
            dimension_metrics.update(task_metrics)
            print(f"  - {task}: {len(task_metrics)} metrics")

        if dimension_metrics:
            dimension_key = str(dimension)
            raw_collected[dimension_key] = dimension_metrics
            tasks_by_dimension[dimension_key] = get_present_tasks(
                dimension_metrics, config.task_list
            )
            print(
                f"  -> dimension {dimension}: "
                f"{len(dimension_metrics)} metrics from "
                f"{len(tasks_by_dimension[dimension_key])} tasks"
            )
        else:
            print(f"  -> dimension {dimension}: no metrics found")

    if tasks_by_dimension:
        common_tasks = set.intersection(*tasks_by_dimension.values())
    else:
        common_tasks = set()
    print(f"Common tasks across collected dimensions: {len(common_tasks)}")

    metadata: dict[str, Any] = {
        "model_name": config.model_name,
        "output_name": config.output_name,
        "result_output_dir": str(config.result_output_dir),
        "num_dimensions": len(raw_collected),
        "configured_tasks": config.task_list,
        "common_tasks": sorted(common_tasks),
        "excluded_tasks": sorted(set(config.task_list) - common_tasks),
        "mean_metric_scope": "common_tasks_only",
    }
    collected: dict[str, dict[str, float]] = {}

    for dimension_key, dimension_metrics in raw_collected.items():
        filtered_metrics = filter_metrics_to_tasks(dimension_metrics, common_tasks)
        if filtered_metrics:
            add_mean_metrics(filtered_metrics, sorted(common_tasks))
            collected[dimension_key] = filtered_metrics

    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)

    with config.output_path.open("w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2, sort_keys=True)
    metadata_path = config.output_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(
        f"Saved {len(collected)} dimensions "
        f"using {len(common_tasks)} common tasks to {config.output_path}"
    )
    print(f"Saved collection metadata to {metadata_path}")


if __name__ == "__main__":
    main()
