import json
import os
import tomllib
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]


def read_toml(toml_file: str) -> dict[str, Any]:
    if not os.path.isfile(toml_file):
        raise FileNotFoundError(f"Not Found: {toml_file}")
    with open(toml_file, "rb") as f:
        return tomllib.load(f)


@dataclass
class Config:
    model_name: str
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
            help="Path for the collected JSON file. Defaults to result_output_dir/<model>.json.",
        )
        args = parser.parse_args()

        config = read_toml(args.config)
        model_name = config["model_name"]
        result_output_dir = Path(config["result_output_dir"])
        model_slug = model_name.replace("/", "-")
        output_path = (
            Path(args.output)
            if args.output
            else result_output_dir / f"{model_slug}.json"
        )

        return cls(
            model_name=model_name,
            result_output_dir=result_output_dir,
            output_path=output_path,
            task_list=config.get("task_list", TASK_LIST_CLASSIFICATION),
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
        file_prefix = "_".join(relative_stem.parts)
        flattened = flatten_numeric_metrics(loaded)

        for metric_name, metric_value in flattened.items():
            if metric_name.startswith("metadata_"):
                continue
            key_parts = [task]
            if file_prefix not in {"results", "scores", task}:
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
            dimension_metrics[f"MTEB_mean_{metric_suffix}"] = sum(values) / len(values)

    return dimension_metrics


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
    collected: dict[str, dict[str, float]] = {}

    for dimension in discover_dimensions(config):
        dimension_metrics: dict[str, float] = {}
        for task in config.task_list:
            task_output_dir = (
                config.result_output_dir / f"{task}_{model_slug}_{dimension}"
            )
            if not task_output_dir.exists():
                continue
            dimension_metrics.update(collect_task_metrics(task, task_output_dir))

        if dimension_metrics:
            add_mean_metrics(dimension_metrics, config.task_list)
            collected[str(dimension)] = dimension_metrics

    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)

    with config.output_path.open("w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2, sort_keys=True)
    print(f"Saved {len(collected)} dimensions to {config.output_path}")


if __name__ == "__main__":
    main()
