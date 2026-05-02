import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from random_embedding_truncation.dimension_attribution_analysis_mteb.collect_results import (
    TASK_LIST_CLASSIFICATION,
    add_mean_metrics,
    filter_metrics_to_tasks,
    flatten_numeric_metrics,
    get_present_tasks,
    read_toml,
)


DEFAULT_INPUT_DIR = Path("./outputs/temp")
PACKAGE_INPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "temp"

@dataclass
class Config:
    model_name: str
    output_name: str
    input_dir: Path
    output_path: Path
    task_list: list[str]
    overwrite: bool

    @classmethod
    def from_args(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument(
            "--input-dir",
            type=str,
            default=None,
            help=(
                "Directory containing task-level JSON files from main_param_mteb.py. "
                "Defaults to ./outputs/temp, or the package-local outputs/temp if "
                "./outputs/temp does not exist."
            ),
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help=(
                "Path for the converted JSON file. Defaults to "
                "./output_summary/<config>__mteb.json, matching collect_results.py."
            ),
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Allow overwriting an existing output_summary JSON file.",
        )
        args = parser.parse_args()

        config = read_toml(args.config)
        output_name = str(config.get("output_name", Path(args.config).stem))
        input_dir = Path(args.input_dir) if args.input_dir else default_input_dir()
        task_list = config.get("task_list")
        if task_list is None:
            task_list = discover_temp_tasks(input_dir, config["model_name"])
        if not task_list:
            task_list = TASK_LIST_CLASSIFICATION
        output_path = (
            Path(args.output)
            if args.output
            else Path("./output_summary") / f"{output_name}__mteb.json"
        )

        return cls(
            model_name=config["model_name"],
            output_name=output_name,
            input_dir=input_dir,
            output_path=output_path,
            task_list=task_list,
            overwrite=args.overwrite,
        )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def default_input_dir() -> Path:
    if DEFAULT_INPUT_DIR.exists():
        return DEFAULT_INPUT_DIR
    return PACKAGE_INPUT_DIR


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "-")


def model_slugs(model_name: str) -> list[str]:
    full_slug = model_slug(model_name)
    short_slug = model_name.rstrip("/").split("/")[-1]
    return [full_slug] if short_slug == full_slug else [full_slug, short_slug]


def discover_temp_tasks(input_dir: Path, model_name: str) -> list[str]:
    discovered: set[str] = set()

    if not input_dir.exists():
        return []

    for slug in model_slugs(model_name):
        for path in input_dir.glob(f"*__{slug}.json"):
            discovered.add(path.name.removesuffix(f"__{slug}.json"))
        for path in input_dir.glob(f"*_{slug}.json"):
            if path.name.endswith(f"__{slug}.json"):
                continue
            discovered.add(path.name.removesuffix(f"_{slug}.json"))

    task_order = {task: index for index, task in enumerate(TASK_LIST_CLASSIFICATION)}
    return sorted(discovered, key=lambda task: (task_order.get(task, len(task_order)), task))


def task_temp_path_candidates(input_dir: Path, task: str, model_name: str) -> list[Path]:
    return [
        input_dir / f"{task}{separator}{slug}.json"
        for slug in model_slugs(model_name)
        for separator in ("__", "_")
    ]


def find_task_temp_path(input_dir: Path, task: str, model_name: str) -> Path | None:
    for candidate in task_temp_path_candidates(input_dir, task, model_name):
        if candidate.exists():
            return candidate
    return None


def select_primary_score_mapping(split_result: Any) -> Any:
    if not isinstance(split_result, dict):
        return split_result

    preferred_keys = ("default", "en")
    for key in preferred_keys:
        value = split_result.get(key)
        if isinstance(value, dict):
            return value

    nested_score_keys = [
        key for key, value in split_result.items() if isinstance(value, dict)
    ]
    if nested_score_keys and all(
        isinstance(split_result[key], dict) for key in nested_score_keys
    ):
        return split_result[sorted(nested_score_keys)[0]]

    return split_result


def normalize_split_scores(split_result: Any) -> list[Any]:
    if isinstance(split_result, list):
        return [select_primary_score_mapping(item) for item in split_result]
    return [select_primary_score_mapping(split_result)]


def normalize_temp_task_result(raw_result: Any) -> dict[str, Any]:
    if not isinstance(raw_result, dict):
        raise ValueError("Each dimension result must be a JSON object")

    raw_scores = raw_result.get("scores") if "scores" in raw_result else raw_result
    scores: dict[str, list[Any]] = {}
    for split, split_result in raw_scores.items():
        if split == "metadata":
            continue
        if not isinstance(split_result, dict | list):
            continue
        scores[str(split)] = normalize_split_scores(split_result)

    if not scores:
        raise ValueError("No split metrics found in dimension result")
    return {"scores": scores}


def convert_task_metrics(task: str, raw_result: Any) -> dict[str, float]:
    normalized = normalize_temp_task_result(raw_result)
    flattened = flatten_numeric_metrics(normalized)
    return {
        f"{task}_{metric_name}": metric_value
        for metric_name, metric_value in flattened.items()
        if not metric_name.startswith("metadata_")
    }


def load_task_file(path: Path) -> dict[int, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level object in {path}")

    results: dict[int, Any] = {}
    for dimension_key, value in raw.items():
        try:
            dimension = int(dimension_key)
        except ValueError:
            continue
        results[dimension] = value
    return results


def main() -> None:
    config = Config.from_args()

    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"{config.output_path} already exists. Pass --overwrite to replace it."
        )

    task_results: dict[str, dict[int, Any]] = {}
    loaded_files: dict[str, str] = {}
    missing_tasks: list[str] = []

    for task in config.task_list:
        path = find_task_temp_path(config.input_dir, task, config.model_name)
        if path is None:
            missing_tasks.append(task)
            continue
        task_results[task] = load_task_file(path)
        loaded_files[task] = str(path)
        print(f"{task}: loaded {len(task_results[task])} dimensions from {path}")

    if not task_results:
        raise FileNotFoundError(
            "No temp result files found for "
            f"{config.model_name} in {config.input_dir}"
        )

    dimensions = sorted(
        {
            dimension
            for results_by_dimension in task_results.values()
            for dimension in results_by_dimension
        }
    )
    raw_collected: dict[str, dict[str, float]] = {}
    tasks_by_dimension: dict[str, set[str]] = {}

    for dimension in dimensions:
        dimension_metrics: dict[str, float] = {}
        for task, results_by_dimension in task_results.items():
            if dimension not in results_by_dimension:
                continue
            dimension_metrics.update(
                convert_task_metrics(task, results_by_dimension[dimension])
            )

        if dimension_metrics:
            dimension_key = str(dimension)
            raw_collected[dimension_key] = dimension_metrics
            tasks_by_dimension[dimension_key] = get_present_tasks(
                dimension_metrics, config.task_list
            )

    if tasks_by_dimension:
        common_tasks = set.intersection(*tasks_by_dimension.values())
    else:
        common_tasks = set()

    collected: dict[str, dict[str, float]] = {}
    for dimension_key, dimension_metrics in raw_collected.items():
        filtered_metrics = filter_metrics_to_tasks(dimension_metrics, common_tasks)
        if filtered_metrics:
            add_mean_metrics(filtered_metrics, sorted(common_tasks))
            collected[dimension_key] = filtered_metrics

    metadata: dict[str, Any] = {
        "model_name": config.model_name,
        "output_name": config.output_name,
        "input_dir": str(config.input_dir),
        "num_dimensions": len(collected),
        "configured_tasks": config.task_list,
        "converted_tasks": sorted(task_results.keys()),
        "loaded_files": loaded_files,
        "missing_tasks": missing_tasks,
        "common_tasks": sorted(common_tasks),
        "excluded_tasks": sorted(set(config.task_list) - common_tasks),
        "mean_metric_scope": "common_tasks_only",
        "source_format": "task_dimension_json",
    }

    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)

    with config.output_path.open("w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2, sort_keys=True)

    metadata_path = config.output_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(
        f"Saved {len(collected)} dimensions using {len(common_tasks)} common tasks "
        f"to {config.output_path}"
    )
    print(f"Saved conversion metadata to {metadata_path}")


if __name__ == "__main__":
    main()
