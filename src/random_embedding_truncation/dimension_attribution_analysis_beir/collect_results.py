import json
import os
import tomllib
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NANOBEIR_TASKS = [
    "NanoArguAna",
    "NanoClimateFEVER",
    "NanoDBPedia",
    "NanoFEVER",
    "NanoFiQA2018",
    "NanoHotpotQA",
    "NanoMSMARCO",
    "NanoNFCorpus",
    "NanoNQ",
    "NanoQuoraRetrieval",
    "NanoSCIDOCS",
    "NanoSciFact",
    "NanoTouche2020",
]

# read configs
def read_toml(toml_file: str) -> dict[str, Any]:
    if not os.path.isfile(toml_file):
        raise FileNotFoundError(f"Not Found: {toml_file}")
    with open(toml_file, "rb") as f:
        return tomllib.load(f)


@dataclass
class Config:
    input_paths: list[Path]
    output_path: Path
    model_name: str | None
    output_name: str
    merge_inputs: bool

    @classmethod
    def from_args(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="BEIR attribution config. Used to read output_path and model_name.",
        )
        parser.add_argument(
            "--input",
            type=str,
            default=None,
            help="Raw BEIR attribution JSON. Overrides config output_path.",
        )
        parser.add_argument(
            "--no-merge-inputs",
            action="store_true",
            help=(
                "Read only the explicit/configured input file. By default, "
                "also merges sibling <output-name>__<dataset>.json files."
            ),
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Summary JSON path. Defaults to ./output_summary/<config-or-input>__beir.json.",
        )
        args = parser.parse_args()

        config: dict[str, Any] = {}
        if args.config:
            config = read_toml(args.config)

        if "output_name" in config:
            output_name = str(config["output_name"])
        elif args.config:
            output_name = Path(args.config).stem
        elif args.input:
            output_name = Path(args.input).stem.split("__")[0]
        else:
            output_name = "unknown"

        if args.input:
            primary_input_path = Path(args.input)
        elif "output_path" in config:
            primary_input_path = Path(config["output_path"])
        else:
            raise ValueError("Pass either --input or --config with output_path.")

        input_paths = [primary_input_path]
        if not args.no_merge_inputs:
            input_paths = discover_input_paths(primary_input_path, output_name)

        output_path = (
            Path(args.output)
            if args.output
            else Path("./output_summary") / f"{output_name}__beir.json"
        )

        return cls(
            input_paths=input_paths,
            output_path=output_path,
            model_name=config.get("model_name"),
            output_name=output_name,
            merge_inputs=not args.no_merge_inputs,
        )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_input_paths(primary_input_path: Path, output_name: str) -> list[Path]:
    paths: list[Path] = []

    if primary_input_path.exists():
        paths.append(primary_input_path)

    for path in sorted(primary_input_path.parent.glob(f"{output_name}__*.json")):
        if path.name.endswith(".metadata.json") or ".baseline" in path.name:
            continue
        if path not in paths:
            paths.append(path)

    if not paths:
        paths.append(primary_input_path)

    return paths


def merge_raw_results(input_paths: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for input_path in input_paths:
        raw = load_json(input_path)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected top-level object in {input_path}")

        for dimension_key, metrics in raw.items():
            if not isinstance(metrics, dict):
                continue
            dimension_metrics = merged.setdefault(str(dimension_key), {})
            for metric_key, value in metrics.items():
                if metric_key.startswith("NanoBEIR_mean_"):
                    continue
                if metric_key in dimension_metrics and dimension_metrics[metric_key] != value:
                    raise ValueError(
                        "Conflicting BEIR metric while merging inputs: "
                        f"dimension={dimension_key!r}, metric={metric_key!r}, "
                        f"existing={dimension_metrics[metric_key]!r}, "
                        f"new={value!r}, input={input_path}"
                    )
                dimension_metrics[metric_key] = value

    return merged


def get_present_tasks(
    dimension_metrics: dict[str, Any], task_list: list[str]
) -> set[str]:
    present_tasks: set[str] = set()

    for key in dimension_metrics:
        for task in task_list:
            if key.startswith(f"{task}_"):
                present_tasks.add(task)
                break

    return present_tasks


def filter_metrics_to_tasks(
    dimension_metrics: dict[str, Any], tasks_to_keep: set[str]
) -> dict[str, float]:
    kept_metrics: dict[str, float] = {}

    for key, value in dimension_metrics.items():
        if key.startswith("NanoBEIR_mean_"):
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if any(key.startswith(f"{task}_") for task in tasks_to_keep):
            kept_metrics[key] = float(value)

    return kept_metrics


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
            dimension_metrics[f"NanoBEIR_mean_{metric_suffix}"] = (
                sum(values) / len(values)
            )

    return dimension_metrics


def collect_common_dataset_results(config: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = merge_raw_results(config.input_paths)

    raw_dimensions: dict[str, dict[str, Any]] = {
        str(dimension): metrics
        for dimension, metrics in raw.items()
        if isinstance(metrics, dict)
    }
    tasks_by_dimension = {
        dimension: get_present_tasks(metrics, NANOBEIR_TASKS)
        for dimension, metrics in raw_dimensions.items()
    }

    if tasks_by_dimension:
        common_tasks = set.intersection(*tasks_by_dimension.values())
    else:
        common_tasks = set()

    collected: dict[str, dict[str, float]] = {}
    for dimension, metrics in raw_dimensions.items():
        filtered_metrics = filter_metrics_to_tasks(metrics, common_tasks)
        if filtered_metrics:
            add_mean_metrics(filtered_metrics, sorted(common_tasks))
            collected[dimension] = filtered_metrics

    metadata = {
        "model_name": config.model_name,
        "output_name": config.output_name,
        "input_paths": [str(path) for path in config.input_paths],
        "merge_inputs": config.merge_inputs,
        "num_dimensions": len(raw_dimensions),
        "configured_tasks": NANOBEIR_TASKS,
        "common_tasks": sorted(common_tasks),
        "excluded_tasks": sorted(set(NANOBEIR_TASKS) - common_tasks),
        "mean_metric_scope": "common_tasks_only",
    }

    return collected, metadata


def main() -> None:
    config = Config.from_args()
    collected, metadata = collect_common_dataset_results(config)

    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)

    with config.output_path.open("w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2, sort_keys=True)
    metadata_path = config.output_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(
        f"Saved {len(collected)} dimensions "
        f"using {len(metadata['common_tasks'])} common datasets to {config.output_path}"
    )
    print(f"Saved collection metadata to {metadata_path}")


if __name__ == "__main__":
    main()
