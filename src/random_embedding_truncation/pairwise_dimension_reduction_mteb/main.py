import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mteb
from sentence_transformers.SentenceTransformer import SentenceTransformer

from random_embedding_truncation.dimension_attribution_analysis_mteb.main import (
    TASK_LIST_CLASSIFICATION,
)
from random_embedding_truncation.dimension_attribution_analysis_mteb.collect_results import (
    add_mean_metrics,
    collect_task_metrics,
)
from random_embedding_truncation.truncator import Truncator
from random_embedding_truncation.utils import read_toml

REDUCTION_CASES = [
    ("only_helpful_1", "helpful_dimensions", 1),
    ("only_helpful_5", "helpful_dimensions", 5),
    ("only_harmful_1", "harmful_dimensions", 1),
    ("only_harmful_5", "harmful_dimensions", 5),
    ("helpful_harmful_1_1", "helpful_harmful", 1),
    ("helpful_harmful_5_5", "helpful_harmful", 5),
]


@dataclass
class Config:
    model_name: str
    output_name: str
    cache_dir: Path
    result_output_dir: Path
    output_path: Path
    attribution_path: Path
    batch_size: int
    task_list: list[str]

    @classmethod
    def from_config(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument(
            "--attribution",
            type=str,
            default=None,
            help=(
                "Dimension attribution JSON. Defaults to "
                "./output_dim_eval/<config-name>__mteb.json."
            ),
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help=(
                "Directory for pairwise reduction outputs. Defaults to the "
                "config result_output_dir."
            ),
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Grouped pairwise result JSON path. Overrides config output_path.",
        )
        args = parser.parse_args()
        config = read_toml(args.config)
        output_name = str(config.get("output_name", Path(args.config).stem))
        attribution_path = (
            Path(args.attribution)
            if args.attribution
            else Path(
                config.get(
                    "attribution_path",
                    Path("./output_dim_eval") / f"{output_name}__mteb.json",
                )
            )
        )

        return cls(
            model_name=config["model_name"],
            output_name=output_name,
            cache_dir=Path(config["cache_dir"]),
            result_output_dir=Path(args.output_dir)
            if args.output_dir
            else Path(config["result_output_dir"]),
            output_path=Path(args.output)
            if args.output
            else Path(
                config.get(
                    "output_path",
                    Path(config["result_output_dir"]) / f"{output_name}.json",
                )
            ),
            attribution_path=attribution_path,
            batch_size=config.get("batch_size", 32),
            task_list=config.get("task_list", TASK_LIST_CLASSIFICATION),
        )

    @property
    def is_e5_mistral(self) -> bool:
        return "e5-mistral" in self.model_name


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_dimensions_to_drop(
    attribution: dict[str, Any], case_kind: str, count: int
) -> list[int]:
    if case_kind == "helpful_harmful":
        dimensions = (
            attribution.get("helpful_dimensions", [])[:count]
            + attribution.get("harmful_dimensions", [])[:count]
        )
        expected_count = count * 2
    else:
        dimensions = attribution.get(case_kind, [])[:count]
        expected_count = count

    if len(dimensions) < expected_count:
        raise ValueError(f"Not enough dimensions for case {case_kind}_{count}")
    return sorted({int(dimension) for dimension in dimensions})


def make_indexes_to_keep(dim_size: int, dimensions_to_drop: list[int]) -> list[int]:
    invalid = [
        dimension
        for dimension in dimensions_to_drop
        if dimension < 0 or dimension >= dim_size
    ]
    if invalid:
        raise ValueError(
            f"Dimensions out of range for embedding size {dim_size}: {invalid}"
        )
    dimensions_to_drop_set = set(dimensions_to_drop)
    return [
        dimension
        for dimension in range(dim_size)
        if dimension not in dimensions_to_drop_set
    ]


def find_score_jsons(task_output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in task_output_dir.rglob("*.json")
        if not path.name.startswith(".")
        and path.name != "dimension_reduction_metadata.json"
        and "metadata" not in path.name.lower()
        and "model_meta" not in path.name.lower()
    )


def load_task_output(task_output_dir: Path) -> dict[str, Any]:
    outputs: dict[str, Any] = {}

    for json_path in find_score_jsons(task_output_dir):
        relative_stem = str(json_path.relative_to(task_output_dir).with_suffix(""))
        outputs[relative_stem] = load_json(json_path)

    return outputs


def collect_case_metrics(
    result_output_dir: Path,
    model_slug: str,
    case_name: str,
    task_list: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for task in task_list:
        output_folder = result_output_dir / f"{task}_{model_slug}_{case_name}"
        if output_folder.exists():
            metrics.update(collect_task_metrics(task, output_folder))

    add_mean_metrics(metrics, task_list)
    return metrics


if __name__ == "__main__":
    config = Config.from_config()

    if not config.result_output_dir.exists():
        config.result_output_dir.mkdir(parents=True)
    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)

    attribution = load_json(config.attribution_path)
    if not isinstance(attribution, dict):
        raise ValueError(f"Expected top-level object in {config.attribution_path}")

    encoder = SentenceTransformer(config.model_name)
    dim_size = encoder.get_sentence_embedding_dimension()
    assert isinstance(dim_size, int)
    model_slug = config.model_name.replace("/", "-")
    if config.output_path.exists():
        grouped_results = load_json(config.output_path)
        if not isinstance(grouped_results, dict):
            raise ValueError(f"Expected top-level object in {config.output_path}")
    else:
        grouped_results = {}

    for case_name, case_kind, count in REDUCTION_CASES:
        dimensions_to_drop = get_dimensions_to_drop(attribution, case_kind, count)
        indexes_to_keep = make_indexes_to_keep(dim_size, dimensions_to_drop)
        print(f"{case_name}: dropping dimensions {dimensions_to_drop}")
        case_results = grouped_results.setdefault(
            case_name,
            {
                "dimensions_to_drop": dimensions_to_drop,
                "num_dropped_dimensions": len(dimensions_to_drop),
                "metrics": {},
                "tasks": {},
            },
        )
        case_results["dimensions_to_drop"] = dimensions_to_drop
        case_results["num_dropped_dimensions"] = len(dimensions_to_drop)

        model = Truncator(
            encoder,
            resize_scale=1.0,
            cache_dir=config.cache_dir,
            indexes_to_keep=indexes_to_keep,
            is_e5=config.is_e5_mistral,
        )

        for task in config.task_list:
            output_folder = (
                config.result_output_dir
                / f"{task}_{model_slug}_{case_name}"
            )
            metadata_path = output_folder / "dimension_reduction_metadata.json"
            if metadata_path.exists():
                print(f"  - {task}: already exists, skipping")
                case_results["tasks"][task] = load_task_output(output_folder)
                continue

            eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
            evaluation = mteb.MTEB(tasks=[task], task_langs=["en"])
            evaluation.run(
                model,
                output_folder=str(output_folder),
                eval_splits=eval_splits,
                encode_kwargs={"batch_size": config.batch_size},
            )
            if not output_folder.exists():
                output_folder.mkdir(parents=True)
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "case_name": case_name,
                        "dimensions_to_drop": dimensions_to_drop,
                        "num_dropped_dimensions": len(dimensions_to_drop),
                        "attribution_path": str(config.attribution_path),
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            case_results["tasks"][task] = load_task_output(output_folder)
            case_results["metrics"] = collect_case_metrics(
                config.result_output_dir,
                model_slug,
                case_name,
                config.task_list,
            )
            with config.output_path.open("w", encoding="utf-8") as f:
                json.dump(grouped_results, f, indent=2, sort_keys=True)
            print(f"  - {task}: saved to {output_folder}")

        case_results["metrics"] = collect_case_metrics(
            config.result_output_dir,
            model_slug,
            case_name,
            config.task_list,
        )
        with config.output_path.open("w", encoding="utf-8") as f:
            json.dump(grouped_results, f, indent=2, sort_keys=True)
        print(f"{case_name}: grouped result saved to {config.output_path}")
