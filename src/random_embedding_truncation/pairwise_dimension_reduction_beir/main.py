import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sienna
from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import cos_sim, dot_score

from random_embedding_truncation.truncator import Truncator, task_name_to_instruct
from random_embedding_truncation.utils import read_toml

DATASET_NAMES = [
    "climatefever",   # 9
    "dbpedia",        # 6
    "fever",          # 13
    "fiqa2018",       # 4
    "hotpotqa",       # 3
    "msmarco",        # 1
    "nfcorpus",       # 7
    "nq",             # 2
    "quoraretrieval", # 11
    "scidocs",        # 5
    "arguana",        # 10
    "scifact",        # 8
    "touche2020",     # 12
]

NANOBEIR_TASKS = [
    "NanoClimateFEVER",     # 9
    "NanoDBPedia",          # 6
    "NanoFEVER",            # 13
    "NanoFiQA2018",         # 4
    "NanoHotpotQA",         # 3
    "NanoMSMARCO",          # 1
    "NanoNFCorpus",         # 7
    "NanoNQ",               # 2
    "NanoQuoraRetrieval",   # 11
    "NanoSCIDOCS",          # 5
    "NanoArguAna",          # 10
    "NanoSciFact",          # 8
    "NanoTouche2020",       # 12
]

REDUCTION_CASES = [
    ("only_helpful_2", "helpful_dimensions", 2),
    ("only_helpful_4", "helpful_dimensions", 4),
    ("only_helpful_10", "helpful_dimensions", 10),
    ("only_helpful_20", "helpful_dimensions", 20),
    ("only_helpful_40", "helpful_dimensions", 40),
    ("only_helpful_100", "helpful_dimensions", 100),
    ("only_harmful_2", "harmful_dimensions", 2),
    ("only_harmful_4", "harmful_dimensions", 4),
    ("only_harmful_10", "harmful_dimensions", 10),
    ("only_harmful_20", "harmful_dimensions", 20),
    ("only_harmful_40", "harmful_dimensions", 40),
    ("only_harmful_100", "harmful_dimensions", 100),
    ("helpful_harmful_1_1", "helpful_harmful", 1),
    ("helpful_harmful_2_2", "helpful_harmful", 2),
    ("helpful_harmful_5_5", "helpful_harmful", 5),
    ("helpful_harmful_10_10", "helpful_harmful", 10),
    ("helpful_harmful_20_20", "helpful_harmful", 20),
    ("helpful_harmful_50_50", "helpful_harmful", 50),
]


@dataclass
class Config:
    model_name: str
    output_name: str
    cache_dir: Path
    output_path: Path
    attribution_path: Path
    use_dot_product: bool
    is_e5_mistral: bool

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
                "./output_dim_eval/<config-name>__beir.json."
            ),
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help=(
                "Pairwise result JSON path. Overrides config output_path."
            ),
        )
        args = parser.parse_args()
        config = read_toml(args.config)
        output_name = str(config.get("output_name", Path(args.config).stem))
        output_path = (
            Path(args.output)
            if args.output
            else Path(config["output_path"])
        )
        attribution_path = (
            Path(args.attribution)
            if args.attribution
            else Path(
                config.get(
                    "attribution_path",
                    Path("./output_dim_eval") / f"{output_name}__beir.json",
                )
            )
        )

        return cls(
            model_name=config["model_name"],
            output_name=output_name,
            cache_dir=Path(config["cache_dir"]),
            output_path=output_path,
            attribution_path=attribution_path,
            use_dot_product=config.get("use_dot_product", False),
            is_e5_mistral="e5-mistral" in config["model_name"],
        )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_mean_metrics(metrics: dict[str, float]) -> dict[str, float]:
    grouped_values: dict[str, list[float]] = {}

    for key, value in metrics.items():
        if key.startswith("NanoBEIR_mean_"):
            continue
        for task in NANOBEIR_TASKS:
            task_prefix = f"{task}_"
            if key.startswith(task_prefix):
                metric_suffix = key.removeprefix(task_prefix)
                grouped_values.setdefault(metric_suffix, []).append(value)
                break

    for metric_suffix, values in grouped_values.items():
        if values:
            metrics[f"NanoBEIR_mean_{metric_suffix}"] = sum(values) / len(values)

    return metrics


def flatten_numeric_metrics(result: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        metrics[key] = float(value)

    return metrics


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


if __name__ == "__main__":
    config = Config.from_config()

    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)

    attribution = load_json(config.attribution_path)
    if not isinstance(attribution, dict):
        raise ValueError(f"Expected top-level object in {config.attribution_path}")

    encoder = SentenceTransformer(config.model_name, trust_remote_code=True)
    dim_size = encoder.get_sentence_embedding_dimension()
    assert isinstance(dim_size, int)

    if config.output_path.exists():
        results = sienna.load(config.output_path)
    else:
        results = {}

    for case_name, case_kind, count in REDUCTION_CASES:
        if case_name in results:
            print(f"{case_name}: already exists, skipping")
            continue

        dimensions_to_drop = get_dimensions_to_drop(attribution, case_kind, count)
        print(f"{case_name}: dropping dimensions {dimensions_to_drop}")
        model = Truncator(
            encoder,
            resize_scale=1.0,
            cache_dir=config.cache_dir,
            indexes_to_keep=make_indexes_to_keep(dim_size, dimensions_to_drop),
            is_e5=config.is_e5_mistral,
        )
        evaluator = NanoBEIREvaluator(
            dataset_names=DATASET_NAMES,
            ndcg_at_k=[5, 10],
            score_functions={SimilarityFunction.COSINE.value: cos_sim}
            if not config.use_dot_product
            else {SimilarityFunction.DOT_PRODUCT.value: dot_score},
            query_prompts=task_name_to_instruct if config.is_e5_mistral else None,
            batch_size=8 if config.is_e5_mistral else 32,
        )
        result = evaluator(model)
        metrics = add_mean_metrics(flatten_numeric_metrics(result))
        results[case_name] = {
            "dimensions_to_drop": dimensions_to_drop,
            "num_dropped_dimensions": len(dimensions_to_drop),
            "metrics": metrics,
            "raw_metrics": result,
        }
        sienna.save(results, config.output_path)
        print(f"{case_name}: saved to {config.output_path}")
