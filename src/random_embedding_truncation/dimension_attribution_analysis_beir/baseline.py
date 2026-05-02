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
    # "climatefever",   # 9
    # "dbpedia",        # 6
    # "fever",          # 13
    "fiqa2018",       # 4
    "hotpotqa",       # 3
    "msmarco",        # 1
    # "nfcorpus",       # 7
    "nq",             # 2
    # "quoraretrieval", # 11
    "scidocs",        # 5
    # "arguana",        # 10
    # "scifact",        # 8
    # "touche2020",     # 12
]

NANOBEIR_TASKS = [
    # "NanoClimateFEVER",     # 9
    # "NanoDBPedia",          # 6
    # "NanoFEVER",            # 13
    "NanoFiQA2018",         # 4
    "NanoHotpotQA",         # 3
    "NanoMSMARCO",          # 1
    # "NanoNFCorpus",         # 7
    "NanoNQ",               # 2
    # "NanoQuoraRetrieval",   # 11
    "NanoSCIDOCS",          # 5
    # "NanoArguAna",          # 10
    # "NanoSciFact",          # 8
    # "NanoTouche2020",       # 12
]


@dataclass
class Config:
    model_name: str
    output_name: str
    cache_dir: Path
    output_path: Path
    summary_output_path: Path
    use_dot_product: bool
    is_e5_mistral: bool

    @classmethod
    def from_config(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help=(
                "Baseline JSON path. Defaults to a .baseline.json file next to "
                "the config output_path."
            ),
        )
        parser.add_argument(
            "--summary-output",
            type=str,
            default=None,
            help=(
                "Flat baseline summary JSON path. Defaults to "
                "./output_summary/<config-name>__beir.baseline.json."
            ),
        )
        args = parser.parse_args()
        config = read_toml(args.config)
        output_name = str(config.get("output_name", Path(args.config).stem))
        output_path = (
            Path(args.output)
            if args.output
            else Path(config["output_path"]).with_name(
                f"{Path(config['output_path']).stem}.baseline.json"
            )
        )
        summary_output_path = (
            Path(args.summary_output)
            if args.summary_output
            else Path("./output_summary") / f"{output_name}__beir.baseline.json"
        )

        return cls(
            model_name=config["model_name"],
            output_name=output_name,
            cache_dir=Path(config["cache_dir"]),
            output_path=output_path,
            summary_output_path=summary_output_path,
            use_dot_product=config.get("use_dot_product", False),
            is_e5_mistral="e5-mistral" in config["model_name"],
        )


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


if __name__ == "__main__":
    config = Config.from_config()

    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)
    if not config.summary_output_path.parent.exists():
        config.summary_output_path.parent.mkdir(parents=True)

    encoder = SentenceTransformer(config.model_name, trust_remote_code=True)
    model = Truncator(
        encoder,
        resize_scale=1.0,
        cache_dir=config.cache_dir,
        is_e5=config.is_e5_mistral,
    )
    evaluator = NanoBEIREvaluator(
        dataset_names=DATASET_NAMES,
        ndcg_at_k=[5, 10, 20],
        score_functions={SimilarityFunction.COSINE.value: cos_sim}
        if not config.use_dot_product
        else {SimilarityFunction.DOT_PRODUCT.value: dot_score},
        query_prompts=task_name_to_instruct if config.is_e5_mistral else None,
        batch_size=8 if config.is_e5_mistral else 32,
    )
    result = evaluator(model)
    sienna.save(result, config.output_path)
    summary = add_mean_metrics(flatten_numeric_metrics(result))
    with config.summary_output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Saved full-embedding BEIR baseline to {config.output_path}")
    print(f"Saved flat BEIR baseline summary to {config.summary_output_path}")
