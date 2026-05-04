"""Evaluate full, untruncated embeddings on the selected MTEB tasks.

This file was added to produce baseline scores that one-dimension-drop MTEB
summaries can be compared against.
"""

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import mteb
from sentence_transformers.SentenceTransformer import SentenceTransformer

from random_embedding_truncation.truncator import Truncator
from random_embedding_truncation.dimension_attribution_analysis_mteb.collect_results import (
    add_mean_metrics,
    collect_task_metrics,
)
from random_embedding_truncation.utils import read_toml

TASK_LIST_CLASSIFICATION = [
    # "AmazonCounterfactualClassification",       # 12
    # "AmazonPolarityClassification",             # 6
    # "AmazonReviewsClassification",              # 7
    "Banking77Classification",                  # 1
    "EmotionClassification",                    # 4
    "ImdbClassification",                       # 5
    "MassiveIntentClassification",              # 2
    # "MassiveScenarioClassification",            # 10
    # "MTOPDomainClassification",                 # 11
    "MTOPIntentClassification",                 # 3
    # "ToxicConversationsClassification",         # 8
    # "TweetSentimentExtractionClassification",   # 9
]


@dataclass
class Config:
    model_name: str
    output_name: str
    cache_dir: Path
    result_output_dir: Path
    summary_output_path: Path
    batch_size: int
    task_list: list[str]

    @classmethod
    def from_config(cls) -> "Config":
        # Reuse the attribution config while allowing separate output paths.
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help=(
                "Directory for baseline task outputs. Defaults to the config "
                "result_output_dir."
            ),
        )
        parser.add_argument(
            "--summary-output",
            type=str,
            default=None,
            help=(
                "Flat baseline summary JSON path. Defaults to "
                "./output_summary/<config-name>__mteb.baseline.json."
            ),
        )
        args = parser.parse_args()
        config = read_toml(args.config)
        output_name = str(config.get("output_name", Path(args.config).stem))
        summary_output_path = (
            Path(args.summary_output)
            if args.summary_output
            else Path("./output_summary") / f"{output_name}__mteb.baseline.json"
        )

        return cls(
            model_name=config["model_name"],
            output_name=output_name,
            cache_dir=Path(config["cache_dir"]),
            result_output_dir=Path(args.output_dir)
            if args.output_dir
            else Path(config["result_output_dir"]),
            summary_output_path=summary_output_path,
            batch_size=config.get("batch_size", 32),
            task_list=config.get("task_list", TASK_LIST_CLASSIFICATION),
        )

    @property
    def is_e5_mistral(self) -> bool:
        return "e5-mistral" in self.model_name


if __name__ == "__main__":
    config = Config.from_config()

    if not config.result_output_dir.exists():
        config.result_output_dir.mkdir(parents=True)
    if not config.summary_output_path.parent.exists():
        config.summary_output_path.parent.mkdir(parents=True)

    # Evaluate the original full embedding model once for each selected task.
    encoder = SentenceTransformer(config.model_name)
    model = Truncator(
        encoder,
        resize_scale=1.0,
        cache_dir=config.cache_dir,
        is_e5=config.is_e5_mistral,
    )
    model_slug = config.model_name.replace("/", "-")

    for task in config.task_list:
        output_folder = config.result_output_dir / f"{task}_{model_slug}_baseline"
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = mteb.MTEB(tasks=[task], task_langs=["en"])
        evaluation.run(
            model,
            output_folder=str(output_folder),
            eval_splits=eval_splits,
            encode_kwargs={"batch_size": config.batch_size},
        )
        print(f"Saved full-embedding MTEB baseline for {task} to {output_folder}")

    # Flatten the raw task folders into the same metric layout as attribution.
    summary: dict[str, float] = {}
    for task in config.task_list:
        output_folder = config.result_output_dir / f"{task}_{model_slug}_baseline"
        if output_folder.exists():
            summary.update(collect_task_metrics(task, output_folder))
    add_mean_metrics(summary, config.task_list)

    with config.summary_output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Saved flat MTEB baseline summary to {config.summary_output_path}")
