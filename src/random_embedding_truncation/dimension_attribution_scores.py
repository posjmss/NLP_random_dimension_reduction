"""Score individual dimensions as helpful, harmful, or neutral.

This file was added to turn collected one-dimension-drop summaries into ranked
dimension attribution lists for later capacity and pairwise experiments.
"""

import json
import math
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    input_path: Path | None
    summary_dir: Path
    embedding: str
    benchmark: str
    metric: str
    output_dir: Path
    output_path: Path | None
    baseline_path: Path | None
    baseline_score: float | None
    epsilon: float
    higher_is_better: bool

    @classmethod
    def from_args(cls) -> "Config":
        # Configure which model, benchmark, metric, and baseline define attribution.
        parser = ArgumentParser()
        parser.add_argument(
            "--input",
            type=str,
            default=None,
            help=(
                "Collected one-dimension-drop summary JSON. Defaults to "
                "./output_summary/<embedding>__<benchmark>.json."
            ),
        )
        parser.add_argument(
            "--summary-dir",
            type=str,
            default="./output_summary",
            help="Directory containing collected summary JSON files.",
        )
        parser.add_argument("--embedding", type=str, required=True)
        parser.add_argument("--benchmark", choices=["beir", "mteb"], required=True)
        parser.add_argument("--metric", type=str, required=True)
        parser.add_argument(
            "--output-dir",
            type=str,
            default="./output_dim_eval",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Optional explicit output JSON path.",
        )
        parser.add_argument(
            "--baseline",
            type=str,
            default=None,
            help=(
                "Full-embedding baseline JSON. Defaults to "
                "./output_summary/<embedding>__<benchmark>.baseline.json "
                "when --baseline-score is not provided."
            ),
        )
        parser.add_argument(
            "--baseline-score",
            type=float,
            default=None,
            help="Full-embedding metric score. Overrides --baseline.",
        )
        parser.add_argument(
            "--epsilon",
            type=float,
            default=0.0,
            help="Absolute attribution score threshold for neutral dimensions.",
        )
        parser.add_argument(
            "--lower-is-better",
            action="store_true",
            help="Use for metrics where lower is better. Most metrics here are higher-is-better.",
        )
        args = parser.parse_args()

        return cls(
            input_path=Path(args.input) if args.input else None,
            summary_dir=Path(args.summary_dir),
            embedding=args.embedding,
            benchmark=args.benchmark,
            metric=args.metric,
            output_dir=Path(args.output_dir),
            output_path=Path(args.output) if args.output else None,
            baseline_path=Path(args.baseline) if args.baseline else None,
            baseline_score=args.baseline_score,
            epsilon=args.epsilon,
            higher_is_better=not args.lower_is_better,
        )


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._@+-]+", "-", value.strip())
    return slug.strip("-") or "unknown"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_input_path(config: Config) -> Path:
    # Default to the collected summary path for the selected embedding/benchmark.
    if config.input_path is not None:
        return config.input_path
    return config.summary_dir / f"{config.embedding}__{config.benchmark}.json"


def parse_dimension_key(key: str) -> int:
    try:
        return int(key)
    except ValueError as exc:
        raise ValueError(f"Dimension key must be an integer string, got {key!r}") from exc


def load_dimension_scores(input_path: Path, metric: str) -> dict[int, float]:
    raw = load_json(input_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level object in {input_path}")
    if "dimensions" in raw and isinstance(raw["dimensions"], dict):
        raw = raw["dimensions"]

    dimension_scores: dict[int, float] = {}
    missing_metric: list[str] = []

    for dimension_key, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue
        if metric not in metrics:
            missing_metric.append(str(dimension_key))
            continue
        score = metrics[metric]
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError(
                f"Metric {metric!r} for dimension {dimension_key!r} is not numeric"
            )
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(
                f"Metric {metric!r} for dimension {dimension_key!r} is not finite"
            )
        dimension_scores[parse_dimension_key(str(dimension_key))] = score

    if not dimension_scores:
        sample_keys = []
        for metrics in raw.values():
            if isinstance(metrics, dict):
                sample_keys = sorted(str(key) for key in metrics.keys())[:20]
                break
        raise ValueError(
            f"No dimension contains metric {metric!r}. "
            f"Sample available metric keys: {sample_keys}"
        )

    return dimension_scores


def default_baseline_path(config: Config) -> Path:
    return (
        config.summary_dir
        / f"{config.embedding}__{config.benchmark}.baseline.json"
    )


def extract_baseline_score(raw: Any, metric: str, baseline_path: Path) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"Baseline score in {baseline_path} must be numeric")
    if isinstance(raw, int | float):
        score = float(raw)
    elif isinstance(raw, dict):
        candidate = None
        if metric in raw:
            candidate = raw[metric]
        elif isinstance(raw.get("metrics"), dict) and metric in raw["metrics"]:
            candidate = raw["metrics"][metric]
        elif (
            isinstance(raw.get("baseline_scores"), dict)
            and metric in raw["baseline_scores"]
        ):
            candidate = raw["baseline_scores"][metric]
        elif isinstance(raw.get("scores"), dict) and metric in raw["scores"]:
            candidate = raw["scores"][metric]

        if isinstance(candidate, bool) or not isinstance(candidate, int | float):
            available_keys = sorted(str(key) for key in raw.keys())
            raise ValueError(
                f"Baseline file {baseline_path} does not contain numeric metric "
                f"{metric!r}. Top-level keys: {available_keys}"
            )
        score = float(candidate)
    else:
        raise ValueError(
            f"Baseline file {baseline_path} must contain a number or JSON object"
        )

    if not math.isfinite(score):
        raise ValueError(f"Baseline score for metric {metric!r} is not finite")
    return score


def get_reference_score(config: Config) -> tuple[float, str, str | None]:
    # Accept either a numeric baseline score or a JSON file containing the metric.
    if config.baseline_score is not None:
        if not math.isfinite(config.baseline_score):
            raise ValueError("--baseline-score must be finite")
        return config.baseline_score, "baseline_score", None

    baseline_path = config.baseline_path or default_baseline_path(config)
    if not baseline_path.exists():
        raise FileNotFoundError(
            "Full-embedding baseline is required. Pass --baseline-score or create/pass "
            f"a baseline JSON at {baseline_path}."
        )
    return (
        extract_baseline_score(load_json(baseline_path), config.metric, baseline_path),
        "baseline_file",
        str(baseline_path),
    )


def classify_dimension(attribution_score: float, epsilon: float) -> str:
    if attribution_score > epsilon:
        return "helpful"
    if attribution_score < -epsilon:
        return "harmful"
    return "neutral"


def build_attribution(config: Config) -> dict[str, Any]:
    # Compare each dropped-dimension score against the full-embedding baseline.
    input_path = get_input_path(config)
    dimension_to_dropped_score = load_dimension_scores(input_path, config.metric)
    reference_score, reference_kind, baseline_path = get_reference_score(config)

    by_dimension: dict[str, dict[str, float | int | str]] = {}
    for dimension, dropped_score in sorted(dimension_to_dropped_score.items()):
        if config.higher_is_better:
            attribution_score = reference_score - dropped_score
        else:
            attribution_score = dropped_score - reference_score
        label = classify_dimension(attribution_score, config.epsilon)
        by_dimension[str(dimension)] = {
            "dimension": dimension,
            "dropped_score": dropped_score,
            "reference_score": reference_score,
            "attribution_score": attribution_score,
            "label": label,
        }

    ranked = sorted(
        by_dimension.values(),
        key=lambda item: (
            -float(item["attribution_score"]),
            int(item["dimension"]),
        ),
    )

    helpful_dimensions = [
        int(item["dimension"]) for item in ranked if item["label"] == "helpful"
    ]
    harmful_dimensions = [
        int(item["dimension"])
        for item in reversed(ranked)
        if item["label"] == "harmful"
    ]
    neutral_dimensions = [
        int(item["dimension"]) for item in ranked if item["label"] == "neutral"
    ]

    # Store both sorted dimension lists and per-dimension details for downstream use.
    return {
        "metadata": {
            "embedding": config.embedding,
            "benchmark": config.benchmark,
            "metric": config.metric,
            "input_path": str(input_path),
            "higher_is_better": config.higher_is_better,
            "reference_kind": reference_kind,
            "reference_score": reference_score,
            "baseline_path": baseline_path,
            "epsilon": config.epsilon,
            "num_dimensions": len(by_dimension),
            "num_helpful": len(helpful_dimensions),
            "num_harmful": len(harmful_dimensions),
            "num_neutral": len(neutral_dimensions),
            "score_definition": (
                "reference_score - dropped_score"
                if config.higher_is_better
                else "dropped_score - reference_score"
            ),
            "label_definition": (
                "helpful if attribution_score > epsilon; "
                "harmful if attribution_score < -epsilon"
            ),
        },
        "helpful_dimensions": helpful_dimensions,
        "harmful_dimensions": harmful_dimensions,
        "neutral_dimensions": neutral_dimensions,
        "ranked_dimensions": [int(item["dimension"]) for item in ranked],
        "by_dimension": by_dimension,
    }


def get_output_path(config: Config) -> Path:
    if config.output_path is not None:
        return config.output_path
    filename = "__".join(
        [
            slugify(config.embedding),
            slugify(config.benchmark),
        ]
    )
    return config.output_dir / f"{filename}.json"


def main() -> None:
    config = Config.from_args()
    attribution = build_attribution(config)
    output_path = get_output_path(config)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(attribution, f, indent=2, sort_keys=True)

    metadata = attribution["metadata"]
    print(
        "Saved "
        f"{metadata['num_dimensions']} dimensions "
        f"({metadata['num_helpful']} helpful, "
        f"{metadata['num_harmful']} harmful, "
        f"{metadata['num_neutral']} neutral) "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
