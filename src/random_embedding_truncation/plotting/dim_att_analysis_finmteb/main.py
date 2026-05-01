import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DEFAULT_METRIC = "FinMTEB_mean_scores_test_0_main_score"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_summary_paths(summary_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in summary_dir.glob("*__finmteb.json")
        if ".baseline" not in path.name and ".metadata" not in path.name
    )


def model_name_from_summary(path: Path) -> str:
    return path.name.removesuffix("__finmteb.json")


def get_baseline_score(path: Path, metric: str) -> float:
    raw = load_json(path)

    if isinstance(raw, bool):
        raise ValueError(f"Baseline score in {path} must be numeric")
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, dict):
        if metric in raw and not isinstance(raw[metric], bool):
            return float(raw[metric])
        if isinstance(raw.get("metrics"), dict) and metric in raw["metrics"]:
            return float(raw["metrics"][metric])
        if isinstance(raw.get("scores"), dict) and metric in raw["scores"]:
            return float(raw["scores"][metric])
        raise ValueError(f"{path} does not contain metric {metric!r}")

    raise ValueError(f"Unsupported baseline JSON format in {path}")


def get_dimension_scores(path: Path, metric: str) -> tuple[list[int], list[float]]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level object in {path}")
    if "dimensions" in raw and isinstance(raw["dimensions"], dict):
        raw = raw["dimensions"]

    points: list[tuple[int, float]] = []
    for dimension_key, metrics in raw.items():
        if not isinstance(metrics, dict) or metric not in metrics:
            continue
        score = metrics[metric]
        if isinstance(score, bool) or not isinstance(score, int | float):
            continue
        points.append((int(dimension_key), float(score)))

    if not points:
        raise ValueError(f"No dimension scores found for metric {metric!r} in {path}")

    points.sort()
    return [dimension for dimension, _ in points], [score for _, score in points]


def split_points(
    dimensions: list[int], scores: list[float], baseline: float, epsilon: float
) -> dict[str, tuple[list[int], list[float]]]:
    groups = {
        "degrading": ([], []),
        "improving": ([], []),
        "tie": ([], []),
    }

    for dimension, score in zip(dimensions, scores, strict=True):
        if score > baseline + epsilon:
            label = "degrading"
        elif score < baseline - epsilon:
            label = "improving"
        else:
            label = "tie"
        groups[label][0].append(dimension)
        groups[label][1].append(score)

    return groups


def plot_model(
    model_name: str,
    dimensions: list[int],
    scores: list[float],
    baseline: float,
    metric: str,
    output_path: Path,
    epsilon: float,
) -> None:
    groups = split_points(dimensions, scores, baseline, epsilon)

    plt.figure(figsize=(12, 6))
    plt.scatter(*groups["degrading"], s=16, color="tab:blue", label="degrading")
    plt.scatter(*groups["improving"], s=16, color="tab:orange", label="improving")
    plt.scatter(*groups["tie"], s=16, color="tab:green", label="tie")
    plt.axhline(baseline, color="red", linewidth=1.5, label="baseline")
    plt.title(model_name)
    plt.xlabel("Removed dimension index")
    plt.ylabel("FinMTEB Performance")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved {metric} plot for {model_name} to {output_path}")
    plt.show()
    plt.close()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--summary-dir", type=str, default="./output_summary")
    parser.add_argument("--output-dir", type=str, default="./figures/dim_att_analysis_finmteb")
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional output_name/model stem to plot, e.g. all-mpnet-base-v2.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-12)
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    output_dir = Path(args.output_dir)
    summary_paths = discover_summary_paths(summary_dir)
    if args.model:
        summary_paths = [
            path for path in summary_paths if model_name_from_summary(path) == args.model
        ]

    if not summary_paths:
        raise FileNotFoundError(f"No FinMTEB summary JSON files found in {summary_dir}")

    for summary_path in summary_paths:
        model_name = model_name_from_summary(summary_path)
        baseline_path = summary_dir / f"{model_name}__finmteb.baseline.json"
        if not baseline_path.exists():
            print(f"Skipping {model_name}: missing baseline {baseline_path}")
            continue

        dimensions, scores = get_dimension_scores(summary_path, args.metric)
        baseline = get_baseline_score(baseline_path, args.metric)
        plot_model(
            model_name=model_name,
            dimensions=dimensions,
            scores=scores,
            baseline=baseline,
            metric=args.metric,
            output_path=output_dir / f"{model_name}__finmteb.png",
            epsilon=args.epsilon,
        )


if __name__ == "__main__":
    main()
