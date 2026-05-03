import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DEFAULT_METRIC = "MTEB_mean_scores_test_0_main_score"
PLOT_GROUPS = {
    "only_helpful": [
        ("only_helpful_5pct", "5%"),
        ("only_helpful_10pct", "10%"),
        ("only_helpful_20pct", "20%"),
    ],
    "only_harmful": [
        ("only_harmful_5pct", "5%"),
        ("only_harmful_10pct", "10%"),
        ("only_harmful_20pct", "20%"),
    ],
    "helpful_harmful": [
        ("helpful_harmful_2_5pct_each", "2.5% + 2.5%"),
        ("helpful_harmful_5pct_each", "5% + 5%"),
        ("helpful_harmful_10pct_each", "10% + 10%"),
    ],
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_pairwise_paths(result_dir: Path) -> list[Path]:
    return sorted(path for path in result_dir.glob("*.json") if path.is_file())


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


def model_name_from_result(path: Path) -> str:
    return path.stem


def case_score(pairwise: dict[str, Any], case_name: str, metric: str) -> float | None:
    case = pairwise.get(case_name)
    if not isinstance(case, dict):
        return None
    metrics = case.get("metrics")
    if not isinstance(metrics, dict):
        return None
    score = metrics.get(metric)
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return float(score)


def case_label(pairwise: dict[str, Any], case_name: str, fallback_label: str) -> str:
    case = pairwise.get(case_name)
    if not isinstance(case, dict):
        return fallback_label

    reduction_plan = case.get("reduction_plan")
    if isinstance(reduction_plan, dict):
        actual_ratio = reduction_plan.get("actual_total_ratio")
        if isinstance(actual_ratio, int | float) and not isinstance(actual_ratio, bool):
            return f"{actual_ratio * 100:.1f}%"

    dropped_count = case.get("num_dropped_dimensions")
    if isinstance(dropped_count, int) and not isinstance(dropped_count, bool):
        return f"{fallback_label}\n({dropped_count} dims)"

    return fallback_label


def plot_group(
    model_name: str,
    group_name: str,
    values: list[tuple[str, float]],
    output_path: Path,
) -> None:
    x_labels = [label for label, _ in values]
    y_values = [relative_score for _, relative_score in values]

    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, y_values, color="tab:blue")
    plt.axhline(100.0, color="red", linewidth=1.5, label="baseline")
    plt.title(f"{model_name} - {group_name}")
    plt.xlabel("Removed dimensions")
    plt.ylabel("Relative performance (%)")
    plt.ylim(min(95, min(y_values) - 1), max(105, max(y_values) + 1))
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved {model_name} {group_name} plot to {output_path}")
    plt.show()
    plt.close()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=str,
        default="./outputs/pairwise_dimension_reduction_mteb/results",
    )
    parser.add_argument("--summary-dir", type=str, default="./output_summary")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./figures/pairwise_dim_truncation_mteb",
    )
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    summary_dir = Path(args.summary_dir)
    output_dir = Path(args.output_dir)
    pairwise_paths = discover_pairwise_paths(result_dir)
    if args.model:
        pairwise_paths = [
            path for path in pairwise_paths if model_name_from_result(path) == args.model
        ]
    if not pairwise_paths:
        raise FileNotFoundError(f"No MTEB pairwise result JSON files found in {result_dir}")

    for pairwise_path in pairwise_paths:
        model_name = model_name_from_result(pairwise_path)
        baseline_path = summary_dir / f"{model_name}__mteb.baseline.json"
        if not baseline_path.exists():
            print(f"Skipping {model_name}: missing baseline {baseline_path}")
            continue

        baseline = get_baseline_score(baseline_path, args.metric)
        pairwise = load_json(pairwise_path)
        if not isinstance(pairwise, dict):
            raise ValueError(f"Expected top-level object in {pairwise_path}")

        for group_name, cases in PLOT_GROUPS.items():
            values: list[tuple[str, float]] = []
            for case_name, fallback_label in cases:
                score = case_score(pairwise, case_name, args.metric)
                if score is None:
                    print(f"Skipping {model_name} {case_name}: missing {args.metric}")
                    continue
                values.append(
                    (
                        case_label(pairwise, case_name, fallback_label),
                        (score / baseline) * 100.0,
                    )
                )
            if values:
                plot_group(
                    model_name=model_name,
                    group_name=group_name,
                    values=values,
                    output_path=output_dir / f"{model_name}__{group_name}.png",
                )


if __name__ == "__main__":
    main()
