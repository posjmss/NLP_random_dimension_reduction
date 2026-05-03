import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

EXCLUDED_TASK_STEMS = {"FINAL"}
NON_METRIC_KEYS = {
    "main_score",
    "scores_per_experiment",
    "hf_subset",
    "languages",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_pairs(data_dir: Path) -> list[tuple[str, str, Path, Path]]:
    pairs: list[tuple[str, str, Path, Path]] = []
    for baseline_path in sorted(data_dir.glob("*__*.baseline.json")):
        stem = baseline_path.name.removesuffix(".baseline.json")
        if "__" not in stem:
            continue
        task, model = stem.split("__", 1)
        if task in EXCLUDED_TASK_STEMS:
            continue
        per_dim_path = data_dir / f"{task}__{model}.json"
        if not per_dim_path.exists():
            print(f"Skipping {task} / {model}: per-dim file missing ({per_dim_path.name})")
            continue
        pairs.append((task, model, per_dim_path, baseline_path))
    return pairs


PREFERRED_TOP_LEVEL_KEYS = ("max",)


def find_metric_path(
    entry: dict, main_score: float, prefix: tuple[str, ...] = ()
) -> tuple[str, ...] | None:
    if not prefix:
        ordered_keys = [k for k in PREFERRED_TOP_LEVEL_KEYS if k in entry] + [
            k for k in entry if k not in PREFERRED_TOP_LEVEL_KEYS
        ]
    else:
        ordered_keys = list(entry.keys())

    for key in ordered_keys:
        value = entry[key]
        if not prefix and key in NON_METRIC_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            if abs(float(value) - main_score) < 1e-9:
                return (*prefix, key)
        elif isinstance(value, dict):
            found = find_metric_path(value, main_score, (*prefix, key))
            if found is not None:
                return found
    return None


def extract_baseline(
    baseline_data: dict,
) -> tuple[str, str, tuple[str, ...], float]:
    scores = baseline_data.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Baseline JSON missing 'scores' object")

    for split, entries in scores.items():
        if not isinstance(entries, list) or not entries:
            continue
        entry = entries[0]
        if not isinstance(entry, dict) or "main_score" not in entry:
            continue
        main_score = float(entry["main_score"])
        subset = str(entry.get("hf_subset", "default"))
        metric_path = find_metric_path(entry, main_score)
        if metric_path is None:
            raise ValueError(
                f"Could not match main_score={main_score} to a named metric in split {split!r}"
            )
        return split, subset, metric_path, main_score

    raise ValueError("No populated split with main_score found in baseline")


def navigate(value: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def get_dimension_scores(
    per_dim_data: dict, split: str, subset: str, metric_path: tuple[str, ...]
) -> tuple[list[int], list[float]]:
    points: list[tuple[int, float]] = []
    for dim_key, by_split in per_dim_data.items():
        try:
            dim = int(dim_key)
        except ValueError:
            continue
        if not isinstance(by_split, dict):
            continue
        sub_dict = by_split.get(split)
        if not isinstance(sub_dict, dict):
            continue
        metric_dict = sub_dict.get(subset)
        if not isinstance(metric_dict, dict):
            continue
        value = navigate(metric_dict, metric_path)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        points.append((dim, float(value)))
    points.sort()
    return [d for d, _ in points], [s for _, s in points]


def split_points(
    dimensions: list[int], scores: list[float], baseline: float, epsilon: float
) -> dict[str, tuple[list[int], list[float]]]:
    groups: dict[str, tuple[list[int], list[float]]] = {
        "degrading": ([], []),
        "improving": ([], []),
        "tie": ([], []),
    }
    for dim, score in zip(dimensions, scores, strict=True):
        if score > baseline + epsilon:
            label = "degrading"
        elif score < baseline - epsilon:
            label = "improving"
        else:
            label = "tie"
        groups[label][0].append(dim)
        groups[label][1].append(score)
    return groups


def plot_pair(
    task: str,
    model: str,
    metric_label: str,
    dimensions: list[int],
    scores: list[float],
    baseline: float,
    output_path: Path,
    epsilon: float,
) -> None:
    groups = split_points(dimensions, scores, baseline, epsilon)

    plt.figure(figsize=(12, 6))
    plt.scatter(*groups["degrading"], s=14, color="tab:blue", label="degrading (drop helped)")
    plt.scatter(*groups["improving"], s=14, color="tab:orange", label="improving (drop hurt)")
    plt.scatter(*groups["tie"], s=14, color="tab:green", label="tie")
    plt.axhline(baseline, color="red", linewidth=1.5, label=f"baseline ({baseline:.4f})")
    plt.title(f"{task} - {model}")
    plt.xlabel("Removed dimension index")
    plt.ylabel(metric_label)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./outputs/one_dim_drop_finmteb",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./figures/dim_att_analysis_finmteb_per_task",
    )
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--model", type=str, default=None, help="Filter to one model.")
    parser.add_argument("--task", type=str, default=None, help="Filter to one task.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    pairs = discover_pairs(data_dir)
    if args.model:
        pairs = [p for p in pairs if p[1] == args.model]
    if args.task:
        pairs = [p for p in pairs if p[0] == args.task]

    if not pairs:
        raise FileNotFoundError(
            f"No (task, model) pairs with both per-dim + baseline JSONs in {data_dir}"
        )

    print(f"Plotting {len(pairs)} (task, model) pairs into {output_dir}")
    for task, model, per_dim_path, baseline_path in pairs:
        try:
            split, subset, metric_path, baseline = extract_baseline(load_json(baseline_path))
        except (ValueError, KeyError) as exc:
            print(f"  ! {task} / {model}: baseline parse failed - {exc}")
            continue

        metric_label = ".".join(metric_path)
        per_dim_data = load_json(per_dim_path)
        dimensions, scores = get_dimension_scores(per_dim_data, split, subset, metric_path)
        if not dimensions:
            print(
                f"  ! {task} / {model}: no per-dim scores for "
                f"split={split!r} subset={subset!r} metric={metric_label!r}"
            )
            continue

        output_path = output_dir / model / f"{task}.png"
        plot_pair(
            task=task,
            model=model,
            metric_label=metric_label,
            dimensions=dimensions,
            scores=scores,
            baseline=baseline,
            output_path=output_path,
            epsilon=args.epsilon,
        )
        print(
            f"  - {task} / {model}: {len(dimensions)} dims, metric={metric_label!r}, "
            f"baseline={baseline:.4f} -> {output_path.relative_to(output_dir.parent)}"
        )


if __name__ == "__main__":
    main()
