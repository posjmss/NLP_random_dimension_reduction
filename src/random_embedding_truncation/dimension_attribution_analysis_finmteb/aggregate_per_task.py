import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

EXCLUDED_TASK_STEMS = {"FINAL"}
NON_METRIC_KEYS = {"main_score", "scores_per_experiment", "hf_subset", "languages"}
PREFERRED_TOP_LEVEL_KEYS = ("max",)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def per_dim_main_scores(
    per_dim_data: dict, split: str, subset: str, metric_path: tuple[str, ...]
) -> dict[int, float]:
    out: dict[int, float] = {}
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
        out[dim] = float(value)
    return out


def discover_per_model_files(
    data_dir: Path,
) -> dict[str, list[tuple[str, Path, Path]]]:
    by_model: dict[str, list[tuple[str, Path, Path]]] = defaultdict(list)
    for baseline_path in sorted(data_dir.glob("*__*.baseline.json")):
        stem = baseline_path.name.removesuffix(".baseline.json")
        if "__" not in stem:
            continue
        task, model = stem.split("__", 1)
        if task in EXCLUDED_TASK_STEMS:
            continue
        per_dim_path = data_dir / f"{task}__{model}.json"
        if not per_dim_path.exists():
            print(f"  - skip {task} / {model}: per-dim file missing")
            continue
        by_model[model].append((task, per_dim_path, baseline_path))
    return by_model


def aggregate_model(
    model: str, tasks: list[tuple[str, Path, Path]]
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
    per_task_dim_scores: dict[str, dict[int, float]] = {}
    per_task_split: dict[str, str] = {}
    per_task_baseline: dict[str, float] = {}
    per_task_metric: dict[str, str] = {}
    skipped: list[tuple[str, str]] = []

    for task, per_dim_path, baseline_path in sorted(tasks):
        try:
            split, subset, metric_path, baseline_score = extract_baseline(
                load_json(baseline_path)
            )
        except (ValueError, KeyError) as exc:
            print(f"    ! {task}: baseline parse failed - {exc}")
            skipped.append((task, f"baseline parse: {exc}"))
            continue
        dim_scores = per_dim_main_scores(
            load_json(per_dim_path), split, subset, metric_path
        )
        if not dim_scores:
            print(f"    ! {task}: no per-dim scores extracted")
            skipped.append((task, "no per-dim scores"))
            continue
        per_task_dim_scores[task] = dim_scores
        per_task_split[task] = split
        per_task_baseline[task] = baseline_score
        per_task_metric[task] = ".".join(metric_path)

    all_dims: set[int] = set()
    for dim_scores in per_task_dim_scores.values():
        all_dims.update(dim_scores.keys())

    per_dim_output: dict[str, dict[str, float]] = {}
    for dim in sorted(all_dims):
        flat: dict[str, float] = {}
        scores_by_split: dict[str, list[float]] = defaultdict(list)
        for task, dim_scores in per_task_dim_scores.items():
            if dim not in dim_scores:
                continue
            split = per_task_split[task]
            score = dim_scores[dim]
            flat[f"{task}_scores_{split}_0_main_score"] = score
            scores_by_split[split].append(score)
        for split, values in scores_by_split.items():
            flat[f"FinMTEB_mean_scores_{split}_0_main_score"] = sum(values) / len(values)
        per_dim_output[str(dim)] = flat

    baseline_output: dict[str, float] = {}
    baseline_by_split: dict[str, list[float]] = defaultdict(list)
    for task, baseline_score in per_task_baseline.items():
        split = per_task_split[task]
        baseline_output[f"{task}_scores_{split}_0_main_score"] = baseline_score
        baseline_by_split[split].append(baseline_score)
    for split, values in baseline_by_split.items():
        baseline_output[f"FinMTEB_mean_scores_{split}_0_main_score"] = sum(values) / len(
            values
        )

    metadata = {
        "model": model,
        "num_tasks_included": len(per_task_dim_scores),
        "tasks": sorted(per_task_dim_scores.keys()),
        "task_split": per_task_split,
        "task_metric_path": per_task_metric,
        "task_baseline": per_task_baseline,
        "task_dim_count": {t: len(s) for t, s in per_task_dim_scores.items()},
        "num_dimensions": len(per_dim_output),
        "skipped": skipped,
    }
    return per_dim_output, baseline_output, metadata


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./outputs/one_dim_drop_finmteb")
    parser.add_argument("--output-dir", type=str, default="./output_summary")
    parser.add_argument("--model", type=str, default=None, help="Aggregate one model only.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_model = discover_per_model_files(data_dir)
    if args.model:
        by_model = {k: v for k, v in by_model.items() if k == args.model}

    if not by_model:
        raise FileNotFoundError(f"No (task, model) pairs with baselines found in {data_dir}")

    print(f"Aggregating {len(by_model)} model(s) into {output_dir}")
    for model, tasks in sorted(by_model.items()):
        print(f"\n[{model}] {len(tasks)} task pair(s)")
        per_dim_output, baseline_output, metadata = aggregate_model(model, tasks)
        if not per_dim_output:
            print("  -> nothing to write")
            continue

        per_dim_path = output_dir / f"{model}__finmteb.json"
        baseline_path = output_dir / f"{model}__finmteb.baseline.json"
        metadata_path = output_dir / f"{model}__finmteb.metadata.json"

        with per_dim_path.open("w", encoding="utf-8") as f:
            json.dump(per_dim_output, f, indent=2, sort_keys=True)
        with baseline_path.open("w", encoding="utf-8") as f:
            json.dump(baseline_output, f, indent=2, sort_keys=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print(
            f"  -> {len(per_dim_output)} dims across "
            f"{metadata['num_tasks_included']} task(s) -> "
            f"{per_dim_path.name}, {baseline_path.name}, {metadata_path.name}"
        )


if __name__ == "__main__":
    main()
