"""Print final pairwise dimension-reduction results as console tables.

This file was added to inspect final_outputs quickly without opening JSON files
or generating plots.
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any


MODE_CONFIGS = {
    "beir-ndcg5": {
        "benchmark": "beir",
        "pairwise_dir": Path("beir") / "pairwise_dimension_reduction_beir_ndcg5",
        "summary_dir": Path("beir") / "output_summary",
        "baseline_suffix": "__beir.baseline.json",
        "metric": "NanoBEIR_mean_cosine_ndcg@5",
    },
    "beir-ndcg10": {
        "benchmark": "beir",
        "pairwise_dir": Path("beir") / "pairwise_dimension_reduction_beir_ndcg10",
        "summary_dir": Path("beir") / "output_summary",
        "baseline_suffix": "__beir.baseline.json",
        "metric": "NanoBEIR_mean_cosine_ndcg@10",
    },
    "mteb": {
        "benchmark": "mteb",
        "pairwise_dir": Path("mteb") / "pairwise_dimension_reduction_mteb",
        "summary_dir": Path("mteb") / "output_summary",
        "baseline_suffix": "__mteb.baseline.json",
        "metric": "MTEB_mean_scores_test_0_main_score",
    },
}

MODE_ROWS = {
    "beir-ndcg5": [
        ("imp 5%", "only_helpful_5pct"),
        ("imp 10%", "only_helpful_10pct"),
        ("imp 15%", "only_helpful_15pct"),
        ("deg 5%", "only_harmful_5pct"),
        ("deg 10%", "only_harmful_10pct"),
        ("deg 15%", "only_harmful_15pct"),
        ("mix 5%(2.5+2.5)", "helpful_harmful_2_5pct_each"),
        ("mix 10%(5+5)", "helpful_harmful_5pct_each"),
        ("mix 15%(7.5+7.5)", "helpful_harmful_7_5pct_each"),
    ],
    "beir-ndcg10": [
        ("imp 5%", "only_helpful_5pct"),
        ("imp 10%", "only_helpful_10pct"),
        ("imp 15%", "only_helpful_15pct"),
        ("deg 5%", "only_harmful_5pct"),
        ("deg 10%", "only_harmful_10pct"),
        ("deg 15%", "only_harmful_15pct"),
        ("mix 5%(2.5+2.5)", "helpful_harmful_2_5pct_each"),
        ("mix 10%(5+5)", "helpful_harmful_5pct_each"),
        ("mix 15%(7.5+7.5)", "helpful_harmful_7_5pct_each"),
    ],
    "mteb": [
        ("imp 5%", "only_helpful_5pct"),
        ("imp 10%", "only_helpful_10pct"),
        ("imp 15%", "only_helpful_15pct"),
        ("imp 18%", "only_helpful_18pct"),
        ("deg 5%", "only_harmful_5pct"),
        ("deg 10%", "only_harmful_10pct"),
        ("deg 15%", "only_harmful_15pct"),
        ("deg 18%", "only_harmful_18pct"),
        ("mix 5%(2.5+2.5)", "helpful_harmful_2_5pct_each"),
        ("mix 10%(5+5)", "helpful_harmful_5pct_each"),
        ("mix 15%(7.5+7.5)", "helpful_harmful_7_5pct_each"),
        ("mix 18%(9+9)", "helpful_harmful_9pct_each"),
    ],
}

MODELS = [
    ("MPNet", "all-mpnet-base-v2"),
    ("BERT", "bert-base"),
    ("E5-large", "e5-large-v2"),
    ("MiniLM", "paraphrase-mini-lm"),
    ("T5-base", "t5-base"),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metric_value(raw: Any, metric: str) -> float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int | float):
        return float(raw)
    if not isinstance(raw, dict):
        return None

    value = raw.get(metric)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)

    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get(metric)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)

    return None


def get_case_metric(pairwise: dict[str, Any], case_name: str, metric: str) -> float | None:
    case = pairwise.get(case_name)
    if not isinstance(case, dict):
        return None
    return get_metric_value(case, metric)


def relative_performance(
    final_outputs: Path,
    mode: str,
    model_slug: str,
) -> dict[str, float | None]:
    # Compute score / baseline as a percentage for every configured case.
    config = MODE_CONFIGS[mode]
    pairwise_path = final_outputs / config["pairwise_dir"] / f"{model_slug}.json"
    baseline_path = (
        final_outputs
        / config["summary_dir"]
        / f"{model_slug}{config['baseline_suffix']}"
    )
    metric = str(config["metric"])

    if not pairwise_path.exists() or not baseline_path.exists():
        return {case_name: None for _, case_name in MODE_ROWS[mode]}

    pairwise = load_json(pairwise_path)
    baseline = get_metric_value(load_json(baseline_path), metric)
    if not isinstance(pairwise, dict) or baseline is None or baseline == 0:
        return {case_name: None for _, case_name in MODE_ROWS[mode]}

    values: dict[str, float | None] = {}
    for _, case_name in MODE_ROWS[mode]:
        score = get_case_metric(pairwise, case_name, metric)
        values[case_name] = None if score is None else (score / baseline) * 100.0
    return values


def format_cell(value: float | None, decimals: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}%"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    # Render a simple fixed-width ASCII table without extra dependencies.
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    separator = "+".join("-" * (width + 2) for width in widths)

    def format_row(row: list[str]) -> str:
        return "|".join(
            f" {cell:<{width}} " for cell, width in zip(row, widths, strict=True)
        )

    print(f"+{separator}+")
    print(f"|{format_row(headers)}|")
    print(f"+{separator}+")
    for row in rows:
        print(f"|{format_row(row)}|")
    print(f"+{separator}+")


def build_rows(final_outputs: Path, mode: str, decimals: int) -> list[list[str]]:
    # Arrange rows by reduction case and columns by embedding model.
    values_by_model = {
        model_slug: relative_performance(final_outputs, mode, model_slug)
        for _, model_slug in MODELS
    }

    rows: list[list[str]] = []
    for label, case_name in MODE_ROWS[mode]:
        row = [label]
        for _, model_slug in MODELS:
            row.append(format_cell(values_by_model[model_slug][case_name], decimals))
        rows.append(row)
    return rows


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Print pairwise dimension reduction relative performance tables "
            "from final_outputs."
        )
    )
    parser.add_argument(
        "mode",
        choices=sorted(MODE_CONFIGS),
        help="Table to print: beir-ndcg5, beir-ndcg10, or mteb.",
    )
    parser.add_argument("--final-outputs", type=str, default="./final_outputs")
    parser.add_argument("--decimals", type=int, default=1)
    args = parser.parse_args()

    final_outputs = Path(args.final_outputs)
    if not final_outputs.exists():
        raise FileNotFoundError(f"final_outputs directory not found: {final_outputs}")

    headers = ["Reduction", *[model_name for model_name, _ in MODELS]]
    rows = build_rows(final_outputs, args.mode, args.decimals)

    print()
    print(f"Pairwise dimension reduction relative performance: {args.mode}")
    print_table(headers, rows)


if __name__ == "__main__":
    main()
