"""Reproducibly pick one FinMTEB dataset per task category.

The candidate pools mirror FinMTEB's active (uncommented) task lists in
third_party/FinMTEB/task_list.py at the time of vendoring. We pick one per
category with a fixed seed so the selection is recorded and rerunnable.

Usage:
    python -m random_embedding_truncation.dimension_attribution_analysis_finmteb.select_tasks
    python -m random_embedding_truncation.dimension_attribution_analysis_finmteb.select_tasks --seed 0
"""

import json
import random
from argparse import ArgumentParser

DEFAULT_SEED = 42

CATEGORY_TASKS: dict[str, list[str]] = {
    "Classification": [
        "FinancialPhraseBankClassification",
        "FinSentClassification",
        "FiQAClassification",
        "SemEva2017Classification",
        "FLSClassification",
        "ESGClassification",
        "FOMCClassification",
        "FinancialFraudClassification",
    ],
    "Retrieval": [
        "FiQA2018Retrieval",
        "FinanceBenchRetrieval",
        "HC3Retrieval",
        "Apple10KRetrieval",
        "FinQARetrieval",
        "TATQARetrieval",
        "USNewsRetrieval",
        "TradeTheEventEncyclopediaRetrieval",
        "TradeTheEventNewsRetrieval",
        "TheGoldmanEnRetrieval",
    ],
    "Clustering": [
        "MInDS14EnClustering",
        "ComplaintsClustering",
        "PiiClustering",
        "FinanceArxivS2SClustering",
        "FinanceArxivP2PClustering",
        "WikiCompany2IndustryClustering",
        "MInDS14ZhClustering",
        "FinNLClustering",
        "CCKS2022Clustering",
        "CCKS2020Clustering",
        "CCKS2019Clustering",
    ],
    "Reranking": [
        "FinFactReranking",
        "FiQA2018Reranking",
        "HC3Reranking",
        "FinEvaReranking",
        "DISCFinLLMReranking",
    ],
    "STS": [
        "FINAL",
        "FinSTS",
        "AFQMC",
        "BQCorpus",
    ],
    "Summarization": [
        "Ectsum",
        "FINDsum",
        "FNS2022sum",
        "FiNNAsum",
        "FinEvaHeadlinesum",
        "FinEvasum",
    ],
    "PairClassification": [
        "HeadlineACPairClassification",
        "HeadlinePDDPairClassification",
        "HeadlinePDUPairClassification",
        "AFQMCPairClassification",
    ],
}


def select_tasks(seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    return {
        category: rng.choice(candidates)
        for category, candidates in CATEGORY_TASKS.items()
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--format",
        choices=["pretty", "toml", "json"],
        default="pretty",
        help="pretty: human-readable. toml: a task_list = [...] line. json: dict.",
    )
    args = parser.parse_args()

    selection = select_tasks(args.seed)

    if args.format == "json":
        print(json.dumps(selection, indent=2))
    elif args.format == "toml":
        tasks = list(selection.values())
        rendered = ", ".join(f'"{task}"' for task in tasks)
        print(f"task_list = [{rendered}]")
    else:
        print(f"Seed: {args.seed}")
        for category, task in selection.items():
            print(f"  {category:20s} -> {task}")
