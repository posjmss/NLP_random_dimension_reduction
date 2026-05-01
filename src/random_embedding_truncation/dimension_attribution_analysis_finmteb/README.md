# Dimension Attribution Analysis on FinMTEB

Mirror of `dimension_attribution_analysis_mteb/` adapted to FinMTEB
(Finance Massive Text Embedding Benchmark, EMNLP 2025).

FinMTEB is vendored at `third_party/FinMTEB/` (clone of
[yixuantt/FinMTEB](https://github.com/yixuantt/FinMTEB)). Its `finance_mteb`
package exposes its own `MTEB` class with finance-specific tasks; this
pipeline imports that class via a `sys.path` shim
(`_finmteb_path.add_finmteb_to_path()`) instead of installing it.

`third_party/` is gitignored. Clone it once (pin to the commit this pipeline
was built against for reproducibility):

```sh
git clone https://github.com/yixuantt/FinMTEB.git third_party/FinMTEB
git -C third_party/FinMTEB checkout ca57d38af7e49c261a13e053f7f6afc105674b51
```

## Task selection

FinMTEB has 7 task categories. We pick **one dataset per category** with a
fixed seed for reproducibility.

```sh
PYTHONPATH=src uv run python \
  src/random_embedding_truncation/dimension_attribution_analysis_finmteb/select_tasks.py
```

With `seed=42` (the default in `select_tasks.DEFAULT_SEED`) the picks are:

| Category           | Dataset                        |
| ------------------ | ------------------------------ |
| Classification     | `FinSentClassification`        |
| Retrieval          | `FiQA2018Retrieval`            |
| Clustering         | `FinanceArxivP2PClustering`    |
| Reranking          | `FiQA2018Reranking`            |
| STS                | `FinSTS`                       |
| Summarization      | `FINDsum`                      |
| PairClassification | `HeadlineACPairClassification` |

Override per-config by adding `task_list = [...]` to the TOML.

## Run

Per-dimension attribution (drops one dimension at a time, runs all 7 tasks):

```sh
PYTHONPATH=src uv run python \
  src/random_embedding_truncation/dimension_attribution_analysis_finmteb/main.py \
  --config src/random_embedding_truncation/dimension_attribution_analysis_finmteb/configs/all-mpnet-base-v2.toml
```

Full-embedding baseline (writes `output_summary/<output_name>__finmteb.baseline.json`):

```sh
PYTHONPATH=src uv run python \
  src/random_embedding_truncation/dimension_attribution_analysis_finmteb/baseline.py \
  --config src/random_embedding_truncation/dimension_attribution_analysis_finmteb/configs/all-mpnet-base-v2.toml
```

Summary collection (writes `output_summary/<output_name>__finmteb.json`):

```sh
PYTHONPATH=src uv run python \
  src/random_embedding_truncation/dimension_attribution_analysis_finmteb/collect_results.py \
  --config src/random_embedding_truncation/dimension_attribution_analysis_finmteb/configs/all-mpnet-base-v2.toml
```

Plot:

```sh
PYTHONPATH=src uv run python \
  src/random_embedding_truncation/plotting/dim_att_analysis_finmteb/main.py \
  --model all-mpnet-base-v2
```

## Notes

- **Eval splits.** Unlike the MTEB pipeline (which forces `eval_splits=["test"]`),
  this pipeline omits the arg so each task uses its own metadata-defined split.
  `FiQA2018Retrieval` defaults to the `train` split in FinMTEB, so its scores
  land under `..._scores_train_0_main_score` while the other six tasks land
  under `..._scores_test_0_main_score`. The default plotter metric
  (`FinMTEB_mean_scores_test_0_main_score`) therefore averages 6 tasks and
  excludes FiQA Retrieval. Pass `--metric ...` to plot a different aggregate.
- **Mean key prefix.** `collect_results.add_mean_metrics` writes
  `FinMTEB_mean_<suffix>` (vs. `MTEB_mean_<suffix>` in the MTEB pipeline).
- **Vendored MTEB version.** `finance_mteb` ships its own MTEB v1.12.49 fork.
  This is independent of the `mteb` pinned in `pyproject.toml`.
