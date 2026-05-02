from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import sienna
from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import cos_sim, dot_score

from random_embedding_truncation.truncator import Truncator, task_name_to_instruct
from random_embedding_truncation.utils import read_toml


@dataclass
class Config:
    model_name: str
    output_name: str
    cache_dir: Path
    output_path: Path
    dataset_name: str
    use_dot_product: bool
    is_e5_mistral: bool
    start_index: int
    end_index: int | None

    @classmethod
    def from_config(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Single NanoBEIR dataset name, e.g. msmarco, nq, fiqa2018.",
        )
        args = parser.parse_args()
        config = read_toml(args.config)
        output_name = str(config.get("output_name", Path(args.config).stem))
        configured_output_path = Path(config["output_path"])
        output_path = configured_output_path.with_name(
            f"{output_name}__{args.dataset}.json"
        )
        return cls(
            model_name=config["model_name"],
            output_name=output_name,
            cache_dir=Path(config["cache_dir"]),
            output_path=output_path,
            dataset_name=args.dataset,
            use_dot_product=config.get("use_dot_product", False),
            is_e5_mistral="e5-mistral" in config["model_name"],
            start_index=config.get("start_index", 0),
            end_index=config.get("end_index", None),
        )


if __name__ == "__main__":
    config = Config.from_config()

    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)

    encoder = SentenceTransformer(config.model_name, trust_remote_code=True)
    dim_size = encoder.get_sentence_embedding_dimension()
    assert isinstance(dim_size, int)

    if config.output_path.exists():
        results = sienna.load(config.output_path)
    else:
        results = {}

    end_index = config.end_index if config.end_index else dim_size

    for dim_to_drop in range(config.start_index, end_index):
        print(
            f"{config.dataset_name}: {dim_to_drop} th dimension processing..."
        )
        dims_to_keep = list(range(dim_size))
        del dims_to_keep[dim_to_drop]
        model = Truncator(
            encoder,
            resize_scale=1.0,
            cache_dir=config.cache_dir,
            indexes_to_keep=dims_to_keep,
            is_e5=config.is_e5_mistral,
        )
        evaluator = NanoBEIREvaluator(
            dataset_names=[config.dataset_name],
            ndcg_at_k=[5, 10, 20],
            score_functions={SimilarityFunction.COSINE.value: cos_sim}
            if not config.use_dot_product
            else {SimilarityFunction.DOT_PRODUCT.value: dot_score},
            query_prompts=task_name_to_instruct if config.is_e5_mistral else None,
            batch_size=8 if config.is_e5_mistral else 32,
        )
        result = evaluator(model)
        if str(dim_to_drop) in results and isinstance(results[str(dim_to_drop)], dict):
            results[str(dim_to_drop)].update(result)
        else:
            results[str(dim_to_drop)] = result
        sienna.save(results, config.output_path)
