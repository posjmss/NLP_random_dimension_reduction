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

DATASET_NAMES = [
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]


@dataclass
class Config:
    model_name: str
    cache_dir: Path
    output_path: Path
    use_dot_product: bool
    is_e5_mistral: bool
    # [customized] Allow running attribution on only a slice of dimensions.
    start_index: int
    end_index: int | None

    @classmethod
    def from_config(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str)
        args = parser.parse_args()
        _config = read_toml(args.config)
        return cls(
            _config["model_name"],
            Path(_config["cache_dir"]),
            Path(_config["output_path"]),
            use_dot_product=_config.get("use_dot_product", False),
            is_e5_mistral="e5-mistral" in _config["model_name"],
            # [customized] Match the MTEB attribution script's partial-run config.
            start_index=_config.get("start_index", 0),
            end_index=_config.get("end_index", None),
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

    # [customized] Use start_index/end_index when testing a small dimension range.
    end_index = config.end_index if config.end_index else dim_size

    for dim_to_drop in range(config.start_index, end_index):
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
            dataset_names=DATASET_NAMES,
            score_functions={SimilarityFunction.COSINE.value: cos_sim}
            if not config.use_dot_product
            else {SimilarityFunction.DOT_PRODUCT.value: dot_score},
            query_prompts=task_name_to_instruct if config.is_e5_mistral else None,
            batch_size=8 if config.is_e5_mistral else 32,
        )
        result = evaluator(model)
        results[str(dim_to_drop)] = result
        sienna.save(results, config.output_path)
