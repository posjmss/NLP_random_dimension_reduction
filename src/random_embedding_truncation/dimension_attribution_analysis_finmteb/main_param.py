from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers.SentenceTransformer import SentenceTransformer

from random_embedding_truncation.dimension_attribution_analysis_finmteb._finmteb_path import (
    add_finmteb_to_path,
)
from random_embedding_truncation.truncator import Truncator
from random_embedding_truncation.utils import read_toml

add_finmteb_to_path()
from finance_mteb import MTEB  # noqa: E402


@dataclass
class Config:
    model_name: str
    cache_dir: Path
    result_output_dir: Path
    task: str
    batch_size: int
    start_index: int
    end_index: int | None

    @classmethod
    def from_config(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument(
            "--task",
            type=str,
            required=True,
            help="Single FinMTEB task name, e.g. FinSentClassification.",
        )
        args = parser.parse_args()
        config = read_toml(args.config)
        return cls(
            model_name=config["model_name"],
            cache_dir=Path(config["cache_dir"]),
            result_output_dir=Path(config["result_output_dir"]),
            task=args.task,
            batch_size=config.get("batch_size", 32),
            start_index=config.get("start_index", 0),
            end_index=config.get("end_index", None),
        )

    @property
    def is_e5_mistral(self) -> bool:
        return "e5-mistral" in self.model_name


if __name__ == "__main__":
    config = Config.from_config()

    if not config.result_output_dir.exists():
        config.result_output_dir.mkdir(parents=True)

    encoder = SentenceTransformer(config.model_name, trust_remote_code=True)
    dim_size = encoder.get_sentence_embedding_dimension()
    assert isinstance(dim_size, int)

    model_slug = config.model_name.replace("/", "-")
    end_index = config.end_index if config.end_index else dim_size

    for dim_to_drop in range(config.start_index, end_index):
        print(f"{config.task}: {dim_to_drop} th dimension processing...")
        dims_to_keep = list(range(dim_size))
        del dims_to_keep[dim_to_drop]
        model = Truncator(
            encoder,
            resize_scale=1.0,
            cache_dir=config.cache_dir,
            indexes_to_keep=dims_to_keep,
            is_e5=config.is_e5_mistral,
        )
        output_folder = (
            config.result_output_dir / f"{config.task}_{model_slug}_{dim_to_drop}"
        )
        evaluation = MTEB(tasks=[config.task])
        evaluation.run(
            model,
            output_folder=str(output_folder),
            encode_kwargs={"batch_size": config.batch_size},
        )
