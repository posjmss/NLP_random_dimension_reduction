from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers.SentenceTransformer import SentenceTransformer

from random_embedding_truncation.dimension_attribution_analysis_finmteb._finmteb_path import (
    add_finmteb_to_path,
)
from random_embedding_truncation.dimension_attribution_analysis_finmteb.select_tasks import (
    DEFAULT_SEED,
    select_tasks,
)
from random_embedding_truncation.truncator import Truncator
from random_embedding_truncation.utils import read_toml

add_finmteb_to_path()
from finance_mteb import MTEB  # noqa: E402

DEFAULT_TASK_LIST = list(select_tasks(DEFAULT_SEED).values())


@dataclass
class Config:
    model_name: str
    cache_dir: Path
    result_output_dir: Path
    use_dot_product: bool
    batch_size: int
    task_list: list[str]
    start_index: int
    end_index: int | None

    @classmethod
    def from_config(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str)
        args = parser.parse_args()
        _config = read_toml(args.config)

        task_list = _config.get("task_list", DEFAULT_TASK_LIST)
        return cls(
            _config["model_name"],
            Path(_config["cache_dir"]),
            Path(_config["result_output_dir"]),
            use_dot_product=_config.get("use_dot_product", False),
            batch_size=_config.get("batch_size", 32),
            task_list=task_list,
            start_index=_config.get("start_index", 0),
            end_index=_config.get("end_index", None),
        )

    @property
    def is_e5_mistral(self) -> bool:
        return "e5-mistral" in self.model_name


if __name__ == "__main__":
    config = Config.from_config()

    if not config.result_output_dir.parent.exists():
        config.result_output_dir.parent.mkdir(parents=True)

    encoder = SentenceTransformer(config.model_name, trust_remote_code=True)

    dim_size = encoder.get_sentence_embedding_dimension()
    assert isinstance(dim_size, int)

    end_index = config.end_index if config.end_index else dim_size

    for dim_to_drop in range(config.start_index, end_index):
        print(f"{dim_to_drop} th dimension processing...")
        dims_to_keep = list(range(dim_size))
        del dims_to_keep[dim_to_drop]
        model = Truncator(
            encoder,
            resize_scale=1.0,
            cache_dir=config.cache_dir,
            indexes_to_keep=dims_to_keep,
            is_e5=config.is_e5_mistral,
        )
        for task in config.task_list:
            output_folder = (
                config.result_output_dir
                / f"{task}_{config.model_name.replace('/', '-')}_{dim_to_drop}"
            )
            evaluation = MTEB(tasks=[task])
            evaluation.run(
                model,
                output_folder=str(output_folder),
                encode_kwargs={"batch_size": config.batch_size},
            )
