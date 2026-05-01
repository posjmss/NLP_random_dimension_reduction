from dataclasses import dataclass
from pathlib import Path

from mteb.models.cache_wrapper import CachedEmbeddingWrapper
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData

task_name_to_instruct: dict[str, str] = {
    "arguana": "Given a claim, find documents that refute the claim",
    "climatefever": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "climate-fever": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "dbpedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "dbpedia-entity": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "fever": "Given a claim, retrieve documents that support or refute the claim",
    "fiqa2018": "Given a financial question, retrieve user replies that best answer the question",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question",
    "hotpotqa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "msmarco": "Given a web search query, retrieve relevant passages that answer the query",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question",
    "quoraretrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "quora": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "scidocs": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "webis-touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "trec-covid": "Given a query on COVID-19, retrieve documents that answer the query",
}


@dataclass
class Truncator:
    st: SentenceTransformer | CachedEmbeddingWrapper
    resize_scale: float | int
    is_e5: bool = False
    cache_dir: Path | None = None
    indexes_to_keep: list[int] | None = None
    query_prompt: str | None = None

    @property
    def model_card_data(self):
        return SentenceTransformerModelCardData()

    def __post_init__(self):
        if self.cache_dir and (not isinstance(self.st, CachedEmbeddingWrapper)):
            self.st = CachedEmbeddingWrapper(self.st, self.cache_dir)

    def encode(self, *args, **kwargs):
        dimension = self.st.get_sentence_embedding_dimension()

        # FinMTEB's evaluators pass prompt_name=task_name unconditionally.
        # SentenceTransformer raises if that name isn't in its prompts dict
        # (which is empty for stock all-mpnet-base-v2 etc.). Drop the lookup
        # when the underlying model has no matching prompt.
        prompt_name = kwargs.get("prompt_name")
        if prompt_name is not None:
            configured = getattr(self.st, "prompts", None) or {}
            if prompt_name not in configured:
                kwargs.pop("prompt_name")

        original_vecs = self.st.encode(*args, **kwargs)

        if self.indexes_to_keep is None:
            if isinstance(self.resize_scale, float):
                end = int(dimension * self.resize_scale)
            else:
                end = self.resize_scale
            if original_vecs.ndim == 1:
                truncated_vecs = original_vecs[:end]
            else:
                truncated_vecs = original_vecs[:, :end]
        else:
            if original_vecs.ndim == 1:
                truncated_vecs = original_vecs[self.indexes_to_keep]
            else:
                truncated_vecs = original_vecs[:, self.indexes_to_keep]

        return truncated_vecs

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs):
        if self.is_e5:
            prompt = "query:"
        elif self.query_prompt is not None:
            prompt = self.query_prompt
        else:
            prompt = None
        return self.encode(queries, batch_size=batch_size, prompt=prompt, **kwargs)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list],
        batch_size: int = 8,
        sep: str = " ",
        **kwargs,
    ):
        if type(corpus) is dict:
            sentences = [
                (
                    (corpus["title"][i] + sep + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                )
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (
                    (doc["title"] + sep + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                )
                for doc in corpus
            ]

        if self.is_e5:
            sentences = [f"passage: {s}" for s in sentences]

        return self.encode(sentences, batch_size=batch_size, **kwargs)
