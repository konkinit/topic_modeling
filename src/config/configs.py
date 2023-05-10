from dataclasses import dataclass
from typing import Any


@dataclass
class sent_transformers_data:
    model_name: str = "dangvantuan/sentence-camembert-large"


@dataclass
class tokenizer_data:
    langage: str = "french"


@dataclass
class tfidf_data:
    reduce_freq_words: bool = True


@dataclass
class mmr_data:
    diversity: float = 0.2
    top_n_words: int = 10


@dataclass
class umap_data:
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "cosine"


@dataclass
class hdbscan_data:
    min_cluster_size: int = 15
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True


@dataclass
class bertopic_data:
    nr_topics: str = "auto"
    top_n_words: int = 10
    n_gram_range: tuple = (1, 3)
    min_topic_size: int = 10
    umap_model: Any
    hdbscan_model: Any
    vectorizer_model: Any
    ctfidf_model: Any
    mmr_model: Any
