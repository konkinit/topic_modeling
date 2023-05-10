from dataclasses import dataclass


@dataclass
class sent_transformers_params:
    model_name: str = "dangvantuan/sentence-camembert-large"


@dataclass
class tokenizer_params:
    langage: str = "french"


@dataclass
class tfidf_params:
    reduce_freq_words: bool = True


@dataclass
class MMR_params:
    diversity: float = 0.2
    top_n_words: int = 10


@dataclass
class umap_params:
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "cosine"


@dataclass
class hdbscan_params:
    min_cluster_size: int = 15
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True
