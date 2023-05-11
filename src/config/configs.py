from dataclasses import dataclass
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class sent_transformers_data:
    model_name: str = "dangvantuan/sentence-camembert-large"


@dataclass
class tokenizer_data:
    language: str = "english"


@dataclass
class tfidf_data:
    reduce_freq_words: bool = True


@dataclass
class mmr_data:
    diversity: float = 0.2
    top_n_words: int = 15


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
    gen_min_span_tree: bool = True


@dataclass
class bertopic_data:
    umap_model: UMAP
    hdbscan_model: HDBSCAN
    vectorizer_model: CountVectorizer
    ctfidf_model: ClassTfidfTransformer
    mmr_model: MaximalMarginalRelevance
    nr_topics: str = "auto"
    top_n_words: int = 15
    n_gram_range: tuple = (1, 3)
    min_topic_size: int = 10
