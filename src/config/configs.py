from dataclasses import dataclass
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import (
    MaximalMarginalRelevance,
    KeyBERTInspired
)
from hdbscan import HDBSCAN
from umap import UMAP
from psutil import cpu_count
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from typing import Union


NB_WORKERS_ = cpu_count(logical=False)//2


@dataclass
class preprocessor_data:
    language: str = "french"
    spacy_model: str = 'fr_core_news_md'


@dataclass
class sent_transformers_data:
    model_name: str = "dangvantuan/sentence-camembert-large"


@dataclass
class tokenizer_data:
    language: str = "french"
    min_df: int = 3
    n_gram_range: tuple = (1, 1)


@dataclass
class tfidf_data:
    reduce_freq_words: bool = True


@dataclass
class mmr_data:
    diversity: float = 0.2
    top_n_words: int = 10


@dataclass
class umap_data:
    n_neighbors: int = 10
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "cosine"


@dataclass
class hdbscan_data:
    min_cluster_size: int = 20
    min_samples: int = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True
    gen_min_span_tree: bool = True


@dataclass
class bertopic_data:
    sent_transformers_model: SentenceTransformer
    umap_model: UMAP
    hdbscan_model: HDBSCAN
    vectorizer_model: CountVectorizer
    ctfidf_model: ClassTfidfTransformer
    mmr_model: MaximalMarginalRelevance
    keybertinspired_model: KeyBERTInspired
    nr_topics: Union[str, int] = "auto"


@dataclass
class st_sess_data:
    TARGET_VAR: str = "target_var"
    DATE_VAR: str = "date"
    ID_DOCS: str = "id_docs"
    DF_DOCS: str = "df_docs"
    N_TOPICS: str = "n_topics"
    LANGUAGE: str = "language"
    SPACY_MODEL: str = "spacy_model"
    PREPROCESSOR: str = "preprocessor"
    CONTEXT_SW: str = "context_sw"
    TRANSFORMER: str = "transformer"
    BERTOPIC_INST: str = "bert_topic_inst"


@dataclass
class parallelism_data:
    N_WORKERS: int = NB_WORKERS_
    SHOW_PROGRESS_BAR: bool = False
    VERBOSE: int = 0
