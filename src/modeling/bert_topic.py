import os
import sys
from bertopic import BERTopic

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import bertopic_data
from src.utils import getEmbeddings


class BERTopic_:
    def __init__(
            self,
            bertopic_params: bertopic_data):
        self.model = BERTopic(
            nr_topics=bertopic_params.nr_topics,
            top_n_words=bertopic_params.top_n_words,
            n_gram_range=bertopic_params.n_gram_range,
            min_topic_size=bertopic_params.min_topic_size,
            umap_model=bertopic_params.umap_model,
            hdbscan_model=bertopic_params.hdbscan_model,
            vectorizer_model=bertopic_params.vectorizer_model,
            ctfidf_model=bertopic_params.ctfidf_model,
            mmr_model=bertopic_params.mmr_model
        )

    def fit(self, transformer_name: str, docs_name: str, docs):
        self.model.fit(
            embeddings=getEmbeddings(transformer_name, docs_name, docs)
        )
