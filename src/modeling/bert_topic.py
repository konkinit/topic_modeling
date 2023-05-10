import os
import sys
from bertopic import BERTopic

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import bertopic_data
from src.utils import getEmbeddings


class BERTopic_:
    def __init__(self, bertopic_params: bertopic_data):
        self.model = BERTopic(
            nr_topics=bertopic_params.nr_topics,
            top_n_words=bertopic_params.top_n_words,
            n_gram_range=bertopic_params.n_gram_range,
            min_topic_size=bertopic_params.min_topic_size,
            umap_model=bertopic_params.umap_model,
            hdbscan_model=bertopic_params.hdbscan_model,
            vectorizer_model=bertopic_params.vectorizer_model,
            ctfidf_model=bertopic_params.ctfidf_model,
            representation_model=bertopic_params.mmr_model,
        )

    def fit_(self, transformer_name: str, docs_name: str, docs):
        self.model.fit(docs, getEmbeddings(transformer_name, docs_name, docs))

    def tabular_inference(self, docs):
        return (self.model.get_topic_info(), self.model.get_document_info(docs))

    def visual_inference(self):
        fig = self.model.visualize_barchart()
        fig.show()
