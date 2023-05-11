import os
import sys
from bertopic import BERTopic
from typing import List
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

    def fit_or_load(
            self,
            transformer_name: str,
            docs_name: str,
            docs: List[str]):
        model_n = transformer_name.split("/")[-1]
        path_ = f"data/model-{docs_name}-{model_n}"
        if os.path.isfile(os.path.join(path_)):
            self.model = BERTopic.load(f"./{path_}")
        else:
            self.model.fit(
                docs,
                getEmbeddings(transformer_name, docs_name, docs)
            )
            self.model.save(f"./{path_}", save_embedding_model=True)

    def tabular_inference(self, docs):
        return (
            self.model.get_topic_info(),
            self.model.get_document_info(docs)
        )

    def visual_inference(self,):
        n_topics_ = max(self.model.topics_)
        fig = self.model.visualize_barchart(
            topics=range(n_topics_),
            n_words=10,
            width=300,
            height=300
        )
        fig.show()
