import os
import sys
from bertopic import BERTopic
from pandas import concat, DataFrame
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

    def fit_or_load(self, transformer_name: str, docs_name: str, docs: List[str]):
        model_n = transformer_name.split("/")[-1]
        path_ = f"data/model-{docs_name}-{model_n}"
        if os.path.isfile(os.path.join(path_)):
            self.model = BERTopic.load(f"./{path_}")
        else:
            self.model.fit(docs, getEmbeddings(transformer_name, docs_name, docs))
            self.model.save(f"./{path_}", save_embedding_model=True)

    def heatmap_(self) -> None:
        """Plot clusters correlation matrix
        """
        fig = self.model.visualize_heatmap()
        fig.show()

    def intertopic_(self) -> None:
        """Plot inter topic distance map
        """
        fig = self.model.visualize_topics()
        fig.show()

    def merge_clusters(self, docs: List[str], topics2merge: Union[List, List[list]]):
        self.model.merge_topics(docs, topics_to_merge)

    def barchart_(self):
        """Display all topics composition barchart
        """
        n_topics_ = max(self.model.topics_)
        fig = self.model.visualize_barchart(
            topics=list(range(n_topics_ + 1)), n_words=10, width=300, height=300
        )
        fig.show()

    def representative_docs(self, docs: List[str]) -> DataFrame:
        """Get representative documents per topic

        Args:
            docs (List[str]): documents to pass through the model

        Returns:
            DataFrame: dataframe with representative document and topic
        """
        df_doc_topic = self.model.get_document_info(docs)[
            ["Document", "Topic", "Name", "Representative_document"]
        ]
        df_docs = DataFrame(data=docs, columns=["doc"])
        df_doc_representative = concat([df_docs, df_doc_topic], axis=1)
        df_doc_representative.columns = [
            "raw_doc",
            "clean_doc",
            "topic_id",
            "topic_name",
            "representative_doc",
        ]
        df_doc_representative = (
            df_doc_representative[df_doc_representative["representative_doc"]]
            .sort_values("topic_name")
            .reset_index(drop=True)
        )
        return df_doc_representative
    
    def topic_infeence(self, topic_id: int):
        pass
