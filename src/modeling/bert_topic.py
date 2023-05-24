import os
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
from bertopic import BERTopic
from pandas import concat, DataFrame, ExcelWriter
from typing import List, Union

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import bertopic_data
from src.utils import (
    getEmbeddings,
    get_wordcloud_object,
    visualize_topic_barchart
)


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
            docs: List[str]
    ) -> None:
        """Fit BERTopic model or load it

        Args:
            transformer_name (str): transformers used for embedding
            docs_name (str): documents name for identifaction
            docs (List[str]): documents
        """
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

    def reduce_topics_(
            self,
            docs: List[str],
            number_topic: int
    ) -> None:
        """Reduce the number of topics

        Args:
            docs (List[str]):  The documents you used when calling
            either `fit` or `fit_transform`
            number_topic (int): number of topics to reduce to
        """
        self.model.reduce_topics(
            docs,
            nr_topics=number_topic
        )

    def barchart_(self):
        """Display all topics composition barchart
        """
        n_topics_ = max(self.model.topics_)
        fig = self.model.visualize_barchart(
            topics=list(range(n_topics_ + 1)),
            n_words=15,
            width=300,
            height=300
        )
        fig.show()

    def representative_docs(
            self,
            docs: List[str],
            raw_docs: List[str]
    ) -> None:
        """Get representative documents per topic

        Args:
            docs (List[str]): documents to pass through the model

        Returns:
            DataFrame: dataframe with representative document and topic
        """
        df_doc_topic = self.model.get_document_info(docs)[
            ["Document", "Topic", "Name", "Representative_document"]
        ]
        df_docs = DataFrame(data=raw_docs, columns=["doc"])
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
        q_repr = ExcelWriter("./data/topic_q_representative.xlsx", engine='xlsxwriter')
        df_doc_representative.to_excel(q_repr, sheet_name="representative docs", index=False)
        q_repr._save()
        return df_doc_representative

    def topic_infeence(
            self,
            docs: List[str],
            raw_docs: List[str],
            topic_id: int
    ) -> None:
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        _wordcloud = get_wordcloud_object(self.model, topic_id)
        ax1.imshow(_wordcloud, interpolation="bilinear")
        ax1.set_axis_off()

        visualize_topic_barchart(ax2, self.model, topic_id, 10)
        plt.tight_layout()
        plt.savefig(
            f"./data/topics_wc/topic_{topic_id}.png",
            bbox_inches="tight", dpi=300
        )
        plt.show()
