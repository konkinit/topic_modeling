import os
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
from bertopic import BERTopic
from pandas import concat, DataFrame
from plotly.graph_objects import Figure
from seaborn import set_theme
from typing import List
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import bertopic_data
from src.utils import (
    get_wordcloud_object,
    visualize_topic_barchart
)


set_theme()


class _BERTopic:
    def __init__(self, bertopic_params: bertopic_data) -> None:
        self.model = BERTopic(
            nr_topics=bertopic_params.nr_topics,
            umap_model=bertopic_params.umap_model,
            embedding_model=bertopic_params.sent_transformers_model,
            hdbscan_model=bertopic_params.hdbscan_model,
            vectorizer_model=bertopic_params.vectorizer_model,
            ctfidf_model=bertopic_params.ctfidf_model,
            representation_model=bertopic_params.keybertinspired_model
        )
        self._min_cluster_size = (
            bertopic_params.hdbscan_model.min_cluster_size,
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
        path_ = f"data/model-{docs_name}-{model_n}-{self._min_cluster_size}"
        if os.path.isfile(os.path.join(path_)):
            self.model = BERTopic.load(f"./{path_}")
        else:
            self.model.fit(docs)
            self.model.save(f"./{path_}", save_embedding_model=True)

    def _heatmap(self) -> Figure:
        """Plot clusters correlation matrix

        Returns:
            Figure: heatmap figure object
        """
        return self.model.visualize_heatmap()

    def _intertopic(self) -> Figure:
        """Plot inter topic distance map

        Returns:
            Figure: intertopic map figure object
        """
        return self.model.visualize_topics()

    def _reduce_topics(
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

    def _barchart(self) -> Figure:
        """Return all topics composition barchart

        Returns:
            Figure: barchat figure object
        """
        n_topics_ = max(self.model.topics_)
        return self.model.visualize_barchart(
            topics=list(range(n_topics_ + 1)),
            title="",
            n_words=15,
            width=300,
            height=300
        )

    def representative_docs(
            self,
            docs: List[str],
            raw_docs: List[str]
    ) -> DataFrame:
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
        ).drop(
            axis=1, columns=["representative_doc", "topic_name", "clean_doc"]
        )
        return df_doc_representative

    def topic_stat(
            self,
            topic_id: int
    ) -> DataFrame:
        """Produce stats of a given topic

        Args:
            topic_id (int): topic id
        """
        return self.model.get_topic_info(topic_id)

    def topic_plot(
            self,
            topic_id: int
    ) -> None:
        """Produce plot of a given topic

        Args:
            topic_id (int): topic id
        """
        fig = plt.figure(figsize=(13.5, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        _wordcloud = get_wordcloud_object(self.model, topic_id)
        ax1.imshow(_wordcloud, interpolation="bilinear")
        ax1.set_axis_off()

        visualize_topic_barchart(ax2, self.model, topic_id, 10)
        fig.tight_layout()
        fig.savefig(
            f"./data/topics_wc/topic_{topic_id}.svg",
            format="svg", bbox_inches="tight", dpi=700
        )
        # fig.savefig(
        #    f"./data/topics_wc/topic_{topic_id}.pgf",
        #    bbox_inches="tight", format="pgf"
        # )
        return fig
