import os
import sys
import matplotlib.pyplot as plt
import pickle as pkl
import warnings
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import (
    umap_data,
    hdbscan_data,
    tfidf_data,
    tokenizer_data,
    mmr_data
)


warnings.filterwarnings("ignore")


def getEmbeddingsModel(transformer_name: str):
    return SentenceTransformer(transformer_name)


def getEmbeddings(transformer_name: str, docs_name: str, docs):
    model_n = transformer_name.split("/")[-1]
    path_ = f"data/embeddings/embeddings-{docs_name}-{model_n}.pkl"
    if os.path.isfile(os.path.join(path_)):
        return pkl.load(open(f"./{path_}", "rb"))
    sentence_model = getEmbeddingsModel(transformer_name)
    embedding_ = sentence_model.encode(docs, show_progress_bar=False)
    file_ = open(f"./{path_}", "wb")
    pkl.dump(embedding_, file_)
    file_.close()
    return embedding_


def getDimReductionModel(params: umap_data):
    return UMAP(
        n_neighbors=params.n_neighbors,
        n_components=params.n_components,
        min_dist=params.min_dist,
        metric=params.metric,
    )


def getClusteringModel(params: hdbscan_data):
    return HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        metric=params.metric,
        cluster_selection_method=params.cluster_selection_method,
        prediction_data=params.prediction_data,
    )


def getTokenizer(params: tokenizer_data):
    return CountVectorizer(stop_words=params.language)


def getTfidfTransformers(params: tfidf_data):
    return ClassTfidfTransformer(
        reduce_frequent_words=params.reduce_freq_words
    )


def getMaximalMarginalRelevance(params: mmr_data):
    return MaximalMarginalRelevance(
        diversity=params.diversity, top_n_words=params.top_n_words
    )


def create_wordcloud(model, topic: int):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
