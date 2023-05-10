import os
import sys
import matplotlib.pyplot as plt
import pickle as pkl
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from .config import (
    umap_params,
    hdbscan_params,
    sent_transformers_params,
    tfidf_params,
    tokenizer_params,
    MMR_params,
)


def getSentenceTransformers(params: sent_transformers_params):
    return SentenceTransformer(params.model_name)


def getEmbeddings(
        params: sent_transformers_params,
        docs_name: str,
        docs):
    model_n = params.model_name.split('/')[-1]
    path_ = f"data/embeddings-{docs_name}-{model_n}.pkl"
    if os.path.isfile(os.path.join(path_)):
        return pkl.load(open(f"./{path_}", 'rb'))
    sentence_model = getSentenceTransformers(params)
    embedding_ = sentence_model.encode(docs, show_progress_bar=False)
    pkl.dump(embedding_, open(f"./{path_}", "wb"))
    return embedding_


def getDimReductionObj(params: umap_params):
    return UMAP(
        n_neighbors=params.n_neighbors,
        n_components=params.n_components,
        min_dist=params.min_dist,
        metric=params.metric,
    )


def getClusteringObj(params: hdbscan_params):
    return HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        metric=params.metric,
        cluster_selection_method=params.cluster_selection_method,
        prediction_data=params.prediction_data,
    )


def getTokenizer(params: tokenizer_params):
    return CountVectorizer(stop_words=params.langage)


def getTfidfTransformers(params: tfidf_params):
    return ClassTfidfTransformer(
        reduce_frequent_words=params.reduce_freq_words
    )


def getMaximalMarginalRelevance(params: MMR_params):
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
