import os
import sys
import re
import matplotlib.pyplot as plt
import pickle as pkl
import multidict as multidict
import warnings
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from numpy import ndarray
from umap import UMAP
from torch import Tensor
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from typing import (
    List,
    Union
)
from wordcloud import WordCloud

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import (
    umap_data, hdbscan_data, tfidf_data, tokenizer_data, mmr_data
)


warnings.filterwarnings("ignore")


def email_check(text: str) -> bool:
    regex = re.compile(
        r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
    )
    return re.fullmatch(regex, text) is not None


def getEmbeddingsModel(
        transformer_name: str
) -> SentenceTransformer:
    """Configure and Return an Embedding model

    Args:
        transformer_name (str): name of the transformer

    Returns:
        SentenceTransformer: an objectif of type
            SentenceTransformer
    """
    return SentenceTransformer(transformer_name)


def getEmbeddings(
        transformer_name: str,
        docs_name: str,
        docs
) -> Union[List[Tensor], ndarray, Tensor]:
    model_n = transformer_name.split("/")[-1]
    path_ = f"data/embeddings-{docs_name}-{model_n}.pkl"
    if os.path.isfile(os.path.join(path_)):
        return pkl.load(open(f"./{path_}", "rb"))
    sentence_model = getEmbeddingsModel(transformer_name)
    embedding_ = sentence_model.encode(docs, show_progress_bar=False)
    file_ = open(f"./{path_}", "wb")
    pkl.dump(embedding_, file_)
    file_.close()
    return embedding_


def getDimReductionModel(params: umap_data) -> UMAP:
    """Configure and Return an UMAP object

    Args:
        params (umap_data): config parameters

    Returns:
        UMAP: object to pass into BERTopic
    """
    return UMAP(
        n_neighbors=params.n_neighbors,
        n_components=params.n_components,
        min_dist=params.min_dist,
        metric=params.metric,
    )


def getClusteringModel(params: hdbscan_data) -> HDBSCAN:
    return HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        metric=params.metric,
        cluster_selection_method=params.cluster_selection_method,
        prediction_data=params.prediction_data,
        gen_min_span_tree=params.gen_min_span_tree,
    )


def context_stopword(language: str, list_custom_sw: List[str]):
    with open(f'./data/sw-{language}.txt') as f:
        sw_ = [line.strip() for line in f.readlines()]
    f.close()
    return sw_ + list_custom_sw


def getTokenizer(params: tokenizer_data, list_custom_sw: List[str]):
    return (
        CountVectorizer(
            stop_words=params.language
            )
        if params.language == "english"
        else CountVectorizer(
            stop_words=context_stopword(
                params.language,
                context_stopword(params.language, list_custom_sw)
            )
        )
    )


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


def getFrequencyDictForText(
            sentence: str,
            language: str,
            list_custom_sw: List[str]):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}
    stopword_list = context_stopword(language, list_custom_sw)
    for text in sentence.split(" "):
        if text in stopword_list:
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict


def global_wordcloud(
        docs: str,
        language: str,
        list_custom_sw: List[str]
) -> None:
    """Plot a worcloud image

    Args:
        docs (str): sentence
        language (str): doc language
        list_custom_sw (List[str]): stopword based on the language
        and the context
    """
    vocab_ = getFrequencyDictForText(docs, language, list_custom_sw)
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(vocab_)
    plt.figure(figsize=(10, 8), facecolor='k')
    plt.imshow(wc, interpolation="bilinear")
    plt.savefig(
        "./data/wordcloud-corpus.png",
        facecolor='k',
        bbox_inches='tight',
        dpi=500
    )
    plt.axis("off")
    plt.show()
