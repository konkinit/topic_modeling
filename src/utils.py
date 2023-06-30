import os
import sys
import re
import matplotlib.pyplot as plt
import pickle as pkl
from multidict import MultiDict
import warnings
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import (
    MaximalMarginalRelevance, KeyBERTInspired
)
from hdbscan import HDBSCAN
from numpy import ndarray, arange
from umap import UMAP
from random import choice
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
    sent_transformers_data,
    umap_data,
    hdbscan_data,
    tfidf_data,
    tokenizer_data,
    mmr_data
)


warnings.filterwarnings("ignore")


def email_check(text: str) -> bool:
    """Check if a token is a email

    Args:
        text (str): string token

    Returns:
        bool: _description_
    """
    regex = re.compile(
        r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
    )
    return re.fullmatch(regex, text) is not None


def getEmbeddingsModel(
        params: sent_transformers_data
) -> SentenceTransformer:
    """Configure and Return an Embedding model

    Args:
        params (sent_transformers_data): params

    Returns:
        SentenceTransformer: an objectif of type
            SentenceTransformer
    """
    return SentenceTransformer(params.model_name)


def getEmbeddings(
        transformer_name: str,
        docs_name: str,
        docs: List[str]
) -> Union[List[Tensor], ndarray, Tensor]:
    """Get inputs embeddings from transformer-based model

    Args:
        transformer_name (str): transformer model name
        docs_name (str): documents name
        docs (List[str]): docs

    Returns:
        Union[List[Tensor], ndarray, Tensor]: embeddings
    """
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
    """Configure and return clustering model

    Args:
        params (hdbscan_data): config parameters

    Returns:
        HDBSCAN: object to pass into BERTopic
    """
    return HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        metric=params.metric,
        cluster_selection_method=params.cluster_selection_method,
        prediction_data=params.prediction_data,
        gen_min_span_tree=params.gen_min_span_tree,
    )


def getTokenizer(
        params: tokenizer_data,
        list_custom_sw: List[str]
) -> CountVectorizer:
    """Configure and return a count vectorizer model

    Args:
        params (tokenizer_data): config params
        list_custom_sw (List[str]): list of custom stopword

    Returns:
        CountVectorizer: model to pass into BERTopi
    """
    return (
        CountVectorizer(
            min_df=params.min_df,
            stop_words=params.language
            )
        if params.language == "english"
        else CountVectorizer(
            # min_df=params.min_df,
            stop_words=context_stopwords(params.language, list_custom_sw)
        )
    )


def getTfidfTransformers(
        params: tfidf_data
) -> ClassTfidfTransformer:
    return ClassTfidfTransformer(
        reduce_frequent_words=params.reduce_freq_words
    )


def getMaximalMarginalRelevance(
        params: mmr_data
) -> MaximalMarginalRelevance:
    """Configure and return Maximal Marginal Relevance model

    Args:
        params (mmr_data): params

    Returns:
        MaximalMarginalRelevance: mmr object to pass into
        BERTopic
    """
    return MaximalMarginalRelevance(
        diversity=params.diversity, top_n_words=params.top_n_words
    )


def getKeyBERTInspired() -> KeyBERTInspired:
    """Configure and return a KeyBERTInspired model

    Returns:
        KeyBERTInspired: configured model for representation
    """
    return KeyBERTInspired()


def context_stopwords(
        language: str,
        list_custom_sw: List[str]
) -> List:
    """Union offical language stopword and context stop_word and
    return a list of stop_word

    Args:
        language (str): documents language
        list_custom_sw (List[str]): custom stopwords based
        on a the context

    Returns:
        List: stop-words
    """
    with open(f'./data/sw-{language}.txt') as f:
        sw_ = [line.strip() for line in f.readlines()]
    f.close()
    return list(set(list_custom_sw+sw_))


def plot_wordcloud(
        model,
        topic: int,
        wc_name: str
) -> None:
    """Plot wordcloud

    Args:
        model (_type_): bertopic model
        topic (int): topic id number
        wc_name (str): wordcloud storage name
    """
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(
        f"./data/topics_wc/{wc_name}.png",
        bbox_inches="tight",
        dpi=300
    )
    plt.show()


def get_wordcloud_object(model, topic: int) -> None:
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    return wc


def getFrequencyDictForText(
        sentence: str,
        language: str,
        list_custom_sw: List[str]
) -> MultiDict:
    fullTermsDict = MultiDict()
    tmpDict = {}
    stopword_list = context_stopwords(language, list_custom_sw)
    for text in sentence.split(" "):
        if text in stopword_list:
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    tmpDict = dict(sorted(tmpDict.items(), key=lambda item: item[1]))
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
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.savefig(
        "./data/wordcloud-corpus.png",
        facecolor='k',
        bbox_inches="tight",
        dpi=300
    )
    plt.axis("off")
    plt.show()


def visualize_topic_barchart(
        ax,
        topic_model,
        topic: int,
        n_words: int = 10
) -> None:
    color_ = choice([
        "#D55E00",
        "#0072B2",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442"
    ])
    words = [
        word + "  " for word, _ in topic_model.get_topic(topic)
    ][:n_words][::-1]
    scores = [
        score for _, score in topic_model.get_topic(topic)
    ][:n_words][::-1]
    words_pos = arange(len(words))
    ax.barh(words_pos, scores, align='center', color=color_)
    ax.set_yticks(words_pos, labels=words)
    ax.set_xlabel("score")


def verbatim_length(x: str) -> int:
    """Get a verbatim number of
    blank-separated tokens

    Args:
        x (str): verbatim

    Returns:
        int: num of tokens
    """
    return len(x.split(" "))


def empty_verbatim_assertion(x) -> bool:
    """Check if a verbatim is empty

    Args:
        x (_type_): verbatim

    Returns:
        bool: empty flag
    """
    return len(x) == 0


def verbatim_lang(x: str) -> str:
    """Extract the language id from language
    string

    Args:
        x (str): language string name e.g.
        french, english

    Returns:
        str: language id in two 2 letters
    """
    if x in ["fr", "en"]:
        return x
    return "other_lang"
