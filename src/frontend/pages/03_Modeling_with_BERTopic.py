import os
import sys
import streamlit as st

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import (
    umap_data,
    hdbscan_data,
    tfidf_data,
    tokenizer_data,
    mmr_data,
    bertopic_data,
)
from src.modeling import _BERTopic
from src.utils import (
    getClusteringModel,
    getDimReductionModel,
    getMaximalMarginalRelevance,
    getTfidfTransformers,
    getTokenizer
)


st.title("Modeling with BERTopic")


df_docs = st.session_state["df_docs"]
id_docs = st.session_state["id_docs"]
target_var = st.session_state["target_var"]
language = st.session_state["language"]
list_context_sw = st.session_state["context_sw"]
preprocessor = st.session_state["preprocessor"]
transformer = st.session_state["transformer"]


df_docs[f"clean_{target_var}"] = df_docs[target_var].apply(
    preprocessor.pipeline
)
df_docs[f"empty_clean_{target_var}"] = df_docs[f"clean_{target_var}"].apply(
    lambda x: len(x) == 0
)
df_docs = df_docs.query(
    f"language == '{language[:2]}' and empty_clean_{target_var} == False"
).reset_index(drop=True)
raw_docs = df_docs[target_var].tolist()
docs = df_docs[f"clean_{target_var}"].tolist()


umap_model = getDimReductionModel(umap_data())
hdbscan_model = getClusteringModel(hdbscan_data())
vectorizer_model = getTokenizer(
    tokenizer_data(language=language),
    list_context_sw
)
ctfidf_model = getTfidfTransformers(tfidf_data())
mmr_model = getMaximalMarginalRelevance(mmr_data())
bertopic_config = bertopic_data(
    umap_model,
    hdbscan_model,
    vectorizer_model,
    ctfidf_model,
    mmr_model
)
bert_topic_inst = _BERTopic(bertopic_config)


bert_topic_inst.fit_or_load(transformer, id_docs, docs)

st.plotly_chart(
    bert_topic_inst._intertopic()
)
