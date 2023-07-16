import os
import sys
import streamlit as st

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import (
    sent_transformers_data,
    umap_data,
    hdbscan_data,
    tfidf_data,
    tokenizer_data,
    mmr_data,
    bertopic_data
)
from src.modeling import _BERTopic
from src.utils import (
    getClusteringModel,
    getDimReductionModel,
    getMaximalMarginalRelevance,
    getTfidfTransformers,
    getTokenizer,
    getEmbeddingsModel,
    getKeyBERTInspired,
    empty_verbatim_assertion
)


st.title("Modeling with BERTopic")
st.markdown(
    """
    This part is dedicated for fitting the stack of models
    topic generation. After initialising the model ,
    """
)


df_docs = st.session_state["df_docs"]
id_docs = st.session_state["id_docs"]
target_var = st.session_state["target_var"]
language = st.session_state["language"]
list_context_sw = st.session_state["context_sw"]
preprocessor = st.session_state["preprocessor"]
transformer = st.session_state["transformer"]

if preprocessor.use_preprocessing:
    df_docs[f"clean_{target_var}"] = df_docs[target_var].apply(
        preprocessor.pipeline
    )
else:
    df_docs[f"clean_{target_var}"] = df_docs[target_var].copy()
df_docs[
    f"empty_clean_{target_var}"
] = df_docs[f"clean_{target_var}"].apply(empty_verbatim_assertion)
df_docs = df_docs.query(
    f"empty_clean_{target_var} == False"
).reset_index(drop=True)
# language == '{language[:2]}' and
st.session_state["df_docs"] = df_docs
raw_docs, docs = (
    df_docs[target_var].tolist(),
    df_docs[f"clean_{target_var}"].tolist()
)

sent_transformers_model = getEmbeddingsModel(
    sent_transformers_data()
)
umap_model = getDimReductionModel(umap_data())
hdbscan_model = getClusteringModel(hdbscan_data())
vectorizer_model = getTokenizer(
    tokenizer_data(language=language),
    list_context_sw
)
ctfidf_model = getTfidfTransformers(tfidf_data())
mmr_model = getMaximalMarginalRelevance(mmr_data())
keybertinspired_model = getKeyBERTInspired()
bertopic_config = bertopic_data(
    sent_transformers_model,
    umap_model,
    hdbscan_model,
    vectorizer_model,
    ctfidf_model,
    mmr_model,
    keybertinspired_model
)
bert_topic_inst = _BERTopic(bertopic_config)


bert_topic_inst.fit_or_load(
    transformer, id_docs, docs
)

st.plotly_chart(
    bert_topic_inst._intertopic()
)

n_topics = max(bert_topic_inst.model.topics_)

n_topics_ = st.number_input(
    'Insert the desied number of topics',
    value=0,
    help=f"Provide a number between 1 and {n_topics}. If you are \
    satisfyed with the the current number of topic enter -1."
)
if n_topics_ > 0:
    bert_topic_inst._reduce_topics(docs, n_topics_)
    st.session_state["n_topics"] = n_topics_
    st.plotly_chart(
        bert_topic_inst._intertopic()
    )
    st.plotly_chart(
        bert_topic_inst._barchart(),
        use_container_width=True
    )
    st.session_state["bert_topic_inst"] = bert_topic_inst
if n_topics_ == -1:
    st.session_state["n_topics"] = n_topics
    st.plotly_chart(
        bert_topic_inst._barchart(),
        use_container_width=True
    )
    st.session_state["bert_topic_inst"] = bert_topic_inst
