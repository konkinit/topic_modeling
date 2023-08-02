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
    bertopic_data,
    st_sess_data
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
    This part is dedicated for fitting the stack of models for
    topic generation. Each composed sub-model is initialised with the
    default values as the following:


    To improve topic representation , tuning can be done
    on HDBSCAN hyperparameters mainly `min_cluster_size` and
    `min_samples`
    """
)


df_docs = st.session_state[st_sess_data.DF_DOCS]
id_docs = st.session_state[st_sess_data.ID_DOCS]
target_var = st.session_state[st_sess_data.TARGET_VAR]
language = st.session_state[st_sess_data.LANGUAGE]
list_context_sw = st.session_state[st_sess_data.CONTEXT_SW]
preprocessor = st.session_state[st_sess_data.PREPROCESSOR]
transformer = st.session_state[st_sess_data.TRANSFORMER]

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

_min_cluster_size = st.number_input(
    'Insert the desired minimal cluster size',
    value=50,
    help=f"Provide a number between 1 and {len(raw_docs)}. The default \
    value used in the algorithm is 20."
)

sent_transformers_model = getEmbeddingsModel(
    sent_transformers_data
)
umap_model = getDimReductionModel(umap_data)
hdbscan_model = getClusteringModel(
    hdbscan_data(min_cluster_size=_min_cluster_size)
)
vectorizer_model = getTokenizer(
    tokenizer_data(language=language),
    list_context_sw
)
ctfidf_model = getTfidfTransformers(tfidf_data)
mmr_model = getMaximalMarginalRelevance(mmr_data)
keybertinspired_model = getKeyBERTInspired()
bertopic_config = bertopic_data(
    sent_transformers_model,
    umap_model,
    hdbscan_model,
    vectorizer_model,
    ctfidf_model,
    keybertinspired_model,
    mmr_model,
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
    'Insert the desired number of topics',
    value=0,
    help=f"Provide a number between 1 and {n_topics}. If you are \
    satisfyed with the current number of topic enter -1."
)
if n_topics_ > 0:
    bert_topic_inst._reduce_topics(docs, n_topics_)
    st.session_state[st_sess_data.N_TOPICS] = n_topics_
    st.plotly_chart(
        bert_topic_inst._intertopic()
    )
    st.plotly_chart(
        bert_topic_inst._barchart(),
        use_container_width=True
    )
    st.session_state[st_sess_data.BERTOPIC_INST] = bert_topic_inst
if n_topics_ == -1:
    st.session_state[st_sess_data.N_TOPICS] = n_topics
    st.plotly_chart(
        bert_topic_inst._barchart(),
        use_container_width=True
    )
    st.session_state[st_sess_data.BERTOPIC_INST] = bert_topic_inst
