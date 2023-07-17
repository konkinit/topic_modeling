import os
import sys
import streamlit as st
from pandas import read_csv

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import st_sess_data


st.title("Topic Modeling with BERTopic")

st.subheader("About the app")

st.markdown(
    """
    The app leverages BERTopic modeling made by transformers and c-TF-IDF \
    to create easy interpretable clusters with relevants words. \
    For more information on BERTOopic consult the \
    [official documentation](https://maartengr.github.io/BERTopic/index.html)
    """
)


st.subheader("How does the algorithm work?")

st.markdown(
    " A BERTopic model is a stack of process and a default model is built \
    with six steps wwhich are:\n\
    - `Embeddings`: \n\
    - `Dimensionality Reduction`: \n\
    - `Clustering`: \n\
    - `Tokenizer`: \n\
    - `Weithing scheme`: "
)


st.subheader("How to use this app?")

st.markdown(
    "Load a dataset and provide the verbatim feature name: "
)

uploaded_file = st.file_uploader(
    "Choose the dataset",
    type=['csv'],
    help="It is recommended to upload a dataset in `.csv` \
    format with `|` as separator."
)
if uploaded_file is not None:
    df_docs = read_csv(uploaded_file, sep="|", encoding="utf-8")
    id_docs = uploaded_file.name.split(".")[0]
    target_var = st.text_input(
        "Enter the verbatim variable name",
    )
    date_var = st.text_input(
        "Enter the verbatim date variable name if it exists ",
        help="Leave it empty if there is no date feature"
    )
    if target_var and date_var:
        st.session_state[st_sess_data.TARGET_VAR] = target_var
        st.session_state[st_sess_data.DATE_VAR] = date_var
        st.session_state[st_sess_data.ID_DOCS] = id_docs
        st.session_state[st_sess_data.DF_DOCS] = df_docs[
            [date_var, target_var]
        ]
