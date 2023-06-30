import os
import sys
import plotly.express as px
import streamlit as st
from datetime import datetime

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import verbatim_lang, verbatim_length


st.title("Metadata Statistics")
st.markdown(
    """
    The dateset can be summarized as following :
    """
)


df_docs = st.session_state["df_docs"]
date_var = st.session_state["date"]
target_var = st.session_state["target_var"]
language = st.session_state["language"]
spacy_model = st.session_state["spacy_model"]
preprocessor = st.session_state["preprocessor"]
list_context_sw = st.session_state["context_sw"]


df_docs[date_var] = df_docs[date_var].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d")
)

df_docs["language"] = (
    df_docs[target_var]
    .apply(preprocessor.getLanguage)
    .apply(verbatim_lang)
)
df_docs["length"] = df_docs[target_var].apply(
    verbatim_length
)
df_lang = df_docs[
    ["language", target_var]
].groupby("language").count().reset_index()


col1, col2, col3 = st.columns(3)
col1.metric(
    label="Verbatims",
    value=df_docs[target_var].shape[0]
)
col2.metric(
    label="First Date",
    value=df_docs[date_var].min().strftime("%Y-%m-%d")
)
col3.metric(
    label="Last Date",
    value=df_docs[date_var].max().strftime("%Y-%m-%d")
)
st.plotly_chart(
    px.pie(
        df_lang,
        values=target_var,
        names="language",
        title="Represented languages in verbatims",
        width=500,
    )
)
st.plotly_chart(
    px.histogram(
        df_docs,
        x="length",
        width=800,
        height=500,
        labels={"length": "Verbatim Length"},
        histnorm="probability density",
    )
)

st.dataframe(df_docs.head(), use_container_width=False)


st.session_state["df_docs"] = df_docs
