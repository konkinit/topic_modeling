import streamlit as st


st.title("Topic Modeling with BERTopic")

st.subheader("About the app")

st.markdown(
    "The app leverages BERTopic modeling made by transformers and c-TF-IDF \
    to create easy interpretable clusters with relevants words. \
    For more information on BERTOopic consult the \
    [official documentation](https://maartengr.github.io/BERTopic/index.html)")


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
