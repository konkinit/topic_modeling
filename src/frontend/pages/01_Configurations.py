import os
import sys
import streamlit as st

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data_preprocess import Preprocessing


languages = ['french', 'english']
dict_spacy_model = {
    'french': 'fr_core_news_md',
    'english': 'en_core_web_sm'
}
dict_transformers = {
    'french': [
        "dangvantuan/sentence-camembert-large",
        "dangvantuan/sentence-camembert-base"
    ],
    'english': [
        'bert-large-cased',
        'bert-base-cased'
    ]
}


st.title("Configuration")
st.markdown(
    """
    To preprocess data , configs data are needed to define models.
    """
)

language = st.selectbox(
    'What is the verbatim language',
    tuple(languages)
)
spacy_model = dict_spacy_model[language]


st.markdown(
    """
    ## Stop words
    Given a language , there exists an official stop words list which can
    be extanded with context stop words. Context stop words are those words
    which are related to the main topic of the documents but are not mandatory
    to understand a verbatim.
    """
)
uploaded_file = st.file_uploader("Load the stop-words file", type=['txt'])
if uploaded_file is not None:
    list_context_sw = [
        line.strip().decode('utf-8') for line in uploaded_file.readlines()
    ]
    assert type(list_context_sw[0]) == str, "Encoding problem while reading sw"
    preprocessor = Preprocessing(
        spacy_model,
        language,
        list_context_sw,
        False
    )

    st.session_state["language"] = language
    st.session_state["spacy_model"] = spacy_model
    st.session_state["preprocessor"] = preprocessor
    st.session_state["context_sw"] = list_context_sw

st.markdown(
    """
    ## Transformer for BERTopic
    At the `Embeddings` of BERTopic , a transformer to be downloaded from
    Hugging Face Hub, is required to encode text data to numeric features.
    """
)

transformer = st.selectbox(
    'Choose the transformer model',
    tuple(dict_transformers[language])
)
st.session_state["transformer"] = transformer
