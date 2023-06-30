import os
import sys
import pytest

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import preprocessor_data
from src.data_preprocess import Preprocessing


# We adopt a school context sentences to wrrite unit tests


with open("./data/test-context-stopwords.txt") as f:
    _list_context_sw = [
        line.strip() for line in f.readlines()
    ]
f.close()


preprocessor = Preprocessing(
    preprocessor_data.spacy_model,
    preprocessor_data.language,
    _list_context_sw
)


@pytest.mark.parametrize(
    "text, language",
    [
        ("Je suis un étudiant dans une école généraliste", "fr"),
        ("I am a graduated student in machine learning", "en")
    ]
)
def test_language(text: str, language: str) -> None:
    assert preprocessor.getLanguage(text) == language


@pytest.mark.parametrize(
    "text, preprocesssed_text",
    [
        (
            "Je suis en école d'ingénieur. Mon addresse \
            est konkobo@idrissa.com",
            "ingénieur addresse email"
        ),
        (
            "Les cours d'hiver sont suspendus pour cause de \
            météo défavorable",
            "cours hiver suspendre cause météo défavorable"
        ),
    ]
)
def test_preprocessing_pipe(
        text: str, preprocesssed_text: str
) -> None:
    assert preprocessor.pipeline(text) == preprocesssed_text
