import os
import sys
import string
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from spacy import load
from typing import List

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import context_stopword


class Preprocessing:
    def __init__(self, model_name, language, _context_stopwords):
        self.nlp = load(model_name)
        self.stemmer = SnowballStemmer(language=language)
        self.stop_words_ = context_stopword(language, _context_stopwords)

    def remove_punct_digit(self, text: str) -> str:
        text_no_punct = "".join(
            [l_.lower() for l_ in text if l_ not in string.punctuation]
        )
        text_no_punct_digit = "".join(
            [l_ for l_ in text_no_punct if not l_.isdigit()]
            )
        return text_no_punct_digit

    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def stem(self, text: str):
        doc = self.nlp(text)
        return " ".join(
            [self.stemmer.stem(token.text) for token in doc]
        )

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def remove_stopwords(self, text: str) -> str:
        return " ".join(
            [i for i in self.tokenize(text) if i not in self.stop_words_]
            )

    def pipeline(self, text: str) -> str:
        return self.remove_stopwords(
                    self.lemmatize(
                        self.remove_punct_digit(text)
                    )
                )
