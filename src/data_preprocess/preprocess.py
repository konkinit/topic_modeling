import os
import sys
import spacy
import string
from nltk.tokenize import word_tokenize
from typing import List

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import context_stopword


class Preprocessing:
    def __init__(self, model_name, language, _context_stopwords):
        self.nlp = spacy.load(model_name)
        self.stop_words_ = context_stopword(language, _context_stopwords)

    def remove_punct_digit(self, text) -> str:
        text_no_punct = "".join(
            [l_.lower() for l_ in text if l_ not in string.punctuation]
            )
        text_no_punct_digit = ''.join(
            [l_ for l_ in text_no_punct if not l_.isdigit()]
            )
        return text_no_punct_digit

    def tokenize(self, text) -> List[str]:
        return word_tokenize(text)
        
    def remove_stopwords(self, text) -> str:
        return " ".join(
            [i for i in self.tokenize(text) if i not in self.stop_words_]
        )

    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def pipeline(self, text) -> str:
        return self.lemmatize(
                    self.remove_stopwords(
                        self.remove_punct_digit(text)
                    )
                )
