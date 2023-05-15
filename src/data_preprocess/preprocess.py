import os
import sys
import string
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from spacy import load
from typing import List
from spacy_langdetect import LanguageDetector
from spacy.language import Language

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import context_stopword, email_check


@Language.factory('language_detector')
def language_detector(nlp, name):
    return LanguageDetector()


class Preprocessing:
    def __init__(self, model_name, language, _context_stopwords):
        self.nlp = load(model_name)
        self.nlp.max_length = 2000000
        self.nlp.add_pipe('language_detector', last=True)
        self.stemmer = SnowballStemmer(language=language)
        self.stop_words_ = context_stopword(language, _context_stopwords)

    def getLanguage(self, text: str) -> str:
        doc = self.nlp(text)
        return doc._.language['language']

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def transform_email_data(self, text: str) -> str:
        text_no_mail = ""
        for token in text.split(" "):
            if email_check(token):
                # token.split("@")[-1]
                text_no_mail = " ".join([text_no_mail, "email"])
            else:
                text_no_mail = " ".join([text_no_mail, token])
        return text_no_mail

    def remove_punct_digit_nonsensstring(self, text: str) -> str:
        text_no_punct = "".join(
            [l_.lower() for l_ in text if l_ not in string.punctuation]
        )
        text_no_punct_digit = "".join(
            [l_ for l_ in text_no_punct if not l_.isdigit()]
            )
        text_no_punct_digit_nsstr = ""
        for token in text_no_punct_digit.split(" "):
            if len(token) > 30:
                text_no_punct_digit_nsstr = " ".join(
                    [text_no_punct_digit_nsstr, ""]
                )
            else:
                text_no_punct_digit_nsstr = " ".join(
                    [text_no_punct_digit_nsstr, token]
                )
        return text_no_punct_digit_nsstr

    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def stem(self, text: str):
        doc = self.nlp(text)
        return " ".join(
            [self.stemmer.stem(token.text) for token in doc]
        )

    def remove_stopwords(self, text: str) -> str:
        return " ".join(
            [i for i in self.tokenize(text) if i not in self.stop_words_]
            )

    def pipeline(self, text: str) -> str:
        return self.remove_stopwords(
                    self.lemmatize(
                        self.remove_punct_digit_nonsensstring(
                            self.transform_email_data(text)
                        )
                    )
                )
