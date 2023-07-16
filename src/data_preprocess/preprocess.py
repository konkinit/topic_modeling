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
from src.utils import context_stopwords, email_check


@Language.factory('language_detector')
def language_detector(nlp, name):
    return LanguageDetector()


class Preprocessing:
    def __init__(
            self,
            model_name: str,
            language: str,
            _context_stopwords: List[str],
            use_preprocessing: bool = True
    ) -> None:
        self.nlp = load(model_name)
        self.nlp.max_length = 2000000
        self.nlp.add_pipe('language_detector', last=True)
        self.stemmer = SnowballStemmer(language=language)
        self.stop_words_ = context_stopwords(language, _context_stopwords)
        self.use_preprocessing = use_preprocessing

    def getLanguage(self, text: str) -> str:
        """Get the language of a doc using spacy lang detect

        Args:
            text (str): text

        Returns:
            str: language
        """
        doc = self.nlp(text)
        return doc._.language['language']

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text

        Args:
            text (str): text

        Returns:
            List[str]: list of token
        """
        return word_tokenize(text)

    def transform_email_data(self, text: str) -> str:
        """Transform email info to generic term

        Args:
            text (str): text

        Returns:
            str: text without email address
        """
        text_no_mail = ""
        for token in text.split(" "):
            if email_check(token):
                text_no_mail = " ".join([text_no_mail, "email"])
            else:
                text_no_mail = " ".join([text_no_mail, token])
        return text_no_mail

    def remove_punct_digit_nonsensstring(self, text: str) -> str:
        """Remove punctuation and non sensical string

        Args:
            text (str): text

        Returns:
            str: clean text
        """
        text_no_punct = text
        for punct in string.punctuation:
            text_no_punct = text_no_punct.replace(punct, " ")
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
        """Lemmatize a text

        Args:
            text (str): raw text

        Returns:
            str: lemmatized text
        """
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def stem(self, text: str):
        """Stem a text

        Args:
            text (str): raw text

        Returns:
            str: stemmed text
        """
        doc = self.nlp(text)
        return " ".join(
            [self.stemmer.stem(token.text) for token in doc]
        )

    def remove_stopwords(self, text: str) -> str:
        """Remove stop words from a text

        Args:
            text (str): raw text

        Returns:
            str: text without stop-word
        """
        clean_token = []
        for t in self.tokenize(text):
            if not (t in self.stop_words_):
                clean_token.append(t)
        return " ".join(clean_token)

    def pipeline(self, text: str) -> str:
        """Preprocessing pipeline

        Args:
            text (str): raw text

        Returns:
            str: preprocessed text
        """
        return self.remove_stopwords(
            self.lemmatize(
                self.remove_punct_digit_nonsensstring(
                    self.transform_email_data(text)
                )
            )
        )
