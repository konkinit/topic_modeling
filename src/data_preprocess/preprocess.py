import spacy


class Preprocessing:
    def __init__(self, model_name):
        self.nlp = spacy.load(model_name)

    def lemmatizer(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])
