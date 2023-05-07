import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from peque_nlu.intent_engines import BasicIntentEngine


class ModelEngine(BasicIntentEngine):
    def __init__(self, language):
        self.stopwords = stopwords.words(language)
        self.stemmer = SnowballStemmer(language)
        self.non_words = list(punctuation)
        self.non_words.extend(["¿", "¡"])
        self.non_words.extend(map(str, range(10)))

        self.vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=self._tokenize,
            lowercase=True,
            stop_words=self.stopwords,
        )

    def _stem_tokens(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def _tokenize(self, text):
        text = "".join([c for c in text if c not in self.non_words])
        tokens = word_tokenize(text)
        try:
            stems = self._stem_tokens(tokens, self.stemmer)
        except Exception as _:
            stems = [""]
        return stems
