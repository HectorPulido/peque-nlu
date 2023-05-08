"""
The ModelEngine base class module.
"""
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

from peque_nlu.intent_engines import BasicIntentEngine


class ModelEngine(BasicIntentEngine):
    """
    The ModelEngine class.

    This class is intended to be a parent class.
    Is used to create a model intent engine.
    """

    def __init__(self, language):
        """
        Initialize the ModelEngine.

        :param language: The language to use.
        :type language: str.
        """

        self.stopwords = stopwords.words(language)
        self.stemmer = SnowballStemmer(language)
        self.non_words = list(punctuation)
        self.non_words.extend(["¿", "¡"])
        self.non_words.extend(map(str, range(10)))

        self.vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=self.tokenize,
            lowercase=True,
            stop_words=self.stopwords,
        )

    def _stem_tokens(self, tokens, stemmer) -> list:
        """
        Stem the tokens.

        Stemming is a technique used to reduce the complexity
        of the vocabulary by word down to its word stem.

        :param tokens: The tokens to stem.
        :type tokens: list.
        :param stemmer: The stemmer to use.
        :type stemmer: SnowballStemmer.
        :return: The stemmed tokens.

        example: _stem_tokens(["programming", "world"], SnowballStemmer("english")) ->
            ["program", "world"]
        """

        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(self, text) -> list:
        """
        Tokenize the text.

        :param text: The text to tokenize.
        :type text: str.
        :return: The tokens.
        :rtype: list.
        """

        text = "".join([c for c in text if c not in self.non_words])
        tokens = word_tokenize(text)
        try:
            stems = self._stem_tokens(tokens, self.stemmer)
        except Exception as _:
            stems = [""]
        return stems
