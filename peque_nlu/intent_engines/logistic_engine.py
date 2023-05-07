"""
The LogisticIntentEngine module.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from peque_nlu.intent_engines import ModelEngine


class LogisticIntentEngine(ModelEngine):
    """
    The LogisticIntentEngine class.

    This class is used to create a logistic regresion
    and use it to detect the intent.
    """

    def __init__(self, language):
        """
        Initialize the LogisticIntentEngine.

        :param language: The language to use.
        :type language: str.
        """

        super().__init__(language)
        self.model = LogisticRegression()

    def predict(self, text) -> tuple:
        """
        Predict the intent of the input text.

        :param texts: The input text.
        :type texts: str.
        :return: The predicted intent and the confidence.
        :rtype: tuple.

        example: predict("hello") -> ("greet", 1)
        """

        vec_texts = self.vectorizer.transform(text)
        intents = self.model.predict(vec_texts)
        probabilities = np.max(self.model.predict_proba(vec_texts), axis=1)
        return intents, probabilities

    def fit(self, text, intent):
        """
        Fit the intent engine to train the model.

        :param text: The input text.
        :type text: str.
        """
        text = self.vectorizer.fit_transform(text)
        self.model.fit(text, intent)
