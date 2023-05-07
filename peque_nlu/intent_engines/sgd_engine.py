"""
The SGDIntentEngine class module.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

from peque_nlu.intent_engines import ModelEngine


class SGDIntentEngine(ModelEngine):
    """
    The SGDIntentEngine class.

    This class is used to create a SGDClassifier
    and use it to detect the intent.
    """

    def __init__(self, language):
        """
        Initialize the SGDIntentEngine.

        :param language: The language to use.
        :type language: str.
        """

        super().__init__(language)
        self.model = Pipeline(
            [
                ("tfidf", TfidfTransformer()),
                (
                    "clf-svm",
                    SGDClassifier(
                        loss="hinge",
                        penalty="l2",
                        alpha=1e-3,
                        max_iter=15,
                        random_state=42,
                    ),
                ),
            ]
        )

        # Use CalibratedClassifierCV to get probabilities
        self.calibrator = CalibratedClassifierCV(self.model, cv="prefit")

    def predict(self, text) -> tuple:
        """
        Predict the intent of the input text.

        :param text: The input text.
        :type text: str.
        :return: The predicted intent and the confidence.
        :rtype: tuple.

        example: predict("hello") -> ("greet", 1)
        """

        vec_texts = self.vectorizer.transform(text)
        intents = self.model.predict(vec_texts)

        probabilities = np.max(self.calibrator.predict_proba(vec_texts), axis=1)
        return intents, probabilities

    def fit(self, text, intent):
        """
        Fit the intent engine to train the model.

        :param text: The input text.
        :type text: str.
        """

        text = self.vectorizer.fit_transform(text)
        self.model.fit(text, intent)
        self.calibrator.fit(text, intent)
