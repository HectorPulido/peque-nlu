import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

from peque_nlu.intent_engines import ModelEngine


class SGDIntentEngine(ModelEngine):
    def __init__(self, language):
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

        self.calibrator = CalibratedClassifierCV(self.model, cv="prefit")

    def predict(self, texts):
        vec_texts = self.vectorizer.transform(texts)
        intents = self.model.predict(vec_texts)

        probabilities = np.max(self.calibrator.predict_proba(vec_texts), axis=1)
        return intents, probabilities

    def fit(self, text, intent):
        text = self.vectorizer.fit_transform(text)
        self.model.fit(text, intent)
        self.calibrator.fit(text, intent)
