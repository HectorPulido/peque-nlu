import numpy as np
from sklearn.linear_model import LogisticRegression

from peque_nlu.intent_engines import ModelEngine


class LogisticIntentEngine(ModelEngine):
    def __init__(self, language):
        super().__init__(language)
        self.model = LogisticRegression()

    def predict(self, texts):
        vec_texts = self.vectorizer.transform(texts)
        intents = self.model.predict(vec_texts)
        probabilities = np.max(self.model.predict_proba(vec_texts), axis=1)
        return intents, probabilities

    def fit(self, text, intent):
        text = self.vectorizer.fit_transform(text)
        self.model.fit(text, intent)
