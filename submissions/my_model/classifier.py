from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


class Classifier(BaseEstimator):
	def __init__(self):
		self.clf = make_pipeline(StandardScaler(),
		                         MLPClassifier(hidden_layer_sizes=(450, 200),
		                                       solver='adam',
		                                       tol=0.01,
		                                       early_stopping=True,
		                                       validation_fraction=0.15,
                                                       verbose=True))

	def fit(self, X, y):
		self.clf.fit(X, y)
		return self

	def predict(self, X):
		return self.clf.predict(X)

	def predict_proba(self, X):
		return self.clf.predict_proba(X)
