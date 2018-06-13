from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


class Classifier(BaseEstimator):
	def __init__(self):
		self.clf = make_pipeline(StandardScaler(),
		                         MLPClassifier(hidden_layer_sizes=(900, 450, 225, 450, 200),
		                                       alpha=0.0001,
		                                       max_iter=1000,
		                                       early_stopping=True,
		                                       validation_fraction=0.15))

	def fit(self, X, y):
		self.clf.fit(X, y)
		return self

	def predict(self, X):
		return self.clf.predict(X)

	def predict_proba(self, X):
		return self.clf.predict_proba(X)
