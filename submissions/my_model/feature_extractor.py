import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
<<<<<<< HEAD
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer

from nilearn.connectome import ConnectivityMeasure
from networkx import to_networkx_graph, betweenness_centrality
=======
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from nilearn.connectome import ConnectivityMeasure
from networkx import betweenness_centrality, to_networkx_graph
>>>>>>> c7419f7a4d62eb8667d72942b105948692e4a602

def _load_fmri(fmri_filenames):
	"""Load time-series extracted from the fMRI using a specific atlas."""
	return np.array([pd.read_csv(subject_filename,
	                             header=None).values
	                 for subject_filename in fmri_filenames])

def _get_centrality(connectivity_matrices):
	func = lambda matrix : np.fromiter(betweenness_centrality(to_networkx_graph(matrix),
	                                                          weight='weight').values(), dtype=float)
	return np.array([func(matrix) for matrix in connectivity_matrices])

class FeatureExtractor(BaseEstimator, TransformerMixin):
	def __init__(self):
		# self.transformer_fmri = make_pipeline(
		# 	FunctionTransformer(func=_load_fmri, validate=False),
		# 	ConnectivityMeasure(kind='tangent', vectorize=True))

		ft = FunctionTransformer(func=_load_fmri, validate=False)
		cm = ConnectivityMeasure(kind='tangent', vectorize=False)
		gc = FunctionTransformer(func=_get_centrality, validate=False)

		self.transformer_fmri = Pipeline([('load_fmri', ft),
		                                  ('connectivity', cm),
		                                  ('centrality', gc)])

	def fit(self, X_df, y):
		fmri_filenames = X_df['fmri_msdl']
		self.transformer_fmri.fit(fmri_filenames, y)
		return self

	def transform(self, X_df):
		fmri_filenames = X_df['fmri_msdl']
		X_connectome = self.transformer_fmri.transform(fmri_filenames)
		X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
		X_connectome.columns = ['connectome_{}'.format(i)
		                        for i in range(X_connectome.columns.size)]
		# get the anatomical information
		X_anatomy = X_df[[col for col in X_df.columns
		                  if col.startswith('anatomy')]]
		X_anatomy = X_anatomy.drop(columns='anatomy_select')
		# concatenate both matrices
		return pd.concat([X_connectome, X_anatomy], axis=1)

