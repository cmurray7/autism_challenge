import numpy as np
import networkx as nx
import pandas as pd
from problem import get_cv, get_train_data

from nilearn.connectome import ConnectivityMeasure


def _load_fmri(fmri_filenames):
	"""Load time-series extracted from the fMRI using a specific atlas."""
	return np.array([pd.read_csv(subject_filename,
	                             header=None).values
	                 for subject_filename in fmri_filenames])

data_train, labels_train = get_train_data()
files=data_train.fmri_msdl[0:10]
fmri_data = _load_fmri(files[0:1])

cm = ConnectivityMeasure()
connectivity = cm.fit_transform(fmri_data)
G = nx.to_networkx_graph(connectivity[0,:,:])
b = nx.betweenness_centrality(G)
for v in G.nodes():
    print("%0.2d %5.3f" % (v, b[v]))

print("Degree centrality")
d = nx.degree_centrality(G)
for v in G.nodes():
    print("%0.2d %5.3f" % (v, d[v]))

print("Closeness centrality")
c = nx.closeness_centrality(G)
for v in G.nodes():
    print("%0.2d %5.3f" % (v, c[v]))
