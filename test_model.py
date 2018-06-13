import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from problem import get_cv
from problem import get_train_data

from submissions.my_model.feature_extractor import FeatureExtractor
from submissions.my_model.classifier import Classifier

def evaluation(X, y):
	pipe = make_pipeline(FeatureExtractor(), Classifier())
	cv = get_cv(X, y)
	results = cross_validate(pipe, X, y, scoring=['roc_auc', 'accuracy'], cv=cv,
	                         verbose=1, return_train_score=True,
	                         n_jobs=1)

	return results


data_train, labels_train = get_train_data()
results = evaluation(data_train, labels_train)

print("Training score ROC-AUC: {:.3f} +- {:.3f}".format(np.mean(results['train_roc_auc']),
                                                        np.std(results['train_roc_auc'])))
print("Validation score ROC-AUC: {:.3f} +- {:.3f} \n".format(np.mean(results['test_roc_auc']),
                                                          np.std(results['test_roc_auc'])))

print("Training score accuracy: {:.3f} +- {:.3f}".format(np.mean(results['train_accuracy']),
                                                         np.std(results['train_accuracy'])))
print("Validation score accuracy: {:.3f} +- {:.3f}".format(np.mean(results['test_accuracy']),
                                                           np.std(results['test_accuracy'])))
