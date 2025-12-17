from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from regression import PCAadapt
import numpy as np


class Uncertainty:

    def __init__(self, n_components=4, confidence=.95):
        self.n_components = n_components
        self.confidence = confidence

    def fit(self, X):
        self.pca = PCAadapt(components=self.n_components)
        X = self.pca.fit_transform(X)
        self.bandwidth = kde_find_bandwidth(X)

        loo = LeaveOneOut()
        likelihoods = []
        for (train_idx, test_idx) in loo.split(X):
            Xtr, Xte = X[train_idx], X[test_idx]
            kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(Xtr)
            log_likelihood = kde.score_samples(Xte)[0]
            likelihoods.append(log_likelihood)

        # refit
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(X)

        self.mean_ll, self.std_ll = np.mean(likelihoods), np.std(likelihoods)
        self.z_threshold = norm.ppf(self.confidence)

        return self

    def predict(self, X):
        X = self.pca.transform(X)
        log_likelihood = self.kde.score_samples(X)
        z_scores = (log_likelihood - self.mean_ll) / self.std_ll
        outliers = np.abs(z_scores) > self.z_threshold
        return outliers


def kde_find_bandwidth(X, cv=10):
    grid = {'bandwidth': np.linspace(0.0001, 1, 100)}
    kde = GridSearchCV(KernelDensity(kernel='gaussian'), grid, cv=cv)
    kde.fit(X)
    bandwidth = kde.best_params_['bandwidth']
    print(f'grid search found {bandwidth=:.5f}')
    return bandwidth

def kde_score(kde, X):
    return np.exp(kde.score_samples(X))