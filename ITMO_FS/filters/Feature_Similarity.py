from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.covariance import EmpiricalCovariance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import copy

class FeatureSimilarity(BaseEstimator, TransformerMixin):
    """
    Python-based implementation of an unsupervised feature selection 
    algorithm based on feature similarity. The method is based on 
    measuring similarity between features whereby redundancy therein 
    is removed. This does not need any search and, therefore, is fast.
    The algorithm is generic in nature and has the capability of multiscale
    representation of data sets.
 
 
    Parameters
    ----------
    k_initial : int, default = 10
        The initial distance at which the similarity measure is taken.
    verbose : bool, default=False
        Controls verbosity of output.
    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of selected features.

    """

    def __init__(self, k_initial=10, verbose=False):
        self.k_initial = k_initial
        self.verbose = verbose

    def fit(self, X):
        """
        Fits the feature selection method.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input.
        """

        return self._fit(X)

    def transform(self, X, return_df=False):
        """
        Removes the features which have not been selected by method.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input.   
        return_df : boolean, default = False
            Returns a Pandas dataframe if true.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected.
        """

        return self._transform(X, return_df)

    def fit_transform(self, X, return_df=False):
        """
        Fits Boruta, then removes irrelevant features.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input.
        y : array-like, shape = [n_samples]
            The target values.     
        return_df : boolean, default = False
            Returns a Pandas dataframe if true.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected.
        """

        self._fit(X)
        return self._transform(X, return_df)

    def score(self, X, y, method, return_df=False):
        """
        Tests method by comparing the performance of a learning method before and after feature selection
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input.
        y : array-like, shape = [n_samples]
            The target values.
        remove_unclear: boolean, default = False
            Remove unclear features if true.    
        method: object
            The learning method.
        """
        
        X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size=0.2)
        method.fit(X_train, y_train)

        X_filtered_train = self.transform(X_train, return_df = return_df)
        X_filtered_test = self.transform(X_test, return_df = return_df)
        
        method_filtered = copy.copy(method)
        method_filtered.fit(X_filtered_train, y_train.ravel())

        score = method.score(X_test, y_test.ravel())
        score_filtered = method_filtered.score(X_filtered_test, y_test.ravel())

        return score_filtered / score
    def _validate_pandas_input(self, arg):
        try:
            return arg.values
        except AttributeError:
            raise ValueError(
                "input needs to be a numpy array or pandas data frame."
            )

    def _fit(self, X):
        
        self._check_params()
        
        data = pd.DataFrame(X)
        n_feat = len(data.columns)
        k = self.k_initial
        threshold = 0

        while k >= 1:
            #Check if K is not higher than the amount of columns in the actual dataset
            k = min(k, len(data.columns) - 1)

            #Create covariance matrix
            covar = np.abs(data.corr(method='pearson').to_numpy())
            ind = np.argsort(-covar, axis=1)

            #Find indices and correlations of the k-nearest neighbouring features
            k_indices = np.zeros(covar[0].shape)
            k_distances = np.zeros(covar[0].shape)

            #Find the feature which is the k-nearest neighbour k_indices[i] of feature i
            #Then find the actual correlation between feature i and neighbour k_indices[i]
            for i in range(covar[0].shape[0]):
                k_indices[i] = ind[i][k]
                k_distances[i] = covar[i][ind[i][k]]

            #Find the feature max_k_ind for which dissimilarity is minimal (correlation is maximal?)
            max_k_ind = np.argsort(-k_distances)[0]
            
            if self.verbose:
                print("Comparing %s-nearest neighbours." % str(k))
                print("The correlation with the K-nearest neighbour is the greatest for feature %s, whose neighbour is feature %s." % (str(max_k_ind), str(int(k_indices[max_k_ind]))))
                print("This correlation equals %s." % str(k_distances[max_k_ind]))

            #if no remaining features are more similar than the threshold; do nothing
            #otherwise, remove features that are more similar to their neighbour than the threshold
            if k_distances[max_k_ind] > threshold:
                threshold = k_distances[max_k_ind]
                to_be_removed = ind[max_k_ind][:k]
                if self.verbose:
                    print("This will be the new threshold.")
                    print("Removing features %s" % str(list(to_be_removed))[1:-1])            
                data.drop(data.columns[to_be_removed],axis=1,inplace=True)
            else:
                if self.verbose:
                    print("This is lower than the threshold; continuing...")
            #Lower k by 1
            k = k - 1
            
        confirmed = data.columns
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        
        # notify user
        if self.verbose:
            print("%s features left." % str(self.n_features_))
                  
        return self

    def _transform(self, X, return_df=False):
        # sanity check
        try:
            self.support_
        except AttributeError:
            raise ValueError('Method needs to be fitted first.')

        indices = self.support_

        if return_df:
            X = X.iloc[:, indices]
        else:
            X = X[:, indices]
        return X

    def _check_params(self):
        """
        Check hyperparameters before proceeding with fit.
        """
        if self.k_initial <= 0 :
            raise ValueError('Initial K should be 1 or more.')
