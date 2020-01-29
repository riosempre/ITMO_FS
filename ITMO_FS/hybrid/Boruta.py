from __future__ import print_function, division
import numpy as np
import random
import scipy as sp
from sklearn.model_selection import train_test_split


class Boruta(BaseEstimator, TransformerMixin):
    """
    Python-based implementation of the Boruta R method.
    
    The Boruta method is based on tree-based methods, such as Random Forest,
    which are capable of calculating feature importance.
    
    Boruta creates "shadow features" which have the same distribution as
    the actual features but do not correlate with anything. The highest 
    feature importance of these shadow features is then used as a gauge
    for determining which features are relevant.
    
    Boruta is not a minimal-optimal method, and finds all relevant features,
    not just the minimal poptimal collection. It should therefore be used
    when a larger amount of features has to be saved.
    

    Parameters
    ----------
    estimator : object
        A supervised learning estimator. Must be able to return feature importance.
    n_estimators : int, default = 1000
        The number of estimators in the chosen ensemble method.
    perc : int, default = 100
        Percentage of maximum shadow feature importance which needs to be surpassed in order to be considered relevant.
    alpha : float, default = 0.05
        Level at which the corrected p-values will get rejected during both
        correction steps.
    max_iter : int, default = 100
        The number of maximum iterations to perform.
    verbose : bool, default=False
        Controls verbosity of output.
    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of selected features - only confirmed ones are True.
    support_unclear_ : array of shape [n_features]
        The mask of selected unclear features, which haven't gained enough
        support during the max_iter number of iterations..
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1 and unclear features are assigned
        rank 2.

    References
    ----------
    [1] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package"
        Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010
    """

    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05,
                 max_iter=100, verbose=False):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fits the Boruta feature selection.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input.
        y : array-like, shape = [n_samples]
            The target values.
        """

        return self._fit(X, y)

    def transform(self, X, remove_unclear=False, return_df=False):
        """
        Removes the features which have not been selected by Boruta.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input.
        remove_unclear: boolean, default = False
            Remove unclear features if true.        
        return_df : boolean, default = False
            Returns a Pandas dataframe if true.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected by Boruta.
        """

        return self._transform(X, remove_unclear, return_df)

    def fit_transform(self, X, y, remove_unclear=False, return_df=False):
        """
        Fits Boruta, then removes irrelevant features.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input.
        y : array-like, shape = [n_samples]
            The target values.
        remove_unclear: boolean, default = False
            Remove unclear features if true.        
        return_df : boolean, default = False
            Returns a Pandas dataframe if true.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected by Boruta.
        """

        self._fit(X, y)
        return self._transform(X, remove_unclear, return_df)

    def score(self, X, y, method, remove_unclear=False):
        """
        Tests Boruta by comparing the performance of a learning method before and after feature selection
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
        method.fit(X_train, y_train.ravel())
        

        X_filtered_train = self.transform(X_train, remove_unclear = remove_unclear)
        X_filtered_test = self.transform(X_test, remove_unclear = remove_unclear)
        method_filtered = method.clone()
        method_filtered.fit(X_filtered_train, y_train.ravel())

        score = method.score(X_test, y_test.ravel())
        score_filtered = method.score(X_filtered_test, y_test.ravel())

        return score_filtered / score
    def _validate_pandas_input(self, arg):
        try:
            return arg.values
        except AttributeError:
            raise ValueError(
                "input needs to be a numpy array or pandas data frame."
            )

    def _fit(self, X, y):
        # check input params
        self._check_params(X, y)

        if not isinstance(X, np.ndarray):
            X = self._validate_pandas_input(X) 
        if not isinstance(y, np.ndarray):
            y = self._validate_pandas_input(y)

        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # dec_reg checks the status of each feature; hit_reg checks the times the feature was rated as important
        dec_reg = np.zeros(n_feat, dtype=np.int)
        hit_reg = np.zeros(n_feat, dtype=np.int)
        impor_history = np.zeros(n_feat, dtype=np.float)
        sha_max_history = []

        # set number of estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            if self.n_estimators == 'auto':
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators=n_tree)

            # add shadow attribute
            cur_imp = self._shadowdow_features(X, y, dec_reg)
            impor_shadow_max = np.percentile(cur_imp[1], self.perc)
            
            # record  history
            sha_max_history.append(impor_shadow_max)
            impor_history = np.vstack((impor_history, cur_imp[0]))

            hit_reg = self._assign_hits(hit_reg, cur_imp, impor_shadow_max)

            dec_reg = self._testing(dec_reg, hit_reg, _iter)

            if self.verbose == True and _iter < self.max_iter:
                self._output_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1

        confirmed = np.where(dec_reg == 1)[0]
        unclear = np.where(dec_reg == 0)[0]

        unclear_median = np.median(impor_history[1:, unclear], axis=0)
        unclear_confirmed = np.where(unclear_median
                                       > np.median(sha_max_history))[0]
        unclear = unclear[unclear_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_unclear_ = np.zeros(n_feat, dtype=np.bool)
        self.support_unclear_[unclear] = 1

        # ranking
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        self.ranking_[unclear] = 2
        selected = np.hstack((confirmed, unclear))
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        impor_history_rejected = impor_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
                iter_ranks = self._nanrankdata(impor_history_rejected, axis=1)
                rank_medians = np.nanmedian(iter_ranks, axis=0)
                ranks = self._nanrankdata(rank_medians, axis=0)

                if unclear.shape[0] > 0:
                    ranks = ranks - np.min(ranks) + 3
                else:
                    # and 2 otherwise
                    ranks = ranks - np.min(ranks) + 2
                self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=np.bool)

        # notify user
        if self.verbose == True:
            self._output_results(dec_reg, _iter, 1)
        return self

    def _transform(self, X, remove_unclear=False, return_df=False):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('Method needs to be fitted first.')

        if not remove_unclear:
            indices = self.support_ + self.support_unclear_
        else:
            indices = self.support_

        if return_df:
            X = X.iloc[:, indices]
        else:
            X = X[:, indices]
        return X

    def _get_tree_num(self, n_feat):
        depth = self.estimator.get_params()['max_depth']
        if depth == None:
            depth = 10
        # how many times feature should be considered on average
        f_repr = 100
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _get_imp(self, X, y):
        try:
            self.estimator.fit(X, y)
        except Exception as e:
            raise ValueError('''Please check your X and y variable. The provided 
                              estimator cannot be fitted to your data.\n''' + str(e))
        try:
            imp = self.estimator.feature_importances_
        except Exception:
            raise ValueError('''Only methods with feature_importance_ attribute 
                             are currently supported in BorutaPy.''')
        return imp

    def _get_shuffle(self, seq):
        random.shuffle(seq)
        return seq

    def _shadowdow_features(self, X, y, dec_reg):
        # find features that are unclear still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = np.copy(X[:, x_cur_ind])
        x_cur_w = x_cur.shape[1]
        x_shadow = np.copy(x_cur)
        while (x_shadow.shape[1] < 5):
            x_shadow = np.hstack((x_shadow, x_shadow))
        x_shadow = np.apply_along_axis(self._get_shuffle, 0, x_shadow)
        imp = self._get_imp(np.hstack((x_cur, x_shadow)), y)
        impor_shadow = imp[x_cur_w:]
        impor_real = np.zeros(X.shape[1])
        impor_real[:] = np.nan
        impor_real[x_cur_ind] = imp[:x_cur_w]
        return impor_real, impor_shadow

    def _assign_hits(self, hit_reg, cur_imp, impor_shadow_max):
        # register hits for features that did better than the best of shadows
        cur_impor_no_nan = cur_imp[0]
        cur_impor_no_nan[np.isnan(cur_impor_no_nan)] = 0
        hits = np.where(cur_impor_no_nan > impor_shadow_max)[0]
        hit_reg[hits] += 1
        return hit_reg

    def _testing(self, dec_reg, hit_reg, _iter):
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        
        # get uncorrected p values based on hit_reg
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()

        # bonferroni correction
        to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))
        to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))

        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        return dec_reg

    def _nanrankdata(self, X, axis=1):
        """
        Replaces bottleneck's nanrankdata with scipy and numpy alternative.
        """
        ranks = sp.stats.mstats.rankdata(X, axis=axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _check_params(self, X, y):
        """
        Check hyperparameters as well as X and y before proceeding with fit.
        """
        # check X and y are consistent len
        if len(X) != len(y):
            raise ValueError('X does not match Y in shape. Check the values.')

        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('Alpha should be between 0 and 1.')

    def _output_results(self, dec_reg, _iter, flag):
        n_iter = str(_iter) + ' / ' + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ['Step: ', 'Confirmed: ', 'Unclear: ', 'Rejected: ']

        # still in feature selection
        if flag == 0:
            n_unclear = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_unclear, n_rejected])
            if self.verbose:
                output = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])

        # Boruta finished running and unclears have been filtered
        else:
            n_unclear = np.sum(self.support_unclear_)
            content = map(str, [n_iter, n_confirmed, n_unclear, n_rejected])
            result = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
            output = "\n\nProcess finished.\n\n" + result
        print(output)
