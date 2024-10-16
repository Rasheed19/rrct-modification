from .utils import rrct


class RRCTFeatureSelection:
    """
    Class that cast the implementation of rrct into a class.
    -------------------------------------------------------

    Args:
    ----
            K:             a non-negative integer specifying the number of features to be selected,
                           default is K=None, which means all features will be selected
            scale_feature: whether features have been scaled or not

    Methods:
    ------
            apply:          apply the algorithm to a given training set
            apply_select:   apply the algorithm and select features from the training set
            select:         select features from the training set based on the fitted rrct object
    """

    # Method for variable initializations
    def __init__(self, K=None, scale_feature=False):
        self.K = K
        self.scale_feature = scale_feature
        self.selected_features_indices_ = None
        self.rrct_values_ = None

    # In case of printing the created object, this will be displayed
    def __str__(self):
        return f"RRCTFeatureSelection(K={self.K}, scale_feature={self.scale_feature}) object"

    def apply(self, X, y, verbose=0):
        """
        A function that applies the rrct algorithm to a training set X, y.
        -----------------------------------------------------------------

        Args:
        ----
             X:       a numpy array of shape N by M
             y:       a numpy array of shape N by 1
             verbose: non-negative integer, controls the verbosity of output

        Returns:
        -------
                self (for the purpose of method cascading)
        """
        self.selected_features_indices_, self.rrct_values_ = rrct(
            X=X, y=y, K=self.K, scale_feature=self.scale_feature, verbose=verbose
        )

        return self

    def select(self, X):
        """
        A function that selects features from a given design
        matrix X based on the results of the application of
        rrct algorithm.
        ----------------------------------------------------

        Args:
        ----
             X: design matrix; numpy array of shape N by M

        Returns:
        -------
                Design matrix with selected features
        """

        return X[:, self.selected_features_indices_]

    def apply_select(self, X, y, verbose=0):
        """
        A function that applies rrct algorithm to a given
        training set X, y and then select features from X.
        --------------------------------------------------

        Args:
        ----
             X:       a numpy array of shape N by M
             y:       a numpy array of shape N by 1
             verbose: non-negative integer, controls the verbosity of output

        Returns:
        -------
                Design matrix with selected features
        """
        self.selected_features_indices_, self.rrct_values_ = rrct(
            X=X, y=y, K=self.K, scale_feature=self.scale_feature, verbose=verbose
        )

        return X[:, self.selected_features_indices_]
