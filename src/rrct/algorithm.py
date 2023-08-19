import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pingouin as pg
import time


def rrct(
    X: np.ndarray,
    y: np.ndarray,
    K: int = None,
    scale_feature: bool = False,
    verbose: int = 0
) -> tuple:
    """
    Function for feature selection (FS), based on the concept of 
    RELEVANCE, REDUNDANCY AND COMPLEMENTARITY TRADE-OFF (RRCT).

    This function is based on solid theoretical principles, integrating the
    key concepts for feature relevance, redundancy, and conditional
    relevance (complementarity).

    Originally developed by A. Tsanas: tsanasthanasis@gmail.com.

    Modified by R. Ibraheem: R.O.Ibraheem@sms.ed.ac.uk (ibraheem.abdulrasheed@gmail.com)

    Args:
    ----
            X:             a numpy array of shape N by M
            y:             a numpy array of shape N by 1
            K:             a non-negative integer specifying the number of features to be selected,
                           default is k=None, which means all features will be selected
            scale_feature: whether features have been scaled or not 
            verbose:       non-negative integer, controls the verbosity of output

    Returns:
    -------
            a tuple containing the indices of selected features and dictionary of rrct metrics.

    """

    M = np.shape(X)[1]

    # guard against the case the user has indicated more features 
    # than the dimensionality of the dataset (or lessthan 1)
    if K is None:
        K = M

    elif K > M:
       raise ValueError(
           'You provided K>M, i.e. more features to be ranked than the dimensionality of the data'
        )
    
    elif K == 0:
        raise ValueError(
           'You provided K=0, i.e. no features to be ranked; valid values are positive intergers > 0 of set K = None to rank all features'
        )
    
    elif K < 0:
        raise ValueError(
           'K must be None or positive integer'
        )
    
    elif not isinstance(K, int):
        raise TypeError(
           'K must be positive integer or set to None'
        )

    # Get Code Block Start Time
    start_time = time.perf_counter()

    # Data preprocessing
    if scale_feature:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Converting array to pandas DataFrame; makes
    #  processing simpler working with pandas
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)

    data_Xy = X_df
    data_Xy = data_Xy.assign(y=y_df)  # Dataframe bringing together X and y

    # Compute relevance, redundancy and build the mRMR matrix  
    relv = X_df.corrwith(
        y_df.squeeze(),
        method="spearman"
    )  # correlation coefficient for each of the variables in X with y: squeeze
    
    # Dataframe into Series
    relevance = -0.5 * np.log(1 - relv ** 2)  # **** RELEVANCE BY DEFINITION

    redundancy = X_df.corr(method="spearman")  # compute correlation matrix using Spearman's method
    redundancy = redundancy - np.identity(M)
    redundancy = -0.5 * np.log(1 - redundancy ** 2)  # **** REDUNDANCY BY DEFINITION

    # Define convenient matrix
    mRMR_matrix = redundancy
    mRMR_matrix = mRMR_matrix + np.diag(relevance)

    # Initialize vectors that will hold the relevance, 
    # redundancy and complementarity values  
    RRCT_metric = np.zeros(K)
    RRCT_all_relevance = np.zeros(K)
    RRCT_all_redundancy = np.zeros(K)
    RRCT_all_complementarity = np.zeros(K)
    features = np.zeros(K, dtype='int64')

    RRCT_metric[0] = relevance.max(skipna=True)
    RRCT_all_relevance[0] = RRCT_metric[0]
    RRCT_all_redundancy[0] = 0
    RRCT_all_complementarity[0] = 0
    features[0] = relevance.idxmax(skipna=True)

    Z = [features[0]]

    # Main loop to obtain the feature subset
    for k in range(1, K):

        candidates = pd.Series(np.arange(M))
        candidates = candidates.drop(Z)  # potential candidates at this step

        mean_redundancy = np.mean(
            mRMR_matrix.iloc[candidates, Z],
            axis=1
        )  # compute average redundancy for the kth step

        df_agg = []
        for i in candidates.index:
            comple_pair = pg.partial_corr(
                data=data_Xy,
                x=i,
                y='y',
                covar=Z,
                method='spearman'
            )  # the controlling variables imm0rtal99 must be in list format
            df_agg.append(comple_pair)

        df_ngram = pd.concat(df_agg, ignore_index=True)
        complementarity = df_ngram.r  # recover all pair partial correlation computations, in Series format

        Csign = np.sign(complementarity.values) * np.sign(complementarity.values - relv[candidates])
        complementarity = -0.5 * np.log(1 - (complementarity) ** 2)
        complementarity.index = Csign.index

        # RRCT heart: max relevance - min redundancy + complementarity optimization
        RRCT_heart = relevance[candidates] - mean_redundancy + Csign * complementarity.values
        RRCT_metric[k] = RRCT_heart.max();
        fs_idx = RRCT_heart.idxmax()

        features[k] = fs_idx
        Z.append(features[k])  # used to condition upon when computing 
                               # the partial correlations (in subsequent for-loop steps)

        # Store the three elements: Relevance, Redundancy, Complementarity
        RRCT_all_relevance[k] = relevance[features[k]]
        RRCT_all_redundancy[k] = mean_redundancy[fs_idx] 
        RRCT_all_complementarity[k] = Csign.get(key=fs_idx) * complementarity.get(key=fs_idx)

    # Recover outputs of the function; note that
    # each metric is calculated relative
    # to the features in the feature subset at each iteration;
    # i.e., with respect to the candidates at each iteration
    RRCT_all = dict([
        ('relevance', RRCT_all_relevance),
        ('redundancy', RRCT_all_redundancy),
        ('complementarity', RRCT_all_complementarity), 
        ('RRCT_metric', RRCT_metric)
    ])

    # Get Code Block End Time
    if verbose > 0:
        end_time = time.perf_counter()
        print(f"Start Time : {start_time}")
        print(f"End Time : {end_time}")
        print(f"Total Execution Time : {end_time - start_time:0.4f} seconds\n")

    return features, RRCT_all


class RRCTFeatureSelection:
    """
    Class that cast the implementation of rrct into a class.
    -------------------------------------------------------

    Args:
    ----
            K:             a non-negative integer specifying the number of features to be selected,
                           default is k=None, which means all features will be selected
            scale_feature: whether features have been scaled or not 

    Methods:
    ------
            apply:          apply the algorithm to a given training set
            apply_select:   apply the algorithm and select features from the training set
            select:         select features from the training set based on to apply results
    """

    # Method for variable initializations 
    def __init__(self, K=None, scale_feature=False):
        self.K = K
        self.scale_feature = scale_feature
        self.selected_features_indices_ = None
        self.rrct_values_ = None

    # In case of printing the created object, this will be displayed
    def __str__(self):
        return f"RRCTFeatureSelection(k={self.K}) object"

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
            X=X,
            y=y,
            K=self.K,
            scale_feature=self.scale_feature,
            verbose=verbose
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
            X=X,
            y=y,
            K=self.K,
            scale_feature=self.scale_feature,
            verbose=verbose
        )

        return X[:, self.selected_features_indices_]