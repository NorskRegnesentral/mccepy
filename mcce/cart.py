import pandas as pd
import numpy as np
from .method import Method
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


NUM_COLS_DTYPES = ['int', 'float', 'datetime', 'float64']
CAT_COLS_DTYPES = ['category', 'bool']

class CARTMethod(Method):
    """
    Implementation of CART method from Breiman et.al. 1984 [1]_.
    Entirely based on https://github.com/hazy/synthpop. 
    
    Parameters
    ----------
    dtype : dict
        Dictionary containing the data types of each feature.
    minibucket : float
       The minimum number of samples required to be in a leaf node. A split point at any depth will only be considered if its leaves have 
       at least this number of training samples in each of the left and right branches.
    random_state : bool, default: False
       Whether a random state should be set before the trees are fit.

    Methods
    -------
    fit :
        Fit the classification and regression tree.
    predict : 
        Find the end node of each observation 
    
    Returns
    -------

    .. [1] Leo Breiman, Jerome Friedman, Richard Olshen, and Charles Stone. 1984. Classification and regression trees. Chapman and Hall.
    """
    def __init__(self, dtype, minibucket=5, max_depth=None, random_state=None, *args, **kwargs):
        self.dtype = dtype
        self.minibucket = minibucket
        self.max_depth = max_depth
        self.random_state = random_state

        if self.dtype in CAT_COLS_DTYPES:
            self.cart = DecisionTreeClassifier(min_samples_leaf=self.minibucket, max_depth=self.max_depth, random_state=self.random_state)
        if self.dtype in NUM_COLS_DTYPES:
            self.cart = DecisionTreeRegressor(min_samples_leaf=self.minibucket, max_depth=self.max_depth, random_state=self.random_state)

    def fit(self, X_df, y_df):
        """
        Fit CART based on the data type of y_df
        Haven't figured out if we should normalize/one-hot encode the features?!

        Parameters
        ----------
        X_df : pd.DataFrame
            DataFrame containing all training observations. 
        y_df : pd.DataFrame
            DataFrame containing response/target
        Returns
        -------
        """
        X_df, y_df = self.prepare_dfs(X_df=X_df, y_df=y_df, normalise_num_cols=False, one_hot_cat_cols=False)
        
        self.X_df_after_one_hot = X_df

        if self.dtype in NUM_COLS_DTYPES:
            self.y_real_min, self.y_real_max = np.min(y_df), np.max(y_df)

        # Turns out, you get the same trees if X and y are numeric or categorical
        X = X_df.to_numpy()
        y = y_df.to_numpy()
        self.cart.fit(X, y)

        # Save the y distribution wrt trained tree nodes
        leaves = self.cart.apply(X)
        leaves_y_df = pd.DataFrame({'leaves': leaves, 'y': y})
        self.leaves_y_dict = leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).to_dict()
        self.fitted_model = self.cart

    def predict(self, X_test_df):
        """
        Find end node of the tree for each observation in X_test_df
        Parameters
        ----------
        X_test_df : pd.DataFrame
            DataFrame containing all test observations. 
        
        Returns
        -------
        """
        ## NEW: CHANGED ONEHOTCATCOLS to FALSE!!! CHANGES LEVELS OF ONE-HOT ENCODED FEATURES
        X_test_df, _ = self.prepare_dfs(X_df=X_test_df, normalise_num_cols=False, one_hot_cat_cols=False, fit=False)
        
        # Find the leaves and for each test obs and randomly sample from the observed values
        X_test = X_test_df.to_numpy()
        leaves_pred = self.cart.apply(X_test)
        y_pred = np.zeros(len(leaves_pred), dtype=object)

        leaves_pred_index_df = pd.DataFrame({'leaves_pred': leaves_pred, 'index': range(len(leaves_pred))})
        leaves_pred_index_dict = leaves_pred_index_df.groupby('leaves_pred').apply(lambda x: x.to_numpy()[:, -1]).to_dict()
        
        for leaf, indices in leaves_pred_index_dict.items():
            np.random.seed(0) # seed seed so we can reproduce MCCE results
            y_pred[indices] = np.random.choice(self.leaves_y_dict[leaf], size=len(indices), replace=True)
            
        return y_pred
    
    