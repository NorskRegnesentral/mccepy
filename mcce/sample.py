import numpy as np
from .method import Method

NUM_COLS_DTYPES = ['int', 'float', 'datetime']
CAT_COLS_DTYPES = ['category', 'bool']


class SampleMethod(Method):
    """
    Method to sample observations
    
    Parameters
    ----------
    dtype : dict
        Dictionary containing the data types of each feature.
    random_state : bool, default: False
       Whether a random state should be set before the trees are fit.

    Methods
    -------
    fit :
        If the feature is continuous, find their minimum and maximum value
    predict : 
        Sample an observation with replacement for each observation in X_test_df.

    Returns
    -------

    """
    def __init__(self, dtype, random_state=None, *args, **kwargs):
        self.dtype = dtype
        self.random_state = random_state

    def fit(self, y_df=None, *args, **kwargs):
        """
        If the feature is continuous, find their minimum and maximum value
        Parameters
        ----------
        y_df : pd.DataFrame
            DataFrame containing response/target
        Returns
        -------
        """
        if self.dtype in NUM_COLS_DTYPES:
            self.x_real_min, self.x_real_max = np.min(y_df), np.max(y_df)

        self.values = y_df.to_numpy()

    def predict(self, X_test_df):
        """
        Sample an observation with replacement for each observation in X_test_df. 
        Parameters
        ----------
        X_test_df : pd.DataFrame
            DataFrame containing the test observations.
        Returns
        -------
        """
        n = X_test_df.shape[0]

        y_pred = np.random.choice(self.values, size=n, replace=True)
        return y_pred