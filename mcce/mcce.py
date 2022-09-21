import re
import sys
import numpy as np
import pandas as pd
import time

from .cart import CARTMethod
from .sample import SampleMethod
from .metrics import distance

METHODS_MAP = {'cart': CARTMethod, 'sample': SampleMethod}

class MCCE:
    """
    MCCE class that is used to fit the decision trees, sample from the end nodes, and preprocess the samples to output desired counterfactuals.
    
    Parameters
    ----------
    dataset : mcce.Data
        Object that stores the data as a pd.DataFrame and attributes about the data.
    model : 
         Object with a predict() function used in the postprocess step of MCCE.
    seed : int

    Methods
    -------
    fit :
        Fit the decision trees iteratively, starting with only the immutables.
    generate : 
        Samples observations from the leaf nodes of the fitted decision trees based on a set of feature values.
    postprocess :
        Removes the samples that would not serve as good counterfactual explanations. 
    calculate_metrics :
        Calculates the distance metrics between the potential counterfactuals and the original factual observations.
    
    
    """
    def __init__(self,
                 dataset,
                 model,
                 seed=1
                 ):

        self.dataset = dataset
        self.continuous = dataset.continuous
        self.categorical = dataset.categorical
        self.immutables = dataset.immutables

        self.seed = seed
        self.model = model

        self.method = None
        self.visit_sequence = None
        self.predictor_matrix = None

        if hasattr(dataset, 'categorical_encoded'):
            self.categorical_encoded = dataset.categorical_encoded
        else:
            # Get the new categorical feature names after encoding
            self.categorical_encoded = dataset.encoder.get_feature_names(self.categorical).tolist()

        if hasattr(dataset, 'immutables_encoded'):
            self.immutables_encoded = dataset.immutables_encoded
        else:
            # Get the new immutable feature names after encoding
            immutables_encoded = []
            for immutable in self.immutables:
                if immutable in self.categorical:
                    for new_col in self.categorical_encoded:
                        match = re.search(immutable, new_col)
                        if match:
                            immutables_encoded.append(new_col)
                else:
                    immutables_encoded.append(immutable)

            self.immutables_encoded = immutables_encoded

        if not hasattr(self.model, "predict"):
            sys.exit("model does not have predict function.")
        

    def fit(self, 
            df, 
            dtypes):
        """
        Fit the decision trees iteratively, starting with only the immutables.

        Parameters
        ----------
        df : pd.DataFrame
            Training data. Does not include response/target feature.
        dtypes : dict
            Dictionary containing the data types of each feature in df.  

        Returns
        -------
        
        """

        self.df_columns = df.columns.tolist()
        self.n_df_rows, self.n_df_columns = np.shape(df)
        self.df_dtypes = dtypes
        self.mutable_features = [col for col in self.df_columns if (col not in self.immutables_encoded)]
        self.continuous = [feat for feat in dtypes.keys() if dtypes[feat] != 'category']

        self.n_immutables, self.n_mutable = len(self.immutables_encoded), len(self.mutable_features)
        
        # column indices of mutable features
        self.visit_sequence = [index for index, col in enumerate(self.df_columns) if (col in self.immutables_encoded)] # if (col in self.mutable_features)
        for index, col in enumerate(self.df_columns):
            if col in self.mutable_features:
                self.visit_sequence.append(index)

        # convert indices to column names
        self.visit_sequence = [self.df_columns[i] for i in self.visit_sequence]

        self.visited_columns = [col for col in self.df_columns if col in self.visit_sequence]
        self.visit_sequence = pd.Series([self.visit_sequence.index(col) for col in self.visited_columns], index=self.visited_columns)

        # create list of methods to use - currently only cart implemented
        self.method = []
        for col in self.visited_columns:
            if col in self.immutables_encoded:
                self.method.append('sample') # these will be fit but not sampled 
            else:
                self.method.append('cart')
        self.method = pd.Series(self.method, index=self.df_columns)

        # predictor_matrix_validator:
        self.predictor_matrix = np.zeros([len(self.visit_sequence), len(self.visit_sequence)], dtype=int)
        self.predictor_matrix = pd.DataFrame(self.predictor_matrix, index=self.visit_sequence.index, columns=self.visit_sequence.index)
        visited_columns = []
        for col, _ in self.visit_sequence.sort_values().iteritems():
            self.predictor_matrix.loc[col, visited_columns] = 1
            visited_columns.append(col)
        
        # fit
        self._fit(df)

    def _fit(self, 
             df):
        
        # Remember to change this back when not testing!
        max_depth = None # 5

        self.saved_methods = {}
        self.trees = {}
        self.fitted_model = {}
        self.predicted_leaves = {}

        # Train
        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()
        for col, _ in self.visit_sequence.sort_values().iteritems():
            
            # Initialise the method
            col_method = METHODS_MAP[self.method[col]](dtype=self.df_dtypes[col],
                                                       minibucket=5,
                                                       max_depth=max_depth, 
                                                       random_state=self.seed)
            
            # Fit tree for each mutable feature
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            col_method.fit(X_df=df[col_predictors], y_df=df[col])

            # save the method
            if self.method[col] == 'cart':
                self.trees[col] = col_method.leaves_y_dict
                self.fitted_model[col] = col_method.fitted_model
                self.predicted_leaves[col] = col_method.fitted_model.apply(df[col_predictors])

            self.saved_methods[col] = col_method

    def generate(self, 
                 test_factual, 
                 k=10000):
        """
        Samples observations from the leaf nodes of the fitted decision trees based on a set of feature values.

        Parameters
        ----------
        test_factual : pd.DataFrame
            DataFrame containing the test observations to generate counterfactuals for.
        k : int, default: 100
            The number of observations to sample per leaf node for each test observation. 

        Returns
        -------
        pd.DataFrame :
            Contains k generated counterfactual explanations for each test observation.
        """
        self.k = k
        n_test = test_factual.shape[0]

        # Create data set with the immutables repeated k times
        synth_df = test_factual[self.immutables_encoded]
        synth_df = pd.concat([synth_df] * self.k)
        synth_df.sort_index(inplace=True)

        # Repeat 0 for mutable features k times
        synth_df_mutable = pd.DataFrame(data=np.zeros([self.k * n_test, self.n_mutable]), columns=self.mutable_features, index=synth_df.index)

        synth_df = pd.concat([synth_df, synth_df_mutable], axis=1)
        for col in self.mutable_features:
            # Reload the fitted tree
            col_method = self.saved_methods[col]
            
            # Predict with the fitted tree
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            
            synth_df[col] = col_method.predict(synth_df[col_predictors]) 
            
            # Map feature to original dtype
            synth_df[col] = synth_df[col].astype(self.df_dtypes[col])
            
        # Return in same ordering as original dataframe
        synth_df = synth_df[test_factual.columns]
        self.synth_df = synth_df
        return synth_df

    def postprocess(self, 
                    cfs, 
                    test_factual, 
                    cutoff=0.5, 
                    higher_cardinality=False):
        """
        Postprocess the sampled observations to get exactly one counterfactual explanation per test observation.

        Parameters
        ----------
        cfs : pd.DataFrame
            DataFrame containing the potential counterfactual explanations.
        test_factual : pd.DataFrame
            DataFrame containing the test observations to generate counterfactuals for.
        cutoff : float, default: 0.5
            Cutoff value that indicates which observations get a positive/desired response and which do not. 
            Observations with a prediction greater than this cutoff value are considered "positive"
            and those with a lower prediction are considered "negative".
        higher_cardinality : bool, default: False
            Whether the categorical features are allowed to have more than two levels each. 
        Returns
        -------
        
        """
        cols = cfs.columns.to_list()
        self.cutoff = cutoff

        # TO DO: Take this out of postprocess since it's not necessary
        # Calculate the mean number of unique samples per test obs
        # synth_un = self.synth_df.reset_index()
        # synth_un = synth_un.drop_duplicates()
        # synth_un = synth_un.set_index(synth_un['index'])
        # synth_un = synth_un.drop(columns=['index'])
        # synth_un_per_test = synth_un.groupby(synth_un.index).size()
        # synth_un_per_test = pd.DataFrame(synth_un_per_test, columns=['nb_unique_samples'])

        # Predict response of generated data
        cfs_positive = cfs[self.model.predict(cfs) >= cutoff]
        
        # We have to do this whole dance to drop duplicates in the same index
        # But not across indices -- there must be a better way!
        cfs_positive = cfs_positive.reset_index()
        cfs_positive = cfs_positive.drop_duplicates()
        cfs_positive = cfs_positive.set_index(cfs_positive['index'])
        cfs_positive = cfs_positive.drop(columns=['index'])
        self.cfs_positive = cfs_positive
        
        # Duplicate original test observations N times where N is number of positive counterfactuals
        n_counterfactuals = cfs_positive.groupby(cfs_positive.index).size()
        n_counterfactuals = pd.DataFrame(n_counterfactuals, columns=['nb_unique_pos'])

        fact_repeated = test_factual.copy()
        
        fact_repeated = fact_repeated.join(n_counterfactuals)
        fact_repeated.dropna(inplace = True)

        fact_repeated = fact_repeated.reindex(fact_repeated.index.repeat(fact_repeated['nb_unique_pos']))
        fact_repeated.drop(['nb_unique_pos'], axis=1, inplace=True)
        
        self.fact_repeated = fact_repeated

        self.results = self.calculate_metrics(cfs=cfs_positive, 
                                              test_factual=self.fact_repeated, 
                                              higher_cardinality=higher_cardinality) 
        
        # Find the best sample for each test obs
        results_sparse = pd.DataFrame(columns=self.results.columns)

        for idx in list(set(self.results.index)):
        
            idx_df = self.results.loc[idx]
            if(isinstance(idx_df, pd.DataFrame)): # If you have multiple rows
                sparse = min(idx_df.L0) # Find least number of features changed
                sparse_df = idx_df[idx_df.L0 == sparse] 
                closest = min(sparse_df.L1) # Find smallest Gower distance
                close_df = sparse_df[sparse_df.L1 == closest].head(1)

            else: # If you have only one row - return that row
                close_df = idx_df.to_frame().T
                
            results_sparse = pd.concat([results_sparse, close_df], axis=0)
        
        # Add the number of positive instances per test observation
        results_sparse = results_sparse.merge(n_counterfactuals, left_index=True, right_index=True)
        #  Add the number of unique instances per test observation
        # results_sparse = results_sparse.merge(synth_un_per_test, left_index=True, right_index=True)

        cols = cols + ['nb_unique_pos'] # , 'nb_unique_samples'
        self.results_sparse = results_sparse[cols]
        

    def calculate_metrics(self, 
                          cfs, 
                          test_factual, 
                          higher_cardinality):
        """
        Calculate the distance between the potential counterfactuals and the original factuals.

        Parameters
        ----------
        cfs : pd.DataFrame
            DataFrame containing the potential counterfactual explanations.
        test_factual : pd.DataFrame
            DataFrame containing the test observations to generate counterfactuals for.
        higher_cardinality : bool, default: False
            Whether the categorical features are allowed to have more than two levels each. 
        Returns
        -------
            pd.DataFrame containing the three distances:
            L0: sparsity = the number of features changed 
            L1: Manhattan distance
            L2: Euclidean distance
        
        """

        features = cfs.columns.to_list()
        cfs.sort_index(inplace=True)

        cfs_metrics = cfs.copy()
        
        # Calculate sparsity and Euclidean distances
        factual = test_factual[features]
        counterfactuals = cfs[features]
        
        distances = pd.DataFrame(distance(counterfactuals, factual, self.dataset, higher_cardinality), index=factual.index)
        cfs_metrics = pd.concat([cfs_metrics, distances], axis=1)

        return cfs_metrics
    