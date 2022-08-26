import time
import numpy as np
import pandas as pd

from .cart import CARTMethod
from .sample import SampleMethod

from .metrics import distance

METHODS_MAP = {'cart': CARTMethod, 'sample': SampleMethod}

class MCCE:
    def __init__(self,
                 dataset=None,
                 fixed_features=None,
                 fixed_features_encoded=None,
                 model=None,
                 seed=None
                 ):

        # initialize arguments
        self.dataset = dataset
        self.fixed_features = fixed_features  # features to condition on - the ones in the dataset
        self.fixed_features_encoded = fixed_features_encoded
        self.continuous = dataset.continuous
        self.categorical = dataset.categorical

        self.seed = seed
        self.model = model

        self.method = None
        self.visit_sequence = None
        self.predictor_matrix = None
        

    def fit(self, df, dtypes=None):

        self.df_columns = df.columns.tolist()
        self.n_df_rows, self.n_df_columns = np.shape(df)
        self.df_dtypes = dtypes
        self.mutable_features = [col for col in self.df_columns if (col not in self.fixed_features_encoded)]
        self.continuous = [feat for feat in dtypes.keys() if dtypes[feat] != 'category']

        self.n_fixed, self.n_mutable = len(self.fixed_features_encoded), len(self.mutable_features)
        
        # column indices of mutable features
        self.visit_sequence = [index for index, col in enumerate(self.df_columns) if (col in self.fixed_features_encoded)] # if (col in self.mutable_features)
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
            if col in self.fixed_features_encoded:
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

    def _fit(self, df):
        self.saved_methods = {}
        self.trees = {}

        # train
        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()
        for col, _ in self.visit_sequence.sort_values().iteritems():
            # initialise the method
            col_method = METHODS_MAP[self.method[col]](dtype=self.df_dtypes[col], random_state=self.seed)
            
            # fit the method
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            
            col_method.fit(X_df=df[col_predictors], y_df=df[col])
            # save the method
            if self.method[col] == 'cart':
                self.trees[col] = col_method.leaves_y_dict
            self.saved_methods[col] = col_method

    def generate(self, test, k):

        self.k = k
        n_test = test.shape[0]

        # create data set with the fixed features repeated k times
        synth_df = test[self.fixed_features_encoded]
        synth_df = pd.concat([synth_df] * self.k)
        synth_df.sort_index(inplace=True)

        # repeat 0 for mutable features k times
        synth_df_mutable = pd.DataFrame(data=np.zeros([self.k * n_test, self.n_mutable]), columns=self.mutable_features, index=synth_df.index)

        synth_df = pd.concat([synth_df, synth_df_mutable], axis=1)
        start_time = time.time()
        for col in self.mutable_features:
            # reload the method
            col_method = self.saved_methods[col]
            # predict with the method
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            synth_df[col] = col_method.predict(synth_df[col_predictors])
            # map dtype to original dtype
            synth_df[col] = synth_df[col].astype(self.df_dtypes[col])

        self.total_generating_seconds = time.time() - start_time

        # return in same ordering as original dataframe
        synth_df = synth_df[test.columns]
        return synth_df

    def postprocess(self, cfs, fact, cutoff=0.5, higher_cardinality=False):
        
        self.cutoff = cutoff

        # Predict response of generated data
        cfs_positive = cfs[self.model.predict(cfs) >= cutoff]
        
        # Duplicate original test observations N times where N is number of positive counterfactuals
        n_counterfactuals = cfs_positive.groupby(cfs_positive.index).size()
        n_counterfactuals = pd.DataFrame(n_counterfactuals, columns = ['N'])

        fact_repeated = fact.copy()

        fact_repeated = fact_repeated.join(n_counterfactuals)
        fact_repeated.dropna(inplace = True)

        fact_repeated = fact_repeated.reindex(fact_repeated.index.repeat(fact_repeated.N))
        fact_repeated.drop(['N'], axis=1, inplace=True)

        self.fact_repeated = fact_repeated

        self.results = self.calculate_metrics(cfs=cfs_positive, 
                                              fact=self.fact_repeated, 
                                              higher_cardinality=higher_cardinality) 

        # Find the best row for each test obs
        time1 = time.time()
        results_sparse = pd.DataFrame(columns=self.results.columns)

        for idx in list(set(self.results.index)):
            idx_df = self.results.loc[idx]
            if(isinstance(idx_df, pd.DataFrame)): # If you have multiple rows
                sparse = min(idx_df.L0) # 1) find least # features changed
                sparse_df = idx_df[idx_df.L0 == sparse] 
                closest = min(sparse_df.L2) # find smallest Gower distance
                close_df = sparse_df[sparse_df.L2 == closest].head(1)

            else: # if you have only one row - return that row
                close_df = idx_df.to_frame().T
                
            results_sparse = pd.concat([results_sparse, close_df], axis=0)
        
        time2 = time.time()
        self.find_best_cpu_time = time2 - time1
        self.results_sparse = results_sparse


    def calculate_metrics(self, cfs, fact, higher_cardinality=False): # inverse_transform

        time1 = time.time()
        features = cfs.columns.to_list()
        cfs.sort_index(inplace=True)

        cfs_metrics = cfs.copy()
        
        # Calculate sparsity and Euclidean distance
        factual = fact[features]
        counterfactuals = cfs[features]
        
        distances = pd.DataFrame(distance(counterfactuals, factual, self.dataset, higher_cardinality), index=factual.index)
        cfs_metrics = pd.concat([cfs_metrics, distances], axis=1)
        time2 = time.time()
        self.distance_cpu_time = time2 - time1

        return cfs_metrics
    