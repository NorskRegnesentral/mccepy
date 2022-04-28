import time
import numpy as np
import pandas as pd

from cart import CARTMethod
from sample import SampleMethod
# from metrics import get_delta, d1_distance, d2_distance, d3_distance, yNN, feasibility, redundancy, constraint_violation
from sklearn.neighbors import NearestNeighbors


METHODS_MAP = {'cart': CARTMethod, 'sample': SampleMethod}

class MCCE:
    def __init__(self,
                 fixed_features=None,
                 model=None,
                 scaler=None,
                 seed=None):

        # initialise arguments
        self.fixed_features = fixed_features  # features to condition on
        self.seed = seed
        self.model = model  # predictive model
        self.scaler = scaler #  use to normalize/scale continuous features
        
        self.method = None
        self.visit_sequence = None
        self.predictor_matrix = None
        

    def fit(self, df, dtypes=None):

        self.df_columns = df.columns.tolist()
        self.n_df_rows, self.n_df_columns = np.shape(df)
        self.df_dtypes = dtypes
        self.mutable_features = [col for col in self.df_columns if (col not in self.fixed_features)]
        self.cont_feat = [feat for feat in dtypes.keys() if dtypes[feat] != 'category']

        self.n_fixed, self.n_mutable = len(self.fixed_features), len(self.mutable_features)
        
        # column indices of mutable features
        self.visit_sequence = [index for index, col in enumerate(self.df_columns) if (col in self.fixed_features)] # if (col in self.mutable_features)
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
            if col in self.fixed_features:
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

        # train
        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()
        for col, _ in self.visit_sequence.sort_values().iteritems():
            # initialise the method
            col_method = METHODS_MAP[self.method[col]](dtype=self.df_dtypes[col], random_state=self.seed)
            
            # fit the method
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            
            col_method.fit(X_df=df[col_predictors], y_df=df[col])
            # save the method
            self.saved_methods[col] = col_method

    def generate(self, test, k):

        self.k = k
        n_test = test.shape[0]

        # create data set with the fixed features repeated k times
        synth_df = test[self.fixed_features]
        synth_df = pd.concat([synth_df] * self.k)
        synth_df.sort_index(inplace=True)

        # repeat 0 for mutable features k times
        synth_df_mutable = pd.DataFrame(data=np.zeros([self.k * n_test, self.n_mutable]), columns=self.mutable_features, index=synth_df.index)

        synth_df = pd.concat([synth_df, synth_df_mutable], axis=1)

        start_time = time.time()
        for col in self.mutable_features: #self.visit_sequence.sort_values().iteritems():
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

    def postprocess(self, data, synth, test, response, scaler=None, cutoff=0.5):

        # predict response of generated data
        synth[response] = self.model.predict(synth)
        synth_positive = synth[synth[response]>=cutoff] # drop negative responses
        
        
        # duplicate original test observations N times where N is number of positive counterfactuals
        n_counterfactuals = synth_positive.groupby(synth_positive.index).size()
        n_counterfactuals = pd.DataFrame(n_counterfactuals, columns = ['N'])

        test = test.join(n_counterfactuals)
        test.dropna(inplace = True)

        test = test.reindex(test.index.repeat(test.N))
        test.drop(['N'], axis=1, inplace=True)

        self.test_repeated = test

        self.results = self.calculate_metrics(synth_positive, self.test_repeated, data, self.model, response, scaler) # SOMETHING BREAKS HERE
        
        # find the best row for each test obs
        results_sparse = pd.DataFrame(columns=self.results.columns)
        # print(results_sparse)
        for idx in list(set(self.results.index)):
            idx_df = self.results.loc[idx]
            if(isinstance(idx_df, pd.DataFrame)): # if you have multiple rows
                sparse = min(idx_df.L0)
                sparse_df = idx_df[idx_df.L0 == sparse]
                closest = min(sparse_df.L2)
                close_df = sparse_df[sparse_df.L2 == closest]
                if(close_df.shape[0]>1):
                    highest_feasibility = max(close_df.feasibility)
                    close_df = close_df[close_df.feasibility == highest_feasibility].head(1)

            else: # if you have only one row
                close_df = idx_df.to_frame().T
                
            # print(sparse)
            # print(close_df)
            results_sparse = pd.concat([results_sparse, close_df], axis=0)

        self.results_sparse = results_sparse

    def calculate_metrics(self, synth, test, data, model, response, scaler):

        features = synth.columns.to_list()
        features.remove(response)

        synth_metrics = synth.copy()

        # 1) Distance: Sparsity and Euclidean distance
        factual = test[features].sort_index().to_numpy()
        counterfactuals = synth[features].sort_index().to_numpy()

        delta = factual - counterfactuals # get_delta(factual, counterfactuals)

        # d1 = d1_distance(delta) # sparsity
        # d2 = d2_distance(delta) # sparsity
        # d3 = d3_distance(delta) # euclidean distance

        d1 = np.sum(delta != 0, axis=1, dtype=float).tolist() # sparsity
        d2 = np.sum(np.abs(delta), axis=1, dtype=float).tolist() # manhatten distance
        d3 = np.sum(np.square(np.abs(delta)), axis=1, dtype=np.float).tolist() # euclidean distance

        synth_metrics['L0'] = d1
        synth_metrics['L1'] = d2
        synth_metrics['L2'] = d3

        # 3) kNN
        # neighb = yNN(data, synth, response, y=5)
        # synth_metrics['yNN'] = neighb


        # 4) Feasibility 
        # feas = feasibility(data, synth, response, y=5)
        # synth_metrics['feasibility'] = feas

        cols = data.columns
        cols.drop(response)

        feas_results = []
        nbrs = NearestNeighbors(n_neighbors=5).fit(synth[cols].values)

        for i, row in synth[cols].iterrows():
            knn = nbrs.kneighbors(row.values.reshape((1, -1)), 5, return_distance=True)[0]
            
            feas_results.append(np.mean(knn))

        synth_metrics['feasibility'] = feas_results


        # 5) Redundancy 
        # redund = redundancy(synth, test, model, response)
        # synth_metrics['redundancy'] = redund

        # 6) Success
        synth_metrics['success'] = 1

        # 7) Violation
        # con_vio = constraint_violation(synth, test, self.cont_feat, self.fixed_features, scaler)
        # synth_metrics['violation'] = con_vio

        # test_inverse = scaler(test) # this will have the original feature names
        # synth_inverse = scaler(synth)

        # print(test_inverse)

        instance = test[self.fixed_features].sort_index().to_numpy()
        cf = synth[self.fixed_features].to_numpy()
        delta = cf - instance

        synth_metrics['violation'] = np.sum(delta != 0, axis=1, dtype=np.float).tolist()

        return synth_metrics