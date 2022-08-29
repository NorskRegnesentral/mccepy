import os
import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np

import torch
torch.manual_seed(0)

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from carla.data.catalog import OnlineCatalog
from carla.models.negative_instances import predict_negative_instances
from carla import MLModel

import pandas as pd
pd.set_option('display.max_columns', None)

from mcce.metrics import distance, constraint_violation, feasibility, success_rate

parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
parser.add_argument(
    "-p",
    "--path",
    required=True,
    help="Path where results are saved",
)
parser.add_argument(
    "-d",
    "--dataset",
    default='adult',
    help="Datasets for experiment",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=100,
    help="Number of instances per dataset",
)
parser.add_argument(
    "-K",
    "--K",
    type=int,
    default=10000,
    help="Number generated counterfactuals per test observation",
)


args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
K = args.K

dataset = OnlineCatalog(data_name)

class RandomForestModel2(MLModel):
    """The default way of implementing RandomForest from sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"""

    def __init__(self, data):
        super().__init__(data)

        # get preprocessed data
        df_train = self.data.df_train
        df_test = self.data.df_test
        
        encoded_features = list(self.data.encoder.get_feature_names(self.data.categorical))
        
        x_train = df_train[self.data.continuous + encoded_features]
        y_train = df_train[self.data.target]

        self._feature_input_order = self.data.continuous + encoded_features

        param = {
            "max_depth": None,  # determines how deep the tree can go
            "n_estimators": 200,
            "min_samples_split": 3, # number of features to consider at each split
            
        }
        np.random.seed(1) # important to use np and not random with sklearn!
        self._mymodel = RandomForestClassifier(**param)
        self._mymodel.fit(
                x_train,
                y_train,
            )

    @property
    def feature_input_order(self):
        # List of the feature order the ml model was trained on
        return self._feature_input_order

    @property
    def backend(self):
        return "xgboost"

    @property
    def raw_model(self):
        return self._mymodel

    @property
    def tree_iterator(self):
        # make a copy of the trees, else feature names are not saved
        booster_it = [booster for booster in self.raw_model.get_booster()]
        # set the feature names
        for booster in booster_it:
            booster.feature_names = self.feature_input_order
        return booster_it

    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))

    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))

ml_model = RandomForestModel2(dataset)

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

df_cfs = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_tree_model_k_{K}_n_{n_test}.csv"), index_col=0)

df_cfs.sort_index(inplace=True)
    
# remove missing values
nan_idx = df_cfs.index[df_cfs.isnull().any(axis=1)]
non_nan_idx = df_cfs.index[~(df_cfs.isnull()).any(axis=1)]

output_factuals = test_factual.copy()
output_counterfactuals = df_cfs.copy()

factual_without_nans = output_factuals.drop(index=nan_idx)
counterfactuals_without_nans = output_counterfactuals.drop(index=nan_idx)

# in case not all test obs have a generated counterfactual!
factual_without_nans = factual_without_nans.loc[counterfactuals_without_nans.index.to_list()]

# calculate metrics
if len(counterfactuals_without_nans) > 0:
    results = dataset.inverse_transform(counterfactuals_without_nans[factuals.columns])
    results['method'] = 'mcce'
    results['data'] = data_name
    
    # distance
    distances = pd.DataFrame(distance(counterfactuals_without_nans, factual_without_nans, dataset, higher_card=False))
    distances.set_index(non_nan_idx, inplace=True)
    results = pd.concat([results, distances], axis=1)

    results['feasibility'] = feasibility(counterfactuals_without_nans, factual_without_nans, dataset.df.columns)
    
    # violation
    violations = []
    df_decoded_cfs = dataset.inverse_transform(counterfactuals_without_nans)
    df_factuals = dataset.inverse_transform(factual_without_nans)
    
    total_violations = constraint_violation(df_decoded_cfs, df_factuals, \
        dataset.continuous, dataset.categorical, dataset.immutables)
    for x in total_violations:
        violations.append(x[0])
    results['violation'] = violations
    
    # success
    results['success'] = success_rate(counterfactuals_without_nans, ml_model, cutoff=0.5)

    # time
    results['time (seconds)'] = df_cfs['time (seconds)'].mean() 


cols = ['method', 'L0', 'L2', 'feasibility', 'success', 'violation', 'time (seconds)']
temp = results[cols]  # pd.concat([all_results[cols], results[cols]], axis=0)

print("Writing results")
to_write = temp[['method', 'L0', 'L2', 'feasibility', 'violation', 'success', 'time (seconds)']].groupby(['method']).mean()
to_write.reset_index(inplace=True)

to_write_sd = temp[['method', 'L0', 'L2', 'feasibility', 'violation', 'success']].groupby(['method']).std()
to_write_sd.reset_index(inplace=True)
to_write_sd.rename(columns={'L0': 'L0_sd', 'L2': 'L2_sd', 'feasibility': 'feasibility_sd', 'violation': 'violation_sd', 'success': 'success_sd'}, inplace=True)

CE_N = temp.groupby(['method']).size().reset_index().rename(columns={0: 'CE_N'})
to_write = pd.concat([to_write, to_write_sd[['L0_sd', 'L2_sd', 'feasibility_sd', 'violation_sd', 'success_sd']], CE_N.CE_N], axis=1)
to_write = to_write[['method', 'L0', 'L0_sd', 'L2', 'L2_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N', 'time (seconds)']]

print(to_write.to_string())