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

parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
parser.add_argument(
    "-p",
    "--path",
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

test_factual = dataset.inverse_transform(test_factual[factuals.columns])

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
factual_without_nans['method'] = 'original'
factual_without_nans['data'] = data_name

# results_inverse = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_tree_model_k_{K}_n_{n_test}_inverse_transform.csv"), index_col=0)
# true_raw = pd.read_csv(os.path.join(path, f"{data_name}_tree_model_n_{n_test}_inverse_transform.csv"), index_col=0)
# true_raw['method'] = 'Original'

# results_inverse['method'] = 'MCCE'
# temp = pd.concat([results_inverse, true_raw], axis=0)

# counterfactuals
if len(counterfactuals_without_nans) > 0:
    results = dataset.inverse_transform(counterfactuals_without_nans[factuals.columns])
    results['method'] = 'mcce'
    results['data'] = data_name

    results = pd.concat([factual_without_nans, results], axis=0)

if data_name == 'adult':
    to_write = results.loc[[1, 31]].sort_index() # , 122, 124
    # to_write.columns = cols

    feature = 'marital-status'
    dct = {'Married': 'M', 'Non-Married': 'NM'}
    to_write[feature] = [dct[item] for item in to_write[feature]]

    feature = 'native-country'
    dct = {'Non-US': 'NUS', 'US': 'US'}
    to_write[feature] = [dct[item] for item in to_write[feature]]

    feature = 'occupation'
    dct = {'Managerial-Specialist': 'MS', 'Other': 'O'}
    to_write[feature] = [dct[item] for item in to_write[feature]]

    feature = 'race'
    dct = {'White': 'W', 'Non-White': 'NW'}
    to_write[feature] = [dct[item] for item in to_write[feature]]

    feature = 'relationship'
    dct = {'Husband': 'H', 'Non-Husband': 'NH'}
    to_write[feature] = [dct[item] for item in to_write[feature]]

    feature = 'sex'
    dct = {'Male': 'M'}
    to_write[feature] = [dct[item] for item in to_write[feature]]

    feature = 'workclass'
    dct = {'Self-emp-not-inc': 'SENI', 'Private': 'P', 'Non-Private': 'NP'}
    to_write[feature] = [dct[item] for item in to_write[feature]]

    cols = ['method', 'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'marital-status', 'native-country', 'occupation', 'race', 'relationship', 'sex', 'workclass']

elif data_name == 'give_me_some_credit':
    cols = ['method', 'age', 'RevolvingUtilizationOfUnsecuredLines', \
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', \
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', \
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', \
    'NumberOfDependents']

    to_write = results[cols].loc[[287, 512]].sort_index() # , 1013, 1612

    cols = ['method', 'Age', 'Unsec. Lines', \
    '30 Days Past', 'Debt Ratio', 'Month Inc', \
    'Credit Lines', '90 Days Late', \
    'Real Est. Loans', '60 Days Past', \
    'Nb Dep.']

    to_write.columns = cols

elif data_name == 'compas':
    cols = ['method', 'age', 'two_year_recid', 'priors_count', 'length_of_stay',
    'c_charge_degree', 'race', 'sex']

    to_write = results[cols].loc[[67, 286]].sort_index()

print(to_write[cols].round(0).to_string())