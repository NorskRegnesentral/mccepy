import os
import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
torch.manual_seed(0)

from sklearn.ensemble import RandomForestClassifier

from carla import MLModel
from carla.data.catalog import OnlineCatalog
from carla.models.negative_instances import predict_negative_instances

class RandomForestModel(MLModel):
    """
    Trains a random forest model using the sklearn RandomForestClassifier method. 
    Takes as input the CARLA MLModel class. 
    
    Parameters
    ----------
    data : mcce.Data
        Object of class mcce.Data which contains the data to train the model on and a set
        of other attributions like which features are continuous, categorical, and fixed.
    
    Methods
    -------
    predict : 
        Predicts the response/target for a data set based on the fitted random forest.
    predict_proba :
        Outputs the predicted probability (between 0 and 1) for a data set based on the fitted random forest.
    get_ordered_features :
        Returns a pd.DataFrame where the features have the same ordering as the original data set. 
    """

    def __init__(self, data):
        super().__init__(data)

        # training data
        df_train = self.data.df_train
        df_test = self.data.df_test
        
        encoded_features = list(self.data.encoder.get_feature_names(self.data.categorical))
        
        x_train = df_train[self.data.continuous + encoded_features]
        y_train = df_train[self.data.target]

        self._feature_input_order = self.data.continuous + encoded_features

        param = {
            "max_depth": None,  # The maximum depth of the tree. If None, then nodes are expanded until 
                                # all leaves are pure or until all leaves contain less than min_samples_split samples.
            "n_estimators": 200, # The number of trees in the forest.
            "min_samples_split": 3 # The minimum number of samples required to split an internal node:
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

parser = argparse.ArgumentParser(description="Print counterfactual examples generated when \
                                              predictive model is a decision tree.")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    required=True,
    help="Path where results are saved",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="adult",
    help="Datasets for experiment. Options are adult, give_me_some_credit, and compas.",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=1000,
    help="Number of test observations to generate counterfactuals for.",
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    default=1000,
    help="Number of observations to sample from each end node for MCCE method.",
)
parser.add_argument(
    "-device",
    "--device",
    type=str,
    default='cpu',
    help="Whether the CARLA methods were trained with a GPU or CPU.",
)
args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
k = args.k
if data_name == 'compas':
    k = 1000
device = args.device

dataset = OnlineCatalog(data_name)

ml_model = RandomForestModel(dataset)

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

test_factual = dataset.inverse_transform(test_factual[factuals.columns])

try:
    df_cfs = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_tree_model_k_{k}_n_{n_test}_{device}.csv"), index_col=0)
except:
    sys.exit(f"No MCCE results saved for k {k} and n_test {n_test} in {path}")

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

# counterfactuals
if len(counterfactuals_without_nans) > 0:
    results = dataset.inverse_transform(counterfactuals_without_nans) # [factuals.columns]
    results['method'] = 'mcce'
    results['data'] = data_name

    results = pd.concat([factual_without_nans, results], axis=0)

if data_name == 'adult':
    to_write = results.loc[[1, 31]].sort_index() # , 122, 124

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
    num_feat = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

elif data_name == 'give_me_some_credit':
    cols = ['method', 'age', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse', 
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 
            'NumberOfDependents']

    to_write = results[cols].loc[[287, 512]].sort_index() # , 1013, 1612

    cols = ['method', 'Age', 'Unsec. Lines', 'Nb Days Past 30', 'Debt Ratio', 'Month Inc.', 'Nb Credit Lines', 
            'Nb Times 90 Days Late', 'Nb Real Estate Loans', 'Nb Times 60 Days Past', 'Nb Dep.']
    num_feat = ['Age', 'Unsec. Lines', 'Nb Days Past 30', 'Debt Ratio', 'Month Inc.', 'Nb Credit Lines', 
                'Nb Times 90 Days Late', 'Nb Real Estate Loans', 'Nb Times 60 Days Past', 'Nb Dep.']

    to_write.columns = cols

elif data_name == 'compas':
    cols = ['method', 'age', 'two_year_recid', 'priors_count', 'length_of_stay', 'c_charge_degree', 'race', 'sex']
    
    to_write = results[cols].loc[[67, 286]].sort_index()

    cols = ['method', 'Age', 'Two Year Recid', 'Priors Count', 'Length of Stay', 'Charge Degree', 'Race', 'Sex']
    num_feat = ['Age', 'Two Year Recid', 'Priors Count', 'Length of Stay']

    to_write.columns = cols

# Fix method names
dct = {'original': 'Original', 
       'cchvae': 'C-CHVAE',
       'cem-vae': 'CEM-VAE',
       'clue': 'CLUE',
       'crud': 'CRUDS',
       'face': 'FACE',
       'revise': 'REViSE',
       'mcce': 'MCCE'}

to_write['method'] = [dct[item] for item in to_write['method']]

to_write = to_write[cols].round(0)

# Order the methods
s1 = to_write[to_write['method'] == 'Original'].iloc[0:1]
s2 = to_write[to_write['method'] == 'MCCE'].iloc[0:1]
s3 = to_write[to_write['method'] == 'Original'].iloc[1:2]
s4 = to_write[to_write['method'] == 'MCCE'].iloc[1:2]

to_write = pd.concat([s1, s2, s3, s4])

# Remove decimal point
to_write[num_feat] = to_write[num_feat].astype(np.int64)

print(to_write.to_string())
print(to_write.to_latex(index=False))  
 
