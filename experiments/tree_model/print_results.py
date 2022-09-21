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

from mcce.metrics import distance, constraint_violation, feasibility, success_rate

# Fit predictive model that takes into account MLModel (a CARLA class!)
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

parser = argparse.ArgumentParser(description="Print MCCE metric results when predictive model is \
                                              a decision tree.")
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
device = args.device # cuda # cpu
k = args.k

# Load data set from CARLA
dataset = OnlineCatalog(data_name)

ml_model = RandomForestModel(dataset)

# Find unhappy customers and choose which ones to make counterfactuals for
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

# Read results
try:
    cfs = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_tree_model_k_{k}_n_{n_test}_{device}.csv"), index_col=0)
except:
    sys.exit(f"No MCCE results saved for {data_name}, k {k}, n_test {n_test}, and device {device} in {path}")

df_cfs = cfs.drop(['method', 'data'], axis=1)
# In case the script was accidentally run twice, we drop the duplicate indices per method
df_cfs = df_cfs[~df_cfs.index.duplicated(keep='first')]
df_cfs.sort_index(inplace=True)
    
# Remove missing values
nan_idx = df_cfs.index[df_cfs.isnull().any(axis=1)]
non_nan_idx = df_cfs.index[~(df_cfs.isnull()).any(axis=1)]

output_factuals = test_factual.copy()
output_counterfactuals = df_cfs.copy()

factual_without_nans = output_factuals.drop(index=nan_idx)
counterfactuals_without_nans = output_counterfactuals.drop(index=nan_idx)

# In case not all test obs have a generated counterfactual!
factual_without_nans = factual_without_nans.loc[counterfactuals_without_nans.index.to_list()]

# Calculate metrics
if len(counterfactuals_without_nans) > 0:
    results = dataset.inverse_transform(counterfactuals_without_nans) # [factuals.columns]
    results['method'] = 'mcce'
    results['data'] = data_name
    
    # calculate distances
    distances = pd.DataFrame(distance(counterfactuals_without_nans, factual_without_nans, dataset, higher_card=False))
    distances.set_index(non_nan_idx, inplace=True)
    results = pd.concat([results, distances], axis=1)

    # calculate feasibility
    feas_col = dataset.df.columns.to_list()
    feas_col.remove(dataset.target)
    results['feasibility'] = feasibility(counterfactuals_without_nans, factual_without_nans, feas_col)
    
    # calculate violation
    violations = []
    df_decoded_cfs = dataset.inverse_transform(counterfactuals_without_nans)
    df_factuals = dataset.inverse_transform(factual_without_nans)
    
    total_violations = constraint_violation(df_decoded_cfs, df_factuals, dataset)
    for x in total_violations:
        violations.append(x[0])
    results['violation'] = violations
    
    # calculate success
    results['success'] = success_rate(counterfactuals_without_nans, ml_model, cutoff=0.5)

    # calculate time
    results['time (seconds)'] = df_cfs['time (seconds)'].mean() 


cols = ['method', 'L0', 'L1', 'feasibility', 'success', 'violation', 'time (seconds)']
temp = results[cols]

to_write_mean = temp[['method', 'L0', 'L1', 'feasibility', 'violation', 'success', 'time (seconds)']].groupby(['method']).mean()
to_write_mean.reset_index(inplace=True)

to_write_sd = temp[['method', 'L0', 'L1', 'feasibility', 'violation', 'success']].groupby(['method']).std()
to_write_sd.reset_index(inplace=True)
to_write_sd.rename(columns={'L0': 'L0_sd', 'L1': 'L1_sd', 'feasibility': 'feasibility_sd', 'violation': 'violation_sd', 'success': 'success_sd'}, inplace=True)

CE_N = temp.groupby(['method']).size().reset_index().rename(columns={0: 'CE_N'})
to_write = pd.concat([to_write_mean, to_write_sd[['L0_sd', 'L1_sd', 'feasibility_sd', 'violation_sd', 'success_sd']], CE_N.CE_N], axis=1)
to_write = to_write[['method', 'L0', 'L0_sd', 'L1', 'L1_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N', 'time (seconds)']]


# Fix method names
dct = {'mcce': 'MCCE'}

to_write['method'] = [dct[item] for item in to_write['method']]

# Order the methods
s1 = to_write[to_write['method'] == 'MCCE']
s2 = to_write[(to_write['method'] != 'Original') & (to_write['method'] != 'MCCE')]
to_write = pd.concat([s2.sort_values('method'), s1])

# Remove decimal point
num_feat = ['CE_N']
to_write[num_feat] = to_write[num_feat].astype(np.int64)

to_write = to_write.round(2)

cols = ['L0', 'L0_sd', 'L1', 'L1_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success']
to_write[cols] = to_write[cols].astype(str)

# Add the standard deviations in original columns
to_write["L0"] = to_write["L0"] + " (" + to_write["L0_sd"] + ")"
to_write["L1"] = to_write["L1"] + " (" + to_write["L1_sd"] + ")"
to_write["feasibility"] = to_write["feasibility"] + " (" + to_write["feasibility_sd"] + ")"
to_write["violation"] = to_write["violation"] + " (" + to_write["violation_sd"] + ")"

print(to_write[['method', 'L0', 'L1', 'feasibility', 'violation', 'success', 'CE_N', 'time (seconds)']].to_latex(index=False))
