import os
import time
import argparse
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import torch
import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from carla.models.negative_instances import predict_negative_instances
from carla import MLModel
from carla.data.catalog import CsvCatalog

from sklearn import preprocessing

from mcce.mcce import MCCE
# must do pip install . in CARLA_version_2 directory

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

        # get preprocessed data
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
        np.random.seed(1)  # important to use np and not random with sklearn!
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

    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))

    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Fit MCCE when the underlying predictive function is a decision tree.")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    required=True,
    help="Path where results are saved",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=100,
    help="Number of test observations to generate counterfactuals for.",
)

args = parser.parse_args()

data_name = 'lendingclub'
n_test = args.number_of_samples
path = args.path

# Load data set from CARLA
print("Read in processed data using CARLA")
categorical = ["term", "grade", "sub_grade", "emp_length", "home_ownership", "verification_status", 
               "purpose", "addr_state", "initial_list_status", "application_type", "disbursement_method"]

continuous = [
   "loan_amnt",
   "funded_amnt", 
   "funded_amnt_inv", 
   "int_rate",
   "installment",
   "annual_inc",
   "delinq_2yrs", 
   "inq_last_6mths", 
   "open_acc",
   "pub_rec",
   "revol_bal",
   "revol_util",
   "total_acc",
   "acc_now_delinq", 
   "tot_coll_amt",
   "tot_cur_bal", 
   "acc_open_past_24mths",
   "avg_cur_bal", 
   "delinq_amnt", 
   "mo_sin_rcnt_tl", 
   "mort_acc", 
   "num_accts_ever_120_pd",
   "num_actv_bc_tl", 
   "num_actv_rev_tl",
   "num_bc_sats",
   "num_bc_tl",
   "num_op_rev_tl",
   "num_rev_accts",
   "num_rev_tl_bal_gt_0",
   "num_sats",
   "num_tl_30dpd",
   "pct_tl_nvr_dlq",
   "pub_rec_bankruptcies",
   "tax_liens",
   "tot_hi_cred_lim",
   "total_bc_limit"    
]

immutable = ["purpose", "loan_amnt"]

encoding_method = preprocessing.OneHotEncoder(
            drop="first", sparse=False
        )
dataset = CsvCatalog(file_path="Data/lending_club_loans_cleaned_small.csv",
                     continuous=continuous,
                     categorical=categorical,
                     immutables=immutable,
                     target='loan_outcome',
                     encoding_method=encoding_method
                     )

ml_model = RandomForestModel(dataset)

pred = ml_model.predict_proba(dataset.df_test)
pred = [row[1] for row in pred]
fpr, tpr, thresholds = metrics.roc_curve(dataset.df_test[dataset.target], pred, pos_label=1)
print(f"AUC of predictive model on out-of-sample test set: {round(metrics.auc(fpr, tpr), 2)}")

print("Find factuals to generate counterfactuals for")
factuals = predict_negative_instances(ml_model, dataset.df)
print(f"Number of possible factuals: {factuals}")
test_factual = factuals.iloc[:n_test]

test_factual.to_csv(os.path.join(path, f"{data_name}_test_factuals_tree_model_k_several_n_{n_test}_{device}.csv"))

print("Prepare data for MCCE")
y_col = dataset.target
cont_feat = dataset.continuous

cat_feat = dataset.categorical
cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)

dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat_encoded:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)

print("Fit trees")
start = time.time()
mcce = MCCE(dataset=dataset,
            model=ml_model)

mcce.fit(df.drop(dataset.target, axis=1), dtypes)
time_fit = time.time()

for k in [100]: # 500, 1000, 5000, 10000, 25000, 50000, 100000
    print(f"K: {k}")

    time_k_start = time.time()

    cfs = mcce.generate(test_factual.drop(dataset.target, axis=1), k=k)
    time_generate = time.time()

    mcce.postprocess(cfs, test_factual, cutoff=0.5, higher_cardinality=False)
    time_postprocess = time.time()

    print(mcce.postprocess_time)

    results = mcce.results_sparse

    results['time (seconds)'] = (time_fit - start) + (time_generate - time_k_start) + (time_postprocess - time_generate)
    results['fit (seconds)'] = time_fit - start
    results['generate (seconds)'] = time_generate - time_k_start
    results['postprocess (seconds)'] = time_postprocess - time_generate
    
    results['postprocess (predict)'] = mcce.postprocess_time['predict']
    results['postprocess (metrics)'] = mcce.postprocess_time['metrics']
    results['postprocess (loop)'] = mcce.postprocess_time['loop']

    results['data'] = data_name
    results['method'] = 'mcce'
    results['n_test'] = n_test
    results['k'] = k
    results['p'] = number_of_features

    # Get the fitted tree depth for each mutable feature
    # tree_depth_cols = []
    # for x in dataset.feature_order:
    #     if x not in dataset.immutables:
    #         tree_depth_cols.append(x + "_tree_depth")
    #         results[x + "_tree_depth"] = mcce.fitted_model[x].get_depth()

    # results_copy = results.copy()
    # results_copy[ml_model.feature_input_order] = results_copy[ml_model.feature_input_order].astype(float)
    # results['prediction'] = ml_model.predict(results_copy)

    cols = ['data', 'method', 'n_test', 'k'] + ['time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)', 'postprocess (predict)', 'postprocess (metrics)', 'postprocess (loop)'] + cat_feat_encoded.tolist() + cont_feat
    results.sort_index(inplace=True)

    path_all = os.path.join(path, f"{data_name}_mcce_results_tree_model_k_several_n_{n_test}_{device}.csv")
    
    if(os.path.exists(path_all)):
        results[cols].to_csv(path_all, mode='a', header=False)
    else:
        results[cols].to_csv(path_all)