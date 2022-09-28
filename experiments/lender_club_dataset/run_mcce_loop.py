import os
import re
import time
import argparse
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import torch
import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from carla import MLModel
from carla.data.catalog import CsvCatalog
from carla.data.catalog import OnlineCatalog
from carla.models.negative_instances import predict_negative_instances

from sklearn import preprocessing

from mcce.mcce import MCCE

class DatasetMCCE():
    def __init__(self, 
                 immutables, 
                 target,
                 categorical,
                 categorical_encoded,
                 immutables_encoded,
                 continuous,
                 feature_order,
                 encoder,
                 scaler,
                 inverse_transform,
                 ):
        
        self.immutables = immutables
        self.target = target
        self.categorical = categorical
        self.categorical_encoded = categorical_encoded
        self.immutables_encoded = immutables_encoded
        self.continuous = continuous
        self.feature_order = feature_order
        self.encoder = encoder
        self.scaler = scaler
        self.inverse_transform = inverse_transform


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

args = parser.parse_args()

data_name = 'lendingclub'
# n_test = args.number_of_samples
path = args.path
# k = args.k

for number_of_features in [13]: #  10, 20, 30, 40, 50
    print(f'NB OF FEATURES: {number_of_features}')

    data = pd.read_csv("Data/lending_club_loans_cleaned.csv")

    cols = data.columns.tolist()
    cols.remove('loan_outcome')
    cols.remove("purpose")
    cols.remove("loan_amnt")
    cols = cols[0:(number_of_features-2)] + ['loan_outcome', 'purpose', 'loan_amnt']
    print(cols)

    data_path = f"Data/lending_club_loans_cleaned_small_nbfeat_{number_of_features}.csv"
    data[cols].to_csv(data_path, index=False)

    categorical = ["term", "grade", "sub_grade", "emp_length", "home_ownership", "verification_status", 
                "purpose", "addr_state", "initial_list_status", "application_type", "disbursement_method"]

    categorical = [value for value in categorical if value in cols]

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

    continuous = [value for value in continuous if value in cols]

    immutable = ["purpose", "loan_amnt"]

    encoding_method = preprocessing.OneHotEncoder(
                drop="first", sparse=False
            )
    dataset = CsvCatalog(file_path=data_path, # "Data/lending_club_loans_cleaned_small.csv",
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target='loan_outcome',
                        encoding_method=encoding_method
                        )

    ml_model = RandomForestModel(dataset)

    pred = ml_model.predict(dataset.df_test)
    fpr, tpr, thresholds = metrics.roc_curve(dataset.df_test[dataset.target], pred, pos_label=1)
    print(f"AUC of predictive model on out-of-sample test set: {round(metrics.auc(fpr, tpr), 2)}")

    # Define new target feature to use while training
    target = dataset.target
    new_target = target + '_High'

    categorical = dataset.categorical + [dataset.target]
    categorical_encoded = dataset.encoder.get_feature_names(dataset.categorical).tolist() + [new_target]
    immutables = dataset.immutables + [dataset.target]

    df = dataset.df

    # Change prediction from numeric to categorical
    pred = ml_model.predict(df)
    df[new_target] = [1 if row >= 0.5 else 0 for row in pred]

    immutable_features_encoded = []
    for immutable in immutables:
        if immutable in categorical:
            for new_col in categorical_encoded:
                match = re.search(immutable, new_col)
                if match:
                    immutable_features_encoded.append(new_col)
        else:
            immutable_features_encoded.append(immutable)

    # Create new dataset object
    dataset_mcce = DatasetMCCE(immutables=immutables, 
                            target=dataset.target,
                            categorical=dataset.categorical,
                            categorical_encoded=categorical_encoded,
                            immutables_encoded=immutable_features_encoded,
                            continuous=dataset.continuous,
                            feature_order=ml_model.feature_input_order,
                            encoder=dataset.encoder,
                            scaler=dataset.scaler,
                            inverse_transform=dataset.inverse_transform
                            )
                        
    #  Create dtypes for MCCE
    dtypes = dict([(x, "float") for x in dataset_mcce.continuous])
    for x in dataset_mcce.categorical_encoded:
        dtypes[x] = "category"
    df = (df).astype(dtypes)

    factuals = predict_negative_instances(ml_model, df)
    
    start = time.time()
    mcce = MCCE(dataset=dataset,
                model=ml_model)

    mcce.fit(df.drop(dataset.target, axis=1), dtypes)
    time_fit = time.time()

    for n_test in [10, 100, 500]:
        for k in [10, 50, 100, 1000]:

            test_factual = factuals.iloc[:n_test]

            # Define new value for predicted target in test factual
            test_factual[new_target] = np.ones(test_factual.shape[0])
            test_factual[new_target] = test_factual[new_target].astype("category")

            # test_factual.to_csv(os.path.join(path, f"{data_name}_test_factuals_tree_model_k_{k}_n_{n_test}_nbfeat_{number_of_features}_{device}.csv"))
            time_k_start = time.time()
            cfs = mcce.generate(test_factual.drop(dataset.target, axis=1), k=k)
            time_generate = time.time()

            mcce.postprocess(cfs, test_factual, cutoff=0.5, higher_cardinality=False)
            time_postprocess = time.time()

            print(f"p: {number_of_features}, n_test: {n_test}, K: {k}, time: {mcce.postprocess_time}")

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

            cols = ['data', 'method', 'n_test', 'k', 'p'] + ['time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)', 'postprocess (predict)', 'postprocess (metrics)', 'postprocess (loop)'] # + dataset_mcce.categorical_encoded + dataset_mcce.continuous
            results.sort_index(inplace=True)

            path_all = os.path.join(path, f"{data_name}_mcce_results_tree_model_k_several_n_several_nbfeat_several_{device}.csv")
                
            if(os.path.exists(path_all)):
                results[cols].to_csv(path_all, mode='a', header=False)
            else:
                results[cols].to_csv(path_all)
