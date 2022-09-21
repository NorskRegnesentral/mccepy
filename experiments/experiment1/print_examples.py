import os
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
torch.manual_seed(0)

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
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
    default='adult',
    help="Datasets for experiment",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=1000,
    help="Number of instances per dataset",
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    default=1000,
    help="Number generated counterfactuals per test observation",
)
parser.add_argument(
    "-ft",
    "--force_train",
    action='store_true',  # default is False
    help="Whether to train the prediction model from scratch or not. Default will not train.",
)
parser.add_argument(
    "-device",
    "--device",
    type=str,
    default='cpu',
    help="Whether the CARLA methods were trained with a GPU or CPU (default).",
)
args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
k = args.k
if data_name == 'compas':
    k = 1000
device = args.device
force_train = args.force_train

print(f"Load {data_name} data set")

dataset = OnlineCatalog(data_name)

ml_model = MLModelCatalog(
        dataset, 
        model_type="ann", 
        load_online=False, 
        backend="pytorch"
    )
if data_name == 'adult':
    ml_model.train(
    learning_rate=0.002,
    epochs=20,
    batch_size=1024,
    hidden_size=[18, 9, 3],
    force_train=force_train,
    )
elif data_name == 'give_me_some_credit':
    ml_model.train(
    learning_rate=0.002,
    epochs=20,
    batch_size=2048,
    hidden_size=[18, 9, 3],
    force_train=force_train,
    )
elif data_name == 'compas':
    ml_model.train(
    learning_rate=0.002,
    epochs=25,
    batch_size=25,
    hidden_size=[18, 9, 3],
    force_train=force_train,
    )

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

results = dataset.inverse_transform(test_factual[factuals.columns])
results['method'] = 'original'
results['data'] = data_name

for method in ['cchvae', 'cem-vae', 'revise', 'clue', 'crud', 'face', 'mcce']:
    print(f"Finding examples for {method}")

    if method == 'mcce':
        try:
            cfs = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_k_{k}_n_{n_test}_{device}.csv"), index_col=0)
        except:
            print(f"No {method} results saved for k {k} and {n_test} in {path}")
            continue
    else:
        try:
            cfs = pd.read_csv(os.path.join(path, f"{data_name}_carla_results_n_{n_test}_{device}.csv"), index_col=0)
        except:
            print(f"No {method} results saved for k {k} and {n_test} in {path}")
            continue
    
    df_cfs = cfs[cfs['method'] == method].drop(['method',	'data'], axis=1)
    # In case the script was accidentally run twice, we drop the duplicate indices per method
    df_cfs = df_cfs[~df_cfs.index.duplicated(keep='first')]
    
    df_cfs.sort_index(inplace=True)
    if dataset.target not in df_cfs.columns:
        df_cfs = df_cfs.join(test_factual[dataset.target])

    # remove missing values
    nan_idx = df_cfs.index[df_cfs.isnull().any(axis=1)]
    non_nan_idx = df_cfs.index[~(df_cfs.isnull()).any(axis=1)]

    output_factuals = test_factual.copy()
    output_counterfactuals = df_cfs.copy()

    factual_without_nans = output_factuals.drop(index=nan_idx)
    counterfactuals_without_nans = output_counterfactuals.drop(index=nan_idx)

    # in case not all test obs have a generated counterfactual!
    factual_without_nans = factual_without_nans.loc[counterfactuals_without_nans.index.to_list()]

    # counterfactuals
    if len(counterfactuals_without_nans) > 0:
        temp = dataset.inverse_transform(counterfactuals_without_nans[factuals.columns])
        temp['method'] = method
        temp['data'] = data_name

        results = pd.concat([results, temp], axis=0)

if data_name == 'adult':
    to_write = results.loc[31]

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
    dct = {'Female': 'F', 'Male': 'M'}
    to_write[feature] = [dct[item] for item in to_write[feature]]

    feature = 'workclass'
    dct = {'Self-emp-not-inc': 'SENI', 'Private': 'P', 'Non-Private': 'NP'}
    to_write[feature] = [dct[item] for item in to_write[feature]]

    cols = ['method', 'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 
            'hours-per-week', 'marital-status', 'native-country', 
            'occupation', 'race', 'relationship', 'sex', 'workclass']
    num_feat = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

elif data_name == 'give_me_some_credit':
    
    cols = ['method', 'age', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse', 
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 
            'NumberOfDependents']

    to_write = results[cols].loc[263]

    cols = ['method', 'Age', 'Unsec. Lines', 'Nb Days Past 30', 'Debt Ratio', 'Month Inc.', 'Nb Credit Lines', 
            'Nb Times 90 Days Late', 'Nb Real Estate Loans', 'Nb Times 60 Days Past', 'Nb Dep.']
    num_feat = ['Age', 'Unsec. Lines', 'Nb Days Past 30', 'Debt Ratio', 'Month Inc.', 'Nb Credit Lines', 
                'Nb Times 90 Days Late', 'Nb Real Estate Loans', 'Nb Times 60 Days Past', 'Nb Dep.']

    to_write.columns = cols

elif data_name == 'compas':
    cols = ['method', 'age', 'two_year_recid', 'priors_count', 'length_of_stay', 'c_charge_degree', 'race', 'sex']
    to_write = results[cols].loc[40]

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
s1 = to_write[to_write['method'] == 'Original']
s2 = to_write[to_write['method'] == 'MCCE']
s3 = to_write[(to_write['method'] != 'Original') & (to_write['method'] != 'MCCE')]

to_write = pd.concat([s1, s3.sort_values('method'), s2])

# Remove decimal point
to_write[num_feat] = to_write[num_feat].astype(np.int64)

# print(to_write.to_string())
print(to_write.to_latex(index=False))  
 

    
