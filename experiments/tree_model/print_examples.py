import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)

import os
import argparse

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
    "-k",
    "--k",
    type=int,
    default=10000,
    help="Number generated counterfactuals per test observation",
)

args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
k = args.k

results_inverse = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_tree_model_k_{k}_n_{n_test}_inverse_transform.csv"), index_col=0)
true_raw = pd.read_csv(os.path.join(path, f"{data_name}_tree_model_n_{n_test}_inverse_transform.csv"), index_col=0)
true_raw['method'] = 'Original'

results_inverse['method'] = 'MCCE'
temp = pd.concat([results_inverse, true_raw], axis=0)

if data_name == 'adult':
       cols = ['method', 'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', \
              'hours-per-week', 'marital-status', 'native-country', \
              'occupation', 'race', 'relationship', 'sex', 'workclass']
       
       to_write = temp[cols].loc[[1, 31, 122, 124]].sort_index()
       to_write.columns = cols

elif data_name == 'give_me_some_credit':
       cols = ['method', 'age', 'RevolvingUtilizationOfUnsecuredLines', \
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', \
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', \
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', \
       'NumberOfDependents']

       to_write = temp[cols].loc[[287, 512, 1013, 1612]].sort_index()

       cols = ['method', 'Age', 'Unsec. Lines', \
       '30 Days Past', 'Debt Ratio', 'Month Inc', \
       'Credit Lines', '90 Days Late', \
       'Real Est. Loans', '60 Days Past', \
       'Nb Dep.']


       to_write.columns = cols

elif data_name == 'compas':
       cols = ['method', 'age', 'two_year_recid', 'priors_count', 'length_of_stay',
       'c_charge_degree', 'race', 'sex']

       to_write = temp[cols].loc[[67, 286]].sort_index()

if data_name == 'adult':
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


print(to_write) # .to_latex(index=False, float_format="%.0f", )