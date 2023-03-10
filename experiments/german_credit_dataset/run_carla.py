import os
import time
import argparse
import pandas as pd

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog

import torch

from carla import MLModel
from carla.data.catalog import CsvCatalog

from sklearn import preprocessing
from sklearn import metrics
from mcce.mcce import MCCE
# must do pip install . in CARLA_version_2 directory

device = "cuda" if torch.cuda.is_available() else "cpu"

# must do pip install . in CARLA_version_3 directory

def save_csv(df, data_name):
    file_name = os.path.join(path, f"{data_name}_carla_results_n_{n_test}_{device}.csv")

    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', header=False, index=True)
    else:
        df.to_csv(file_name, mode='a', header=True, index=True)


parser = argparse.ArgumentParser(description="Fit various recourse methods from CARLA.")
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
    nargs="*",
    default=["german_credit"],
    choices=["german_credit"],
    help="Datasets for experiment",
)
parser.add_argument(
    "-r",
    "--recourse_method",
    nargs="*",
    default=[
        # "cchvae",
        "cem-vae",
        "clue",
        "crud",
        "revise",
        "face"
    ],
    choices=[
        # "cchvae",
        "cem-vae",
        "clue",
        "crud",
        "revise",
        "face"
    ],
    help="Recourse methods for experiment",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=100,
    help="Number of instances per dataset",
)
parser.add_argument(
    "-ft",
    "--force_train",
    action='store_true',  # default is False
    help="Whether to train the prediction model from scratch or not. Default will not train.",
)

args = parser.parse_args()
n_test = args.number_of_samples
force_train = args.force_train
path = args.path
k = 1000

for data_name in args.dataset:
    print(f"Training data set: {data_name}")
    categorical = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
    continuous = ["Age", "Credit amount", "Duration"]
    immutable = ["Purpose", "Age", "Sex"]

    encoding_method = preprocessing.OneHotEncoder(
                drop="first", sparse=False
            )
    dataset = CsvCatalog(file_path="Data/german_credit_data_complete.csv",
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target='Risk',
                        encoding_method=encoding_method
                        )

    torch.manual_seed(0)
    
    ml_model = MLModelCatalog(
        dataset, 
        model_type="ann", 
        load_online=False, 
        backend="pytorch"
    )

    ml_model.train(
        learning_rate=0.0005,
        epochs=20,
        batch_size=8, # 64
        hidden_size=[81, 27, 3], # 64
        force_train=force_train,
    )

    pred = ml_model.predict_proba(dataset.df_test)
    pred = [row[1] for row in pred]
    fpr, tpr, thresholds = metrics.roc_curve(dataset.df_test[dataset.target], pred, pos_label=1)
    print(f"AUC of predictive model on out-of-sample test set: {round(metrics.auc(fpr, tpr), 2)}")


    factuals = predict_negative_instances(ml_model, dataset.df)
    test_factual = factuals.iloc[:n_test]
    # Read in factuals from MCCE since model is not reproducible! 
    test_factual = pd.read_csv(os.path.join(path, f"german_credit_test_factuals_ann_model_k_{k}_n_{n_test}_{device}.csv"), index_col=0)


    nb_immutables = len(dataset.immutables)

    recourse_methods = [
                        "cchvae",
                        # "clue",
                        # "crud",
                        "revise",
                        # "face"
                        ]


    for rm in recourse_methods:
        print(f"Finding counterfactuals using: {rm}")
        
        if rm == 'revise':
            
            start = time.time()
            hyperparams = {
            "data_name": dataset.name,
                "lambda": 0.5,
                "optimizer": "adam",
                "lr": 0.1,
                "max_iter": 1500, # 1000,
                "target_class": [0, 1],
                "binary_cat_features": True,
                "vae_params": {
                    "layers": [len(ml_model.feature_input_order) - nb_immutables, 512, 256, 8],
                    "train": True,
                    "lambda_reg": 1e-6,
                    "epochs": 5,
                    "lr": 1e-3,
                    "batch_size": 32,
                },
                }

            revise = recourse_catalog.Revise(ml_model, dataset, hyperparams)
            df_cfs = revise.get_counterfactuals(test_factual)

            df_cfs.index = test_factual.index 
            df_cfs.insert(0, 'method', rm)
            df_cfs.insert(1, 'data', data_name)
            timing = time.time() - start
            df_cfs['time (seconds)'] = timing
            save_csv(df_cfs, data_name)
            
        elif rm == 'cchvae':
            
            start = time.time()
            hyperparams = {
                "data_name": dataset.name,
                "n_search_samples": 100,
                "p_norm": 1,
                "step": 0.1,
                "max_iter": 1000,
                "clamp": True,
                "binary_cat_features": False,
                "vae_params": {
                    "layers": [len(ml_model.feature_input_order) - nb_immutables, 512, 256, 8],
                    "train": True,
                    "lambda_reg": 1e-6,
                    "epochs": 5,
                    "lr": 1e-3,
                    "batch_size": 32,
                },
            }

            cchvae = recourse_catalog.CCHVAE(ml_model, hyperparams)
            df_cfs = cchvae.get_counterfactuals(test_factual)

            df_cfs.index = test_factual.index
            df_cfs.insert(0, 'method', rm)
            df_cfs.insert(1, 'data', data_name)
            timing = time.time() - start
            df_cfs['time (seconds)'] = timing
            save_csv(df_cfs, data_name)

        elif rm == 'crud':

            start = time.time()
            hyperparams = {
                "data_name": dataset.name, # this get's non NA for 10/10
                "target_class": [0, 1],
                "lambda_param": 0.001,
                "optimizer": "RMSprop",
                "lr": 0.008,
                "max_iter": 2000,
                "binary_cat_features": False,
                "vae_params": {
                    "layers": [len(ml_model.feature_input_order)-nb_immutables, 16, 8],
                    "train": True,
                    "epochs": 5,
                    "lr": 0.0001,
                    "batch_size": 4,
                },
            }
            crud = recourse_catalog.CRUD(ml_model, hyperparams)
            df_cfs = crud.get_counterfactuals(test_factual)

            df_cfs.index = test_factual.index
            df_cfs.insert(0, 'method', rm)
            df_cfs.insert(1, 'data', data_name)
            timing = time.time() - start
            df_cfs['time (seconds)'] = timing
            save_csv(df_cfs, data_name)

        elif rm == 'clue':

            start = time.time()
            hyperparams = { # this get's non-NA for 2/10
                "data_name": dataset.name,
                "train_vae": True,
                "width": 10,
                "depth": 3, 
                "latent_dim": 12, 
                "batch_size": 4, # 64
                "epochs": 20, 
                "lr": 0.0005, # 0.001
                "early_stop": 10,
            }

            clue = recourse_catalog.Clue(dataset, ml_model, hyperparams)
            df_cfs = clue.get_counterfactuals(test_factual)

            df_cfs.index = test_factual.index
            df_cfs.insert(0, 'method', rm)
            df_cfs.insert(1, 'data', data_name)
            timing = time.time() - start
            df_cfs['time (seconds)'] = timing
            save_csv(df_cfs, data_name)

        # elif rm == 'cem-vae':

        #     start = time.time()
        #     hyperparams = {
        #         "data_name": dataset.name,
        #         "batch_size": 1,
        #         "kappa": 0.1,
        #         "init_learning_rate": 0.01,
        #         "binary_search_steps": 9,
        #         "max_iterations": 100,
        #         "initial_const": 10,
        #         "beta": 0.9,
        #         "gamma": 1.0, # 0.0, #   1.0
        #         "mode": "PN",
        #         "num_classes": 2,
        #         "ae_params": {"hidden_layer": [20, 10, 7], "train_ae": True, "epochs": 5},
        #     }

        #     from tensorflow import Graph, Session

        #     graph = Graph()
        #     with graph.as_default():
        #         ann_sess = Session()
        #         with ann_sess.as_default():
        #             ml_model_sess = MLModelCatalog(dataset, "ann", "tensorflow")

        #             factuals_sess = predict_negative_instances(
        #                 ml_model_sess, dataset.df
        #             )
        #             factuals_sess = factuals_sess.iloc[:n_test].reset_index(drop=True)

        #             cem = recourse_catalog.CEM(ann_sess, ml_model_sess, hyperparams)
        #             df_cfs = cem.get_counterfactuals(factuals_sess)
                    
        #             df_cfs.index = test_factual.index
        #             df_cfs.insert(0, 'method', rm)
        #             df_cfs.insert(1, 'data', data_name)
        #             timing = time.time() - start
        #             df_cfs['time (seconds)'] = timing
        #             save_csv(df_cfs, data_name)

        elif rm == 'face':
            
            start = time.time()
            hyperparams = { # this get's non NA for 100/100
                "data_name": dataset.name,
                "mode": "knn",
                "fraction": 0.75
            }

            face = recourse_catalog.Face(ml_model, hyperparams)
            df_cfs = face.get_counterfactuals(test_factual)

            df_cfs.index = test_factual.index
            df_cfs.insert(0, 'method', rm)
            df_cfs.insert(1, 'data', data_name)
            timing = time.time() - start
            df_cfs['time (seconds)'] = timing
            save_csv(df_cfs, data_name)