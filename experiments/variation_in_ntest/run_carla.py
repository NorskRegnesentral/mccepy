import os
import time
import argparse

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    default=["adult", "give_me_some_credit"], # "compas"
    choices=["adult", "give_me_some_credit"], # , "compas"
    help="Datasets for experiment",
)
parser.add_argument(
    "-r",
    "--recourse_method",
    nargs="*",
    default=[
        "cchvae"
    ],
    choices=[
        "cchvae"
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

for data_name in args.dataset:
    print(f"Training data set: {data_name}")
    dataset = OnlineCatalog(data_name)

    torch.manual_seed(0)
    
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
    
    nb_immutables = len(dataset.immutables)

    if data_name == 'adult':
        N_TEST = [10, 50, 100, 500, 1000, 2500, 5000, 10000]
    elif data_name == 'give_me_some_credit':
        N_TEST = [10, 50, 100, 500, 1000, 1900]


    fitting_start = time.time()
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
    fitting_end = time.time()

    for n_test in N_TEST:
        
        sampling_start = time.time()
        test_factual = factuals.iloc[:n_test]

        df_cfs = cchvae.get_counterfactuals(test_factual)
        sampling_end = time.time()
        
        
        df_cfs.index = test_factual.index
        df_cfs.insert(0, 'method', 'cchvae')
        df_cfs.insert(1, 'data', data_name)
        
        df_cfs['time (seconds)'] = (fitting_end - fitting_start) + (sampling_end - sampling_start)
        df_cfs['fitting (seconds)'] = (fitting_end - fitting_start)
        df_cfs['sampling (seconds)'] = (sampling_end - sampling_start)
        
        df_cfs['n_test'] = n_test

        path_all = os.path.join(path, f"{data_name}_carla_results_n_several_{device}.csv")
        
        if(os.path.exists(path_all)):
            df_cfs.to_csv(path_all, mode='a', header=False)
        else:
            df_cfs.to_csv(path_all)
