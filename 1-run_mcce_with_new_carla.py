import argparse
import pandas as pd
import time

import numpy as np
from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

import torch

    
from mcce import MCCE

parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    default=["adult"],
    choices=["adult", "give_me_some_credit", "compas"],
    help="Datasets for experiment",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=10,
    help="Number of instances per dataset",
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    default=100,
    help="Number generated counterfactuals per test observation",
)

args = parser.parse_args()

n_test = args.number_of_samples
seed = 1
K = args.k
results_all = None

# Use CARLA to load dataset and predictive model
print("Loading data from Carla...")

for data_name in args.dataset:

    dataset = OnlineCatalog(data_name)
    
    # (1) Load predictive model and predict probabilities

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
        force_train=True, # don't forget to add this or it might load an older model from disk
        )
    elif data_name == 'give_me_some_credit':
        ml_model.train(
        learning_rate=0.002,
        epochs=20,
        batch_size=2048,
        hidden_size=[18, 9, 3],
        force_train=True, # don't forget to add this or it might load an older model from disk
        )
    elif data_name == 'compas':
        ml_model.train(
        learning_rate=0.002,
        epochs=25,
        batch_size=25,
        hidden_size=[18, 9, 3],
        force_train=True, # don't forget to add this or it might load an older model from disk
        )

    # (2) Find unhappy customers and choose which ones to make counterfactuals for
    
    factuals = predict_negative_instances(ml_model, dataset.df)
    test_factual = factuals.iloc[:n_test]
    # test_factual_inverse = dataset.inverse_transform(test_factual)
    
    y_col = dataset.target
    features_and_response = dataset.df.columns
    cont_feat = dataset.continuous
    cat_feat = [x for x in features_and_response if x not in cont_feat] #  these have new names since encode_normalize_order_factuals()
    
    if data_name == 'adult' : 
        fixed_features = ['age', 'sex_Male']
        immutables = ['age', 'sex']
    elif data_name == 'give_me_some_credit':
        fixed_features = ['age']
        immutables = ['age']
    elif data_name == 'compas':
        fixed_features = ['age', 'sex_Male', 'race_White']
        immutables = ['age', 'sex', 'race']
    
    #  Create dtypes for MCCE()
    dtypes = dict([(x, "float") for x in cont_feat])
    for x in cat_feat:
        dtypes[x] = "category"
    df = (dataset.df).astype(dtypes)

    start = time.time()

    print(dataset.inverse_transform(factuals).loc[263])

    # (3) Fit MCCE object
    print("Fitting MCCE model...")
    mcce = MCCE(fixed_features=fixed_features, immutables=immutables, model=ml_model, seed=1,\
        continuous=cont_feat, categorical=cat_feat)

    mcce.fit(df.drop(y_col, axis=1), dtypes)

    print("Generating counterfactuals with MCCE...")
    synth_df = mcce.generate(test_factual.drop(y_col, axis=1), k=K)

    print(dataset.inverse_transform(synth_df).loc[263])

    # (4) Postprocess generated counterfactuals
    print("Postprocessing counterfactuals with MCCE...")
    mcce.postprocess(data=df, synth=synth_df, test=test_factual, response=y_col, \
    inverse_transform=dataset.inverse_transform, cutoff=0.5)
    
    print(dataset.inverse_transform(mcce.results_sparse).loc[263])

    timing = time.time() - start
    print(f"timing: {timing}")
    
    mcce.results_sparse['time (seconds)'] = timing

    # (5) Print results 
    mcce.results_sparse.to_csv(f"/nr/samba/user/anr/pkg/MCCE_Python/Results/{data_name}_mcce_results_k_{K}_n_{n_test}.csv")
    

    results = mcce.results_sparse.copy()
    results['data'] = data_name
    results['method'] = 'mcce'
    results.rename(columns={'violation': 'violations'}, inplace=True)

    preds = ml_model.predict_proba(results)
    new_preds = []
    for x in preds:
        new_preds.append(x[1])
    results['prediction'] = new_preds
    results = dataset.inverse_transform(results)
    results.head(1)

    results['validity'] = np.where(np.asarray(new_preds) >= 0.5, 1, 0)

    results.to_csv(f"/nr/samba/user/anr/pkg/MCCE_Python/Results/{data_name}_mcce_results_k_{K}_n_{n_test}_inverse_transform.csv")



