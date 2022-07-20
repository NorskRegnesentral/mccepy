import os
import argparse
import time
import torch

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

from mcce import MCCE

PATH = '../../Results2/experiment1/'

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
    
    y_col = dataset.target
    cont_feat = dataset.continuous
    
    cat_feat = dataset.categorical
    cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)

    if data_name == 'adult': 
        fixed_features = ['age', 'sex']
        fixed_features_encoded = ['age', 'sex_Male']
    elif data_name == 'give_me_some_credit':
        fixed_features = ['age']
        fixed_features_encoded = ['age']
    elif data_name == 'compas':
        fixed_features = ['age', 'sex', 'race']
        fixed_features_encoded = ['age', 'sex_Male', 'race_Other']

    #  Create dtypes for MCCE()
    dtypes = dict([(x, "float") for x in cont_feat])
    for x in cat_feat_encoded:
        dtypes[x] = "category"
    df = (dataset.df).astype(dtypes)

    start = time.time()

    # (3) Fit MCCE object
    mcce = MCCE(fixed_features=fixed_features,\
        fixed_features_encoded=fixed_features_encoded,
            continuous=dataset.continuous, categorical=dataset.categorical,\
                model=ml_model, seed=1)

    mcce.fit(df.drop(dataset.target, axis=1), dtypes)

    synth_df = mcce.generate(test_factual.drop(dataset.target, axis=1), k=100)
    mcce.postprocess(data=df, synth=synth_df, test=test_factual, response=y_col, \
        inverse_transform=dataset.inverse_transform, cutoff=0.5)

    timing = time.time() - start

    mcce.results_sparse['time (seconds)'] = timing

    # (5) Save results 
    mcce.results_sparse.to_csv(os.path.join(PATH, "{data_name}_mcce_results_k_{K}_n_{n_test}.csv"))
    
    # (6) Print the counterfactuals inverted to their original feature values/ranges
    results = mcce.results_sparse.copy()
    results['data'] = data_name
    results['method'] = 'mcce'
    results['prediction'] = ml_model.predict_proba(results)[:, [1]]
    
    # preds = ml_model.predict_proba(results)
    # new_preds = []
    # for x in preds:
    #     new_preds.append(x[1])
    # results['prediction'] = new_preds
    results = dataset.inverse_transform(results)

    # results['validity'] = np.where(np.asarray(new_preds) >= 0.5, 1, 0)

    results.to_csv(os.path.join(PATH, f"{data_name}_mcce_results_k_{K}_n_{n_test}_inverse_transform.csv"))




