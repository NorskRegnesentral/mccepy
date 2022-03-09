import yaml
import argparse
import pandas as pd
from carla import DataCatalog, MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods import * # Load all recourse methods

from carla.evaluation.distances import get_distances
from carla.evaluation.nearest_neighbours import yNN
from carla.evaluation.redundancy import redundancy
from carla.evaluation.success_rate import success_rate
from carla.evaluation.violations import constraint_violation

from tensorflow import Graph, Session # for CEM and CEM_VAE

def load_setup():
    with open("experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)

    return setup_catalog["recourse_methods"]

parser = argparse.ArgumentParser(description="Fit various recourse methods from CARLA.")
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    default=["adult", "give_me_some_credit"],
    choices=["adult", "give_me_some_credit"],
    help="Datasets for experiment",
)
parser.add_argument(
    "-r",
    "--recourse_method",
    nargs="*",
    default=[
        "dice",
        "cem",
        "cem-vae",
        "gs",
    ],
    choices=[
        "dice",
        "cem",
        "cem-vae",
        "gs",
    ],
    help="Recourse methods for experiment",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=3,
    help="Number of instances per dataset",
)

args = parser.parse_args()
setup = load_setup()

n_test = args.number_of_samples

results_all = None

# Use CARLA to load dataset and predictive model
print("Loading data from Carla...")

for data_name in args.dataset:
    print(f"Loading {data_name} dataset.")

    dataset = DataCatalog(data_name)
    y_col = dataset.target
    cont_feat = dataset.continous

    pd.set_option('display.max_columns', None)

    # ------------------ CEM ---------------
    
    rm = 'cem'
    if(rm in args.recourse_method):
        # Load hyperparameters of model
        hyperparams = setup[rm]["hyperparams"]
        hyperparams["data_name"] = data_name
        graph = Graph()
        with graph.as_default():
            ann_sess = Session()
            with ann_sess.as_default():
                # Load predictive model and extract observations that need counterfactuals
                mlmodel_sess = MLModelCatalog(dataset, "ann", "tensorflow")
                
                factuals_sess = predict_negative_instances(mlmodel_sess, dataset)
                factuals_sess = factuals_sess.iloc[: n_test]
                factuals_sess = factuals_sess.reset_index(drop=True)

                # Load recourse method        
                recourse_method_sess = CEM(ann_sess, mlmodel_sess, hyperparams)

                # Generate counterfactual explanations
                counterfactuals = recourse_method_sess.get_counterfactuals(factuals_sess)

                enc_norm_factuals = recourse_method_sess.encode_normalize_order_factuals(factuals=factuals_sess, with_target=True)

                arr_f = enc_norm_factuals.drop(y_col, axis=1).to_numpy()
                arr_cf = counterfactuals.drop(y_col, axis=1).to_numpy()
                distance = get_distances(arr_f, arr_cf) # n_test values per data.frame

                ynn = yNN(counterfactuals, recourse_method_sess, mlmodel_sess, 5) # 1 value per data.frame
                redund = redundancy(enc_norm_factuals, counterfactuals, mlmodel_sess) # n_test values per data.frame
                success = success_rate(counterfactuals) # 1 value per data.frame
                violation = constraint_violation(mlmodel_sess, counterfactuals, factuals_sess) # n_test values per data.frame

                results_method = counterfactuals.copy()
                results_method[cont_feat] = mlmodel_sess.scaler.inverse_transform(results_method[cont_feat])

                distance = pd.DataFrame(distance, columns=['L0', 'L1', 'L2', "L_inf"])
                results_method['L0'] = distance['L0']
                results_method['L1'] = distance['L1']
                results_method['L2'] = distance['L2']
                results_method['yNN'] = ynn
                results_method['feasibility'] = 0
                results_method['feasibility'] = 0
                results_method['redundancy'] = [item for sublist in redund for item in sublist]
                results_method['success'] = success
                results_method['violation'] = [item for sublist in violation for item in sublist]
                results_method.insert(0, 'method', rm)
                results_method.insert(1, 'data', data_name)
                
                if results_all is None:
                    results_all = results_method
                else:
                    results_all = pd.concat([results_all, results_method], axis=0)

    # ------------------ GS ---------------


    rm = 'gs'
    if(rm in args.recourse_method):
        mlmodel = MLModelCatalog(dataset, "ann", "tensorflow")
                
        factuals = predict_negative_instances(mlmodel, dataset)
        factuals = factuals.iloc[: n_test]
        factuals = factuals.reset_index(drop=True)

        recourse_method = GrowingSpheres(mlmodel)
        counterfactuals = recourse_method.get_counterfactuals(factuals)

        enc_norm_factuals = recourse_method.encode_normalize_order_factuals(factuals=factuals, with_target=True)

        arr_f = enc_norm_factuals.drop(y_col, axis=1).to_numpy()
        arr_cf = counterfactuals.drop(y_col, axis=1).to_numpy()
        distance = get_distances(arr_f, arr_cf) # n_test values per data.frame

        ynn = yNN(counterfactuals, recourse_method, mlmodel, 5) # 1 value per data.frame
        redund = redundancy(enc_norm_factuals, counterfactuals, mlmodel) # n_test values per data.frame
        success = success_rate(counterfactuals) # 1 value per data.frame
        violation = constraint_violation(mlmodel, counterfactuals, factuals) # n_test values per data.frame

        results_method = counterfactuals.copy()
        results_method[cont_feat] = mlmodel.scaler.inverse_transform(results_method[cont_feat])

        distance = pd.DataFrame(distance, columns=['L0', 'L1', 'L2', "L_inf"])
        results_method['L0'] = distance['L0']
        results_method['L2'] = distance['L2']
        results_method['yNN'] = ynn
        results_method['feasibility'] = 0
        results_method['feasibility'] = 0
        results_method['redundancy'] = [item for sublist in redund for item in sublist]
        results_method['success'] = success
        results_method['violation'] = [item for sublist in violation for item in sublist]
        results_method.insert(0, 'method', rm)
        results_method.insert(1, 'data', data_name)
        
        if results_all is None:
            results_all = results_method
        else:
            results_all = pd.concat([results_all, results_method], axis=0)

    # ------------------ DiCE ---------------


    rm = 'dice'
    if(rm in args.recourse_method):
        # Load hyperparameters of model
        hyperparams = setup[rm]["hyperparams"]

        mlmodel = MLModelCatalog(dataset, "ann", "tensorflow")
                
        factuals = predict_negative_instances(mlmodel, dataset)
        factuals = factuals.iloc[: n_test]
        factuals = factuals.reset_index(drop=True)

        recourse_method = Dice(mlmodel, hyperparams)
        mlmodel.use_pipeline = True
        counterfactuals = recourse_method.get_counterfactuals(factuals)

        enc_norm_factuals = recourse_method.encode_normalize_order_factuals(factuals=factuals, with_target=True)

        arr_f = enc_norm_factuals.drop(y_col, axis=1).to_numpy()
        arr_cf = counterfactuals.drop(y_col, axis=1).to_numpy()
        distance = get_distances(arr_f, arr_cf) # n_test values per data.frame

        mlmodel.use_pipeline = False # if False, will not normalize features
        ynn = yNN(counterfactuals, recourse_method, mlmodel, 5) # 1 value per data.frame
        redund = redundancy(enc_norm_factuals, counterfactuals, mlmodel) # n_test values per data.frame
        success = success_rate(counterfactuals) # 1 value per data.frame
        violation = constraint_violation(mlmodel, counterfactuals, factuals) # n_test values per data.frame

        results_method = counterfactuals.copy()
        results_method[cont_feat] = mlmodel.scaler.inverse_transform(results_method[cont_feat])

        distance = pd.DataFrame(distance, columns=['L0', 'L1', 'L2', "L_inf"])
        results_method['L0'] = distance['L0']
        results_method['L2'] = distance['L2']
        results_method['yNN'] = ynn
        results_method['feasibility'] = 0
        results_method['feasibility'] = 0
        results_method['redundancy'] = [item for sublist in redund for item in sublist]
        results_method['success'] = success
        results_method['violation'] = [item for sublist in violation for item in sublist]
        results_method.insert(0, 'method', rm)
        results_method.insert(1, 'data', data_name)

        if results_all is None:
            results_all = results_method
        else:
            results_all = pd.concat([results_all, results_method], axis=0)



    # ------------------ CEM-VAE ---------------
    # The only difference between CEM and CEM-VAE is that CEM has gamma = 0 and CEM-VAE has gamma = 1

    rm = 'cem-vae'
    if(rm in args.recourse_method):
        # Load hyperparameters of model
        hyperparams = setup[rm]["hyperparams"]
        hyperparams["data_name"] = data_name
        graph = Graph()
        with graph.as_default():
            ann_sess = Session()
            with ann_sess.as_default():
                # Load predictive model and extract observations that need counterfactuals
                mlmodel_sess = MLModelCatalog(dataset, "ann", "tensorflow")
                
                factuals_sess = predict_negative_instances(mlmodel_sess, dataset)
                factuals_sess = factuals_sess.iloc[: n_test]
                factuals_sess = factuals_sess.reset_index(drop=True)

                # Load recourse method        
                recourse_method_sess = CEM(ann_sess, mlmodel_sess, hyperparams)

                # Generate counterfactual explanations
                counterfactuals = recourse_method_sess.get_counterfactuals(factuals_sess)

                enc_norm_factuals = recourse_method_sess.encode_normalize_order_factuals(factuals=factuals_sess, with_target=True)

                arr_f = enc_norm_factuals.drop(y_col, axis=1).to_numpy()
                arr_cf = counterfactuals.drop(y_col, axis=1).to_numpy()
                distance = get_distances(arr_f, arr_cf) # n_test values per data.frame

                ynn = yNN(counterfactuals, recourse_method_sess, mlmodel_sess, 5) # 1 value per data.frame
                redund = redundancy(enc_norm_factuals, counterfactuals, mlmodel_sess) # n_test values per data.frame
                success = success_rate(counterfactuals) # 1 value per data.frame
                violation = constraint_violation(mlmodel_sess, counterfactuals, factuals_sess) # n_test values per data.frame

                results_method = counterfactuals.copy()
                results_method[cont_feat] = mlmodel_sess.scaler.inverse_transform(results_method[cont_feat])

                distance = pd.DataFrame(distance, columns=['L0', 'L1', 'L2', "L_inf"])
                results_method['L0'] = distance['L0']
                results_method['L2'] = distance['L2']
                results_method['yNN'] = ynn
                results_method['feasibility'] = 0
                results_method['feasibility'] = 0
                results_method['redundancy'] = [item for sublist in redund for item in sublist]
                results_method['success'] = success
                results_method['violation'] = [item for sublist in violation for item in sublist]
                results_method.insert(0, 'method', rm)
                results_method.insert(1, 'data', data_name)

                if results_all is None:
                    results_all = results_method
                else:
                    results_all = pd.concat([results_all, results_method], axis=0)


# print(results_all)
recourse_methods_all = '_'.join(args.recourse_method)
datasets_all = '_'.join(args.dataset)

results_all.to_csv(f"Results/Carla_recourse_method_{recourse_methods_all}_{datasets_all}.csv")

