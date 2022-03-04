import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def get_delta(instance: np.ndarray, cf: np.ndarray):
    """
    Compute difference between original instance and counterfactual
    Parameters
    ----------
    instance: np.ndarray
        Normalized and encoded array with factual data.
        Shape: NxM
    cf: : np.ndarray
        Normalized and encoded array with counterfactual data.
        Shape: NxM
    Returns
    -------
    np.ndarray
    """
    return cf - instance

def d1_distance(delta: np.ndarray): # L0 norm
    """
    Computes D1 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """
    # compute elements which are greater than 0
    return np.sum(delta != 0, axis=1, dtype=float).tolist()


def d2_distance(delta: np.ndarray): # L1 norm
    """
    Computes D2 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """

    return np.sum(np.abs(delta), axis=1, dtype=float).tolist()

def d3_distance(delta: np.ndarray): # L2 norm
    """
    Computes D3 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """
    return np.sum(np.square(np.abs(delta)), axis=1, dtype=np.float).tolist()


def yNN(
    data,
    counterfactuals,
    response,
    y):
    """
    Parameters
    ----------
    data: pd.DataFrame, original training data
    counterfactuals: pd.DataFrame, possible counterfactuals
    response: string, name of response
    y: Number of neighbours
    Returns
    -------
    List[float]
    """
    
    results = []
    N = counterfactuals.shape[0]

    nbrs = NearestNeighbors(n_neighbors=y).fit(data.values) # data includes response

    for i, row in counterfactuals.iterrows():
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), y, return_distance=False)[0]
        cf_label = row[response]
        
        number_of_diff_labels = 0
        for idx in knn:
            neighbour = data.iloc[idx]
            
            neighbour_label = neighbour[response]

            number_of_diff_labels += np.abs(cf_label - neighbour_label)
        results.append(1 - (1 / (N * y)) * number_of_diff_labels)

    # print(1 - (1 / (N * y)) * number_of_diff_labels)
    return results

def feasibility(
    data,
    counterfactuals,
    response,
    y):
    """
    Parameters
    ----------
    data: pd.DataFrame, original training data
    counterfactuals: pd.DataFrame, possible counterfactuals
    response: string, name of response
    y: Number of neighbours
    Returns
    -------
    List[float]
    """
    cols = data.columns
    cols.drop(response)

    results = []
    nbrs = NearestNeighbors(n_neighbors=y).fit(data[cols].values)

    for i, row in counterfactuals[cols].iterrows():
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), y, return_distance=True)[0]
        
        results.append(np.mean(knn))
    return results



def redundancy(
    counterfactuals,
    factuals,
    model,
    response
):
    """
    Parameters
    ----------
    counterfactuals: pd.DataFrame, possible counterfactuals
    factuals: pd.DataFrame, original test observations
    model: fit model object
    response: string, name of response
    Returns
    -------
    List[float]
    """

    df_fact = factuals.reset_index(drop=True)
    df_cfs = counterfactuals.reset_index(drop=True)

    labels = df_cfs[response]
    df_cfs = df_cfs.drop(response, axis=1)
    df_fact = df_fact.drop(response, axis=1)
    
    df_cfs["redundancy"] = df_cfs.apply(
        lambda x: compute_redundancy(
            df_fact.iloc[x.name].values, x.values, model, labels.iloc[x.name]
        ),
        axis=1,
    )
    return df_cfs["redundancy"].values.reshape((-1, 1))


def compute_redundancy(
    fact, 
    cf, 
    mlmodel, 
    label_value):
    red = 0
    for col_idx in range(cf.shape[0]):  # input array has one-dimensional shape
        if fact[col_idx] != cf[col_idx]:
            temp_cf = np.copy(cf)

            temp_cf[col_idx] = fact[col_idx]

            temp_pred = mlmodel.predict(temp_cf.reshape((1, -1)))

            if temp_pred == label_value:
                red += 1

    return red


def constraint_violation(
    counterfactuals, 
    factuals,
    cont_feat,
    fixed_features,
    scaler=None,
):
    """
    Parameters
    ----------
    counterfactuals: pd.DataFrame, possible counterfactuals
    factuals: pd.DataFrame, original test observations
    model: fit model object
    cont_feat: list, continuous features
    fixed_feat: list, fixed features
    scaler: if continuous features have been scaled, pass scaler here
    Returns
    -------
    List[float]
    """
    immutables = fixed_features

    df_decoded_cfs = counterfactuals.copy()
    df_decoded_fact = factuals.copy()

    if scaler: 
        # Decode counterfactuals to compare immutables with not encoded factuals
        df_decoded_cfs[cont_feat] = scaler.inverse_transform(df_decoded_cfs[cont_feat])
        df_decoded_cfs[cont_feat] = df_decoded_cfs[cont_feat].astype("int64")  # avoid precision error

        df_decoded_fact[cont_feat] = scaler.inverse_transform(df_decoded_fact[cont_feat])
        df_decoded_fact[cont_feat] = df_decoded_fact[cont_feat].astype("int64")  # avoid precision error
    

    df_decoded_cfs = df_decoded_cfs[immutables]
    df_factuals = df_decoded_fact[immutables]

    logical = df_factuals != df_decoded_cfs
    logical = np.sum(logical.values, axis=1).reshape((-1, 1))
    
    return logical


def success_rate(counterfactuals):
    """
    Computes success rate for all counterfactuals
    Parameters
    ----------
    counterfactuals: All counterfactual examples inclusive nan values
    Returns
    -------
    """
    return (counterfactuals.dropna().shape[0]) / counterfactuals.shape[0]