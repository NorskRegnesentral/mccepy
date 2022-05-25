import numpy as np
from sklearn.neighbors import NearestNeighbors


def distance(counterfactuals_without_nans, factual_without_nans, ml_model):
    
        
    arr_f = ml_model.get_ordered_features(factual_without_nans).to_numpy()
    arr_cf = ml_model.get_ordered_features(
        counterfactuals_without_nans
    ).to_numpy()

    delta = arr_f - arr_cf 

    d1 = np.sum(np.invert(np.isclose(delta, np.zeros_like(delta))), axis=1, dtype=np.float).tolist()
    d1_old = np.sum(delta.round(2) != 0, axis=1, dtype=np.float).tolist()

    d2 = np.sum(np.abs(delta), axis=1, dtype=np.float).tolist()
    d3 = np.sum(np.square(np.abs(delta)), axis=1, dtype=np.float).tolist()

    return({'L0': d1, 'L1': d2, 'L2': d3})



def feasibility(
    counterfactuals_without_nans,
    factual_without_nans,
    dataset
    ):
    

    cols = dataset.df.columns
    cols.drop(dataset.target)

    nbrs = NearestNeighbors(n_neighbors=5).fit(factual_without_nans[cols].values)

    results = []
    for i, row in counterfactuals_without_nans[cols].iterrows():
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), 5, return_distance=True)[0]
        
        results.append(np.mean(knn))

    return results


def constraint_violation(
    counterfactuals_without_nans, 
    factual_without_nans,
    dataset,
    ):
    
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))


    df_decoded_cfs = dataset.inverse_transform(counterfactuals_without_nans.copy())

    df_factuals = dataset.inverse_transform(factual_without_nans.copy())

    cfs_continuous_immutable = df_decoded_cfs[
        intersection(dataset.continuous, dataset.immutables)
    ]
    factual_continuous_immutable = df_factuals[
        intersection(dataset.continuous, dataset.immutables)
    ]

    continuous_violations = np.invert(
        np.isclose(cfs_continuous_immutable, factual_continuous_immutable)
    )
    continuous_violations = np.sum(continuous_violations, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    # check categorical by boolean comparison
    cfs_categorical_immutable = df_decoded_cfs[
        intersection(dataset.categorical, dataset.immutables)
    ]
    factual_categorical_immutable = df_factuals[
        intersection(dataset.categorical, dataset.immutables)
    ]

    categorical_violations = cfs_categorical_immutable != factual_categorical_immutable
    categorical_violations = np.sum(categorical_violations.values, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    return (continuous_violations + categorical_violations)


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