import numpy as np
from sklearn.neighbors import NearestNeighbors


def distance(counterfactuals_without_nans, factual_without_nans, ml_model):
    
    arr_f = ml_model.get_ordered_features(factual_without_nans).to_numpy()
    arr_cf = ml_model.get_ordered_features(counterfactuals_without_nans).to_numpy()

    delta = arr_f - arr_cf 

    d1 = np.sum(np.invert(np.isclose(delta, np.zeros_like(delta), atol=1e-05)), axis=1, dtype=np.float).tolist()
    d1_old = np.sum(delta.round(2) != 0, axis=1, dtype=np.float).tolist()

    d2 = np.sum(np.abs(delta), axis=1, dtype=np.float).tolist()
    d3 = np.sum(np.square(np.abs(delta)), axis=1, dtype=np.float).tolist()

    return({'L0': d1, 'L1': d2, 'L2': d3})



def feasibility(
    counterfactuals_without_nans,
    factual_without_nans,
    cols,
    ):
    
    nbrs = NearestNeighbors(n_neighbors=5).fit(factual_without_nans[cols].values)

    results = []
    for i, row in counterfactuals_without_nans[cols].iterrows():
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), 5, return_distance=True)[0]
        
        results.append(np.mean(knn))

    return results


def constraint_violation(
    df_decoded_cfs,
    df_factuals,
    continuous,
    categorical,
    fixed_features,
    ):
    
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))


    cfs_continuous_immutable = df_decoded_cfs[
        intersection(continuous, fixed_features)
    ]
    factual_continuous_immutable = df_factuals[
        intersection(continuous, fixed_features)
    ]

    continuous_violations = np.invert(
        np.isclose(cfs_continuous_immutable, factual_continuous_immutable)
    )
    continuous_violations = np.sum(continuous_violations, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    # check categorical by boolean comparison
    cfs_categorical_immutable = df_decoded_cfs[
        intersection(categorical, fixed_features)
    ]
    factual_categorical_immutable = df_factuals[
        intersection(categorical, fixed_features)
    ]

    categorical_violations = cfs_categorical_immutable != factual_categorical_immutable
    categorical_violations = np.sum(categorical_violations.values, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    return (continuous_violations + categorical_violations)


def success_rate(counterfactuals_without_nans, ml_model, cutoff=0.5):
    
    preds = ml_model.predict_proba(counterfactuals_without_nans)[:, [1]]
    preds = preds >= cutoff
    # {'success': preds>=cutoff, 'prediction': preds}
    return ([int(x) for x in preds])