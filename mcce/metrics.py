import numpy as np
from sklearn.neighbors import NearestNeighbors

def distance(counterfactuals_without_nans, factual_without_nans, dataset, higher_card=False):
    counterfactuals_without_nans.sort_index(inplace=True)
    factual_without_nans.sort_index(inplace=True)
    
    # arr_f = ml_model.get_ordered_features(factual_without_nans).to_numpy()
    # arr_cf = ml_model.get_ordered_features(counterfactuals_without_nans).to_numpy()

    cont_feat = dataset.continuous
    cat_feat = dataset.categorical
    cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)


    if higher_card:
        cf_inverse_transform = dataset.inverse_transform(counterfactuals_without_nans.copy())
        fact_inverse_transform = dataset.inverse_transform(factual_without_nans.copy())
        cfs_categorical = cf_inverse_transform[cat_feat].sort_index().to_numpy()
        factual_categorical = fact_inverse_transform[cat_feat].sort_index().to_numpy()
    

    else:
        cfs_categorical = counterfactuals_without_nans[cat_feat_encoded].sort_index().to_numpy()
        factual_categorical = factual_without_nans[cat_feat_encoded].sort_index().to_numpy()

    cfs_continuous = counterfactuals_without_nans[cont_feat].sort_index().to_numpy()
    factual_continuous = factual_without_nans[cont_feat].sort_index().to_numpy()
    
    delta_cont = factual_continuous - cfs_continuous
    delta_cat = factual_categorical - cfs_categorical
    delta_cat = np.where(np.abs(delta_cat) > 0, 1, 0)

    delta = np.concatenate((delta_cont, delta_cat), axis=1)

    d1 = np.sum(np.invert(np.isclose(delta, np.zeros_like(delta), atol=1e-05)), axis=1, dtype=np.float).tolist()
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
    df_decoded_cfs.sort_index(inplace=True)
    df_factuals.sort_index(inplace=True)
    
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