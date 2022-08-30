import numpy as np
from sklearn.neighbors import NearestNeighbors

def distance(cfs, 
             fact, 
             dataset, 
             higher_card=False):
    """
    Calculates three distance functions between potential counterfactuals and original factuals.
    
    Parameters
    ----------
    cfs : pd.DataFrame
        pd.DataFrame containing the potential counterfactuals (in their standardized/encoded versions).
    fact : pd.DataFrame
        pd.DataFrame containing the factuals (in their standardized/encoded versions).
    dataset : mcce.Data
        Object containing various attributes of the trained dataset.
    higher_card : bool
        If True, the categorical features are allowed to have more than two levels. 
        If False, the categorical features are binarized.
    
    
    Returns
    -------
        Dictionary containing the three distances: L0 (sparsity), L1 (Manhattan distance), L2 (Euclidean distance). 
    """
    
    cfs.sort_index(inplace=True)
    fact.sort_index(inplace=True)
    
    cont_feat = dataset.continuous
    cat_feat = dataset.categorical
    cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)


    if higher_card:
        cf_inverse_transform = dataset.inverse_transform(cfs.copy())
        fact_inverse_transform = dataset.inverse_transform(fact.copy())

        cfs_categorical = cf_inverse_transform[cat_feat].sort_index().to_numpy()
        factual_categorical = fact_inverse_transform[cat_feat].sort_index().to_numpy()
    

    else:
        cfs_categorical = cfs[cat_feat_encoded].sort_index().to_numpy()
        factual_categorical = fact[cat_feat_encoded].sort_index().to_numpy()

    cfs_continuous = cfs[cont_feat].sort_index().to_numpy()
    factual_continuous = fact[cont_feat].sort_index().to_numpy()
    
    delta_cont = factual_continuous - cfs_continuous
    delta_cat = factual_categorical - cfs_categorical
    delta_cat = np.where(np.abs(delta_cat) > 0, 1, 0)

    delta = np.concatenate((delta_cont, delta_cat), axis=1)

    L0 = np.sum(np.invert(np.isclose(delta, np.zeros_like(delta), atol=1e-05)), axis=1, dtype=np.float).tolist()
    L1 = np.sum(np.abs(delta), axis=1, dtype=np.float).tolist()
    L2 = np.sum(np.square(np.abs(delta)), axis=1, dtype=np.float).tolist()

    return({'L0': L0, 'L1': L1, 'L2': L2})


def feasibility(cfs,
                df,
                cols,
                ):
    """
    Calculates the feasibility metric between the counterfactual explanations and the closest observations in the 
    original dataset (df).
    
    Parameters
    ----------
    cfs : pd.DataFrame
        pd.DataFrame containing the counterfactuals (in their standardized/encoded versions).
    df : pd.DataFrame
        pd.DataFrame containing the original training data. The features must be standardized/encoded in the 
        same way as the cfs.
    cols : list
        List containing the features to calculate feasibility on. 
    
    Returns
    -------
        List containing the feasibility metric for each counterfactual in cfs. 
    """
    
    nbrs = NearestNeighbors(n_neighbors=5).fit(df[cols].values)

    results = []
    for _, row in cfs[cols].iterrows():
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), 5, return_distance=True)[0]
        
        results.append(np.mean(knn))

    return results


def constraint_violation(
    decoded_cfs,
    decoded_factuals,
    dataset,
    ):
    """
    Calculates the feasibility metric between the counterfactual explanations and the closest observations in the 
    original dataset (df).
    
    Parameters
    ----------
    decoded_cfs : pd.DataFrame
        pd.DataFrame containing the counterfactuals (in their original decoded feature versions).
    decoded_factuals : pd.DataFrame
        pd.DataFrame containing the original training data. The features must be in their original 
        decoded feature versions.
    dataset : mcce.Data
        Object containing various attributes of the trained dataset.
    
    Returns
    -------
        List containing the number of violations (i.e., fixed feature changes) per counterfactual. 
    """

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    continuous = dataset.continuous
    categorical = dataset.categorical
    fixed_features = dataset.immutables

    decoded_cfs.sort_index(inplace=True)
    decoded_factuals.sort_index(inplace=True)

    cfs_continuous_fixed = decoded_cfs[intersection(continuous, fixed_features)]
    factual_continuous_fixed = decoded_factuals[intersection(continuous, fixed_features)]

    continuous_violations = np.invert(
        np.isclose(cfs_continuous_fixed, factual_continuous_fixed)
    )
    continuous_violations = np.sum(continuous_violations, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    # check categorical by boolean comparison
    cfs_categorical_fixed = decoded_cfs[intersection(categorical, fixed_features)]
    factual_categorical_fixed = decoded_factuals[intersection(categorical, fixed_features)]
    
    categorical_violations = cfs_categorical_fixed != factual_categorical_fixed
    categorical_violations = np.sum(categorical_violations.values, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    return (continuous_violations + categorical_violations)


def success_rate(
    cfs, 
    ml_model, 
    cutoff=0.5):
    """
    Calculates the feasibility metric between the counterfactual explanations and the closest observations in the 
    original dataset (df).
    
    Parameters
    ----------
    cfs : pd.DataFrame
        pd.DataFrame containing the counterfactuals (in their standardized/encoded versions).
    ml_model : 
        trained prediction model, for example an sklearn model. 
        Must contain a predict_proba function.
    cutoff : float, default: 0.5
        Cutoff value that indicates which observations get a positive/desired response and which do not. 
        Observations with a prediction greater than this cutoff value are considered "positive"
        and those with a lower prediction are considered "negative".
        
    Returns
    -------
        List containing a 1 if the cfs has a prediction probability greater than the cutoff
        and a 0 otherwise.
    """
    
    preds = ml_model.predict_proba(cfs)[:, [1]]
    preds = preds >= cutoff
    # {'success': preds>=cutoff, 'prediction': preds}
    return ([int(x) for x in preds])