from sklearn.ensemble import RandomForestClassifier

class RandomForestModel():
    """
    Trains a random forest model using the sklearn RandomForestClassifier method. 
    
    Parameters
    ----------
    data : mcce.Data
        Object of class mcce.Data which contains the data to train the model on and a set
        of other attributions like which features are continuous, categorical, and fixed.
    
    Methods
    -------
    predict : 
        Predicts the response/target for a data set based on the fitted random forest.
    predict_proba :
        Outputs the predicted probability (between 0 and 1) for a data set based on the fitted random forest.
    get_ordered_features :
        Returns a pd.DataFrame where the features have the same ordering as the original data set. 
    """

    def __init__(self, data):

        df_train = data.df
        
        x_train = df_train[data.continuous + data.categorical_encoded]
        y_train = df_train[data.target]

        self._feature_input_order = data.continuous + data.categorical_encoded

        param = {
            "max_depth": None,  # The maximum depth of the tree. If None, then nodes are expanded until 
                                # all leaves are pure or until all leaves contain less than min_samples_split samples.
            "n_estimators": 200, # The number of trees in the forest.
            "min_samples_split": 3 # The minimum number of samples required to split an internal node:
        }
        self._mymodel = RandomForestClassifier(**param)
        self._mymodel.fit(
                x_train,
                y_train,
            )

    @property
    def feature_input_order(self):
        return self._feature_input_order

    @property
    def backend(self):
        return "TensorFlow"

    @property
    def raw_model(self):
        return self._mymodel

    @property
    def tree_iterator(self):
        # make a copy of the trees, else feature names are not saved
        booster_it = [booster for booster in self.raw_model.get_booster()]
        # set the feature names
        for booster in booster_it:
            booster.feature_names = self.feature_input_order
        return booster_it

    def predict(self, x):
        """
        Predicts the response/target for a data set based on the fitted random forest.
        
        Parameters
        ----------
        x : pd.DataFrame
            DataFrame to predict a response for.
        
        Returns
        -------
            List of predictions
        
        """

        return self._mymodel.predict(self.get_ordered_features(x))

    def predict_proba(self, x):
        """
        Outputs the predicted probability (between 0 and 1) for a data set based on the fitted random forest.
        
        Parameters
        ----------
        x : pd.DataFrame
            DataFrame to predict a response for.
        
        Returns
        -------
            List of predicted probabilities.
        
        """
        return self._mymodel.predict_proba(self.get_ordered_features(x))
    
    def get_ordered_features(self, x):
        """
        Returns a pd.DataFrame where the features have the same ordering as the original data set. 
        
        Parameters
        ----------
        x : pd.DataFrame
            DataFrame to predict a response for.
        
        Returns
        -------
            pd.DataFrame with a potentially new column order
        
        """
        return x[self.feature_input_order]