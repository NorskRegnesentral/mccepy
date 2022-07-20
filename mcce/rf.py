from sklearn.ensemble import RandomForestClassifier

class RandomForestModel():
    """The default way of implementing RandomForest from sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"""

    def __init__(self, data):

        df_train = data.df
        
        x_train = df_train[data.continuous + data.categorical_encoded]
        y_train = df_train[data.target]

        self._feature_input_order = data.continuous + data.categorical_encoded

        param = {
            "max_depth": None,  # determines how deep the tree can go
            "n_estimators": 5, # number of trees
            "min_samples_split": 3 # number of features to consider at each split
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
        return self._mymodel.predict(self.get_ordered_features(x))

    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))
    
    def get_ordered_features(self, x):
        return x[self.feature_input_order]
