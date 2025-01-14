from sktime.transformations.base import BaseTransformer


class AttacvhAreaConsumerType(BaseTransformer):

    def _transform(self, X, y=None):
        X["area_exog"] = X.index.get_level_values(0)
        X["consumer_type_exog"] = X.index.get_level_values(1)
        return X
    
    def _inverse_transform(self, X, y=None):
        return X.drop(columns=["area_exog", "consumer_type_exog"])