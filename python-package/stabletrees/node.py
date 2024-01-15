class Node:
    def __init__(self, split_value,score, split_feature, num_samples, prediction, y_var, w_var, features_indices):
        self.split_value = split_value
        self.split_feature=split_feature
        self.num_samples = num_samples
        self.prediction = prediction
        self.y_var = y_var
        self.w_var = w_var
        self.score = score
        self.left_child = None
        self.right_child = None
        self.features_indices = features_indices
        self.parent_expected_max_S = None

    def is_leaf(self):
        if self.left_child is None:
            return True
        if self.right_child is None:
            return True
        return False
    
    def get_split_feature(self):
        return self.split_feature
    
    def get_impurity(self):
        return self.score
    
    def get_features_indices(self):
        return self.features_indices
    
    def get_split_value(self):
        return self.split_value
    
    def predict(self):
        return self.prediction
    def nsamples(self):
        return self.num_samples
