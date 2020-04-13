import numpy as np

class Model:
    def __init__(self):
        self.num_candidates = 0

    def fit(self, A_da, A_dd, T):
        self.num_candidates = A_da.shape[1]

    def predict(self, d, mask = None):
        if mask is not None:
            self.num_candidates = len(mask)
        return np.random.rand(self.num_candidates)


