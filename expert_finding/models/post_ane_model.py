import numpy as np
from sklearn.preprocessing import normalize
import scipy.sparse
import logging

logger = logging.getLogger()

class Model:

    def __init__(self, ane_model_class):
        self.ane_model_class = ane_model_class

    def fit(self, A_da, A_dd, T):
        dd = A_da @ A_da.T
        dd.setdiag(0)
        self.network = normalize(normalize(dd) + normalize(A_dd))
        self.documents = T

        self.model = self.ane_model_class()
        self.model.fit(self.network, self.documents)
        self.embeddings = self.model.get_embeddings()

        self.document_vectors = normalize(self.embeddings)
        self.candidate_vectors = normalize(normalize(A_da.T, norm='l1', axis=1) @ self.embeddings)

    def predict(self, d, mask = None):
        if mask is not None:
            candidate_vectors = self.candidate_vectors[mask]
        else:
            candidate_vectors = self.candidate_vectors
        query_vector = self.document_vectors[d]
        candidates_scores = np.squeeze(query_vector.dot(candidate_vectors.T))
        return candidates_scores





