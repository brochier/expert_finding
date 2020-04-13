import numpy as np
from sklearn.preprocessing import normalize
import expert_finding.models.tadw
import scipy.sparse
import logging

logger = logging.getLogger()

class Model:

    def __init__(self):
        pass

    def fit(self, A_da, A_dd, T):
        self.A_da = A_da
        dd = A_da @ A_da.T
        dd.setdiag(0)
        self.network = dd + A_dd
        self.documents = T

        self.model = expert_finding.models.tadw.Model()
        self.model.fit(self.network, self.documents)
        self.embeddings = self.model.get_embeddings()

        self.docs_vectors = normalize(self.embeddings)

    def predict(self, d, mask = None):
        query_vector = self.docs_vectors[d]
        documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T))
        documents_sorting_indices = documents_scores.argsort()[::-1]
        document_ranks = documents_sorting_indices.argsort() + 1
        # Sort scores and get ranks
        candidates_scores = np.ravel(
            self.A_da.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
                axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b
        if mask is not None:
            candidates_scores = candidates_scores[mask]
        return candidates_scores





