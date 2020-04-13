import numpy as np
import scipy.sparse
import expert_finding.preprocessing.text.dictionary
import expert_finding.preprocessing.text.vectorizers
import expert_finding.models.idne
from sklearn.preprocessing import normalize
import os
import logging

logger = logging.getLogger()

class Model:

    def __init__(self, limit = 100, min_error = 0.0001):
        self.limit = limit
        self.min_error = min_error
        self.eta = 0.5


    def fit(self, A_da, A_dd, T):
        self.A_da = A_da
        self.A_dd = A_dd

        dd = A_da @ A_da.T
        dd.setdiag(0)
        self.network = dd + A_dd
        self.documents = T

        self.model = expert_finding.models.idne.Model()

        self.model.fit(self.network, self.documents)
        self.embeddings = self.model.get_embeddings()

        self.docs_vectors = normalize(self.embeddings)

        D = self.A_da.shape[0]
        C = self.A_da.shape[1]

        aa_zeros = scipy.sparse.csr_matrix((C,C))

        left_side = scipy.sparse.vstack([aa_zeros, self.A_da])
        right_side = scipy.sparse.vstack([self.A_da.T, A_dd])

        self.bigraph = scipy.sparse.hstack([left_side, right_side])
        self.bigraph = normalize(self.bigraph, axis=0, norm='l1')

    def predict(self, d, mask = None):
        query_vector = self.docs_vectors[d]
        C = self.A_da.shape[1]
        # Create jumping vector
        Pd = np.squeeze(query_vector.dot(self.docs_vectors.T))

        if Pd.sum() > 0:
            Pd = Pd / Pd.sum()
        Pc = np.zeros(self.A_da.shape[1])
        P = np.vstack([Pc.reshape(Pc.shape[0],1), Pd.reshape(Pd.shape[0],1)])
        if P.sum() > 0:
            P = P / P.sum()
        #P = scipy.sparse.crs_matrix(P)

        #  Build x init
        x = np.vstack([np.zeros(C).reshape(C,1), Pd.reshape(Pd.shape[0],1)])
        if x.sum() > 0:
            x = x / x.sum()

        # eta
        eta = self.eta

        # Q
        Q = self.bigraph.copy()
        error = self.min_error
        for i in range(self.limit):
            xprev = x
            #x = (1-eta) * Q.dot(x)
            x = (1 - eta) * Q.dot(x) + eta * P # double walk
            error = np.linalg.norm(x - xprev)
            if error < self.min_error:
                break

        logger.debug(f"Ended propagation at iter ={i+1} with error ={error}")

        # Remove query node
        x = Q.dot(x)
        x = x.reshape((len(x),))
        candidates_scores = x[:C]

        if mask is not None:
            candidates_scores = candidates_scores[mask]

        return candidates_scores

