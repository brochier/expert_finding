from sklearn.preprocessing import normalize
import expert_finding.preprocessing.text.dictionary
import expert_finding.preprocessing.text.vectorizers
from sklearn.decomposition import TruncatedSVD
import numpy as np
import scipy.sparse.linalg
import logging
import scipy.spatial.distance

logger = logging.getLogger()


class TADW(object):

    def __init__(self, dim=128, lamb=0.2):
        self.lamb = lamb
        self.dim = int(dim/2)
        self.embeddings = None
        self.I = None
        self.J = None
        self.adj = None
        self.M = None
        self.T = None
        self.node_size = None
        self.feature_size = None

    def fit(self, adjacency_matrix, features, num_iterations=20):
        logger.debug("Building vocab")
        self.vocab = expert_finding.preprocessing.text.dictionary.Dictionary(features, min_df=5, max_df_ratio=0.25)
        logger.debug("Building tfidf vectors")
        tfidf_vectors = expert_finding.preprocessing.text.vectorizers.get_tfidf_dictionary(self.vocab)
        logger.debug("Building svd vectors")
        self.svd = TruncatedSVD(n_components=self.dim)
        svd_vectors = self.svd.fit_transform(tfidf_vectors)
        #self.dim = self.svd.components_.shape[0]

        self.adj = adjacency_matrix.A
        #self.adj = svd_vectors @ svd_vectors.T
        self.adj = normalize(self.adj, axis=1, norm='l1')

        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        #self.M = self.adj
        self.M = (self.adj + np.dot(self.adj, self.adj) ) / 2

        # T is feature_size*node_num, text features
        self.T = svd_vectors.T
        self.node_size = self.adj.shape[0]
        self.feature_size = self.T.shape[0]
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)

        # Update
        for i in range(num_iterations):
            logger.debug(f'Iteration {i}')
            # Update W
            B = np.dot(self.H, self.T)
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - \
                  2 * np.dot(B, self.M.T) + self.lamb * self.W
            Hess = 2 * np.dot(B, B.T) + self.lamb * np.eye(self.dim)
            drv = np.reshape(drv, [self.dim * self.node_size, 1])
            rt = -drv
            dt = rt
            vecW = np.reshape(self.W, [self.dim * self.node_size, 1])
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [
                    self.dim * self.node_size, 1])

                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecW = vecW + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            # Update H
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)
                          - np.dot(self.W, self.M.T)), self.T.T) + self.lamb * self.H
            drv = np.reshape(drv, (self.dim * self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim * self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))
                                 + self.lamb * dtS, (self.dim * self.feature_size, 1))
                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecH = vecH + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))

        self.vectors = np.hstack(
            (normalize(self.W.T), normalize(np.dot(self.T.T, self.H.T))))

    def get_embeddings(self):
        return self.vectors


class Model:
    def __init__(self, embedding_size=256, number_iterations=20):
        self.number_iterations = number_iterations
        self.embedding_size = embedding_size
        self.model = TADW(dim=embedding_size, lamb=0.2)

    def fit(self, X, M):
        self.X = X
        self.M = M
        self.model.fit(self.X, self.M, self.number_iterations)

    def get_embeddings(self):
        return self.model.get_embeddings()
