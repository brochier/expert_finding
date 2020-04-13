import numpy as np
from sklearn.preprocessing import normalize
import scipy.sparse
import logging
import expert_finding.preprocessing.text.dictionary


logger = logging.getLogger()

class Model:

    def __init__(self, ane_model_class):
        self.ane_model_class = ane_model_class

    def fit(self, A_da, A_dd, T):
        C = A_da.shape[1]
        sparse_zeros_candidates = scipy.sparse.csr_matrix((C, C))

        dd = A_da @ A_da.T + A_dd
        dd.setdiag(0)
        aa = A_da.T @ A_da
        aa.setdiag(0)

        left_side = scipy.sparse.vstack([normalize(aa), normalize(A_da)])
        right_side = scipy.sparse.vstack([normalize(A_da.T), normalize(dd)])
        self.network = scipy.sparse.hstack([left_side, right_side])

        logger.debug("Building vocab")
        self.vocab = expert_finding.preprocessing.text.dictionary.Dictionary(T, min_df=5, max_df_ratio=0.25)
        docs = list()
        for t in T:
            docs.append(" ".join([self.vocab.ids_to_words[w] for w in self.vocab.get_sequence(t)]))

        self.documents = list()
        for i in range(A_da.shape[1]):
            self.documents.append(" ")
            for j in A_da[:,i].nonzero()[0]:
                self.documents[i] += docs[j] + " "

        #for i,t in enumerate(self.documents):
        #    self.documents[i] = " ".join(set().intersection(*t))

        self.documents.extend(docs)

        self.model = self.ane_model_class()
        self.model.fit(self.network, self.documents)
        self.embeddings = normalize(self.model.get_embeddings())

        self.candidate_vectors = self.embeddings[:A_da.shape[1]]
        self.document_vectors = self.embeddings[A_da.shape[1]:]

    def predict(self, d, mask = None):
        if mask is not None:
            candidate_vectors = self.candidate_vectors[mask]
        else:
            candidate_vectors = self.candidate_vectors
        query_vector = self.document_vectors[d]
        candidates_scores = np.squeeze(query_vector.dot(candidate_vectors.T))
        return candidates_scores





