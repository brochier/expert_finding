import numpy as np
import expert_finding.preprocessing.text.dictionary
import expert_finding.preprocessing.text.vectorizers
import scipy.sparse
import logging

logger = logging.getLogger()

class Model:

    def __init__(self):
        pass

    def fit(self, A_da, A_dd, T):
        self.A_da=A_da
        logger.debug("Building vocab")
        self.vocab = expert_finding.preprocessing.text.dictionary.Dictionary(T, min_df=5, max_df_ratio=0.25)
        logger.debug("Building tfidf vectors")
        self.docs_vectors = expert_finding.preprocessing.text.vectorizers.get_tfidf_dictionary(self.vocab)

    def predict(self, d, mask = None):
        query_vector = self.docs_vectors[d]
        documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T).A)
        documents_sorting_indices = documents_scores.argsort()[::-1]
        document_ranks = documents_sorting_indices.argsort() + 1
        # Sort scores and get ranks
        candidates_scores = np.ravel(
            self.A_da.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
                axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b
        if mask is not None:
            candidates_scores = candidates_scores[mask]
        return candidates_scores





