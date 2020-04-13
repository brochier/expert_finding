import numpy as np
import expert_finding.preprocessing.text.dictionary
import expert_finding.preprocessing.text.vectorizers
import logging

logger = logging.getLogger()

class Model:

    def __init__(self):
        self.authors_vectors = None
        self.docs_vectors = None

    def fit(self, A_da, A_dd, T):
        logger.debug("Building vocab")
        self.vocab = expert_finding.preprocessing.text.dictionary.Dictionary(T, min_df=5, max_df_ratio=0.25)
        logger.debug("Building tfidf vectors")
        self.docs_vectors = expert_finding.preprocessing.text.vectorizers.get_tfidf_dictionary(self.vocab)
        authors_metadocs = list()
        for i in range(A_da.shape[1]):
            authors_metadocs.append(" ".join([T[j] for j in A_da[:,i].nonzero()[0] ]))
        self.authors_vectors = expert_finding.preprocessing.text.vectorizers.get_tfidf_N(self.vocab, authors_metadocs)

    def predict(self, d, mask = None):
        query_vector = self.docs_vectors[d]
        candidates_scores = np.squeeze(query_vector.dot(self.authors_vectors.T).A)
        if mask is not None:
            candidates_scores = candidates_scores[mask]
        return candidates_scores