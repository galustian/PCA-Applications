import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
# from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

class LatentSemanticAnalysis():
    Ut = None
    S = None
    cv = None

    def fit(self, data, n_components=200):
        cv = CountVectorizer()
        cv.fit(data)
        matrix_data = cv.transform(data)  # document vectors in rows

        U, S, Vt = svds(matrix_data.T.astype('float32'), k=n_components)

        self.cv = cv
        self.Ut = U.T
        self.S = S

    # base is the text to compare the data to
    # return indexes of n_most_similar documents (data)
    def predict_most_similar(self, base, data, n_most_similar=10):
        vector_base = self.cv.transform([base])
        matrix_data = self.cv.transform(data)

        assert(vector_base.shape[1] == matrix_data.shape[1])

        reduced_base = self.Ut @ vector_base.T
        reduced_data = self.Ut @ matrix_data.T


        n_most_similar_indexes = []
        n_most_similar_scores = []
        
        for i in range(reduced_data.shape[1]):
            cosine_angle = reduced_data[:, i].T @ reduced_base / (np.linalg.norm(reduced_data[:, i]) * np.linalg.norm(reduced_base))
            n_most_similar_scores.append(cosine_angle[0])
            n_most_similar_indexes.append(i)

        # sort both lists to return n most similar (bubble sort)
        for i1 in range(n_most_similar):
            largest = -999
            for i2 in range(i1, len(n_most_similar_scores)):
                score = n_most_similar_scores[i2]
                
                if score > largest:
                    # swap 2 items
                    n_most_similar_scores[i1], n_most_similar_scores[i2] = n_most_similar_scores[i2], n_most_similar_scores[i1]
                    n_most_similar_indexes[i1], n_most_similar_indexes[i2] = n_most_similar_indexes[i2], n_most_similar_indexes[i1]
                    largest = score

        return n_most_similar_indexes[:10], n_most_similar_scores[:10]