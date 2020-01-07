import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import matplotlib.pyplot as plt

class LDAClassifier(LDA):

    def __init__(self):
        pass

    def fit(self, x, y):
        x = pd.DataFrame(x)
        self.x = x.values.reshape(x.shape[1], -1)
        self.y = y
        self.sigma = x.cov().values
        self.mean_vectors = []
        self.p_list = []
        for c in np.unique(y):
            self.mean_vectors.append(x[y == c].mean().values)
            self.p_list.append(np.count_nonzero(y == c) / len(y))

    def predict(self, x):
        return self.map(x, self.mean_vectors, self.sigma, self.p_list)

    def get_lda_score(self, x, mu_k, sigma, p_k):
        """
        :param mu_k:
        :param sigma:
        :param pi_k:
        :return: Returns the value of the linear discriminant score function for a given class "k" and a given x value X
        """
        return (np.log(p_k) - 1 / 2 * (mu_k).T @ np.linalg.inv(sigma) @ (mu_k) + x.T @ np.linalg.inv(sigma) @ (
            mu_k)).flatten()[0]

    def map(self, x, mu_list, sigma, p_list):
        '''
        :param mu_list:
        :param sigma:
        :param pi_list:
        :param y:
        :return: Returns the class for which the the linear discriminant score function is largest
        '''
        scores_list = []

        classes = np.unique(self.y)

        for p in range(len(classes)):
            score = self.get_lda_score(x.reshape(-1, 1), mu_list[p].reshape(-1, 1), sigma, p_list[p])
            scores_list.append(score)
        return classes[np.argmax(scores_list)]

    def score(self, x, y):
        err = 0
        y = y.reshape(x.shape[0], -1)
        for i in range(x.shape[0]):
            if(self.predict(x[i,:]) != y[i]):
                err += 1
        err = err/x.shape[0]
        #print(err)
        return err

