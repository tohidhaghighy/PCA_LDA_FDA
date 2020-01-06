import pandas as pd
import numpy as np

class FDA:
    def __init__(self,n_components=2):
        self.n_components=n_components
        self.FDA_List=[]

    def LoadFile(self,name):
        return pd.read_csv(name, header=None)

    def OnlyData(self,data):
        x = data.loc[:, 0:3]
        y = data.loc[:, [4]]
        return x

    def get_lda_score(self,X,MU_k,SIGMA,pi_k): 
        #Returns the value of the linear discriminant score function for a given class "k" and 
        # a given x value X
        print("-----------------------------")
        print(X)
        print(MU_k)
        print(SIGMA)
        print(pi_k)
        return (np.log(pi_k) - 1/2 * (MU_k).T @ np.linalg.inv(SIGMA)@(MU_k) + X.T @ np.linalg.inv(SIGMA)@ (MU_k)).flatten()[0]

    def predict_lda_class(self,X,MU_list,SIGMA,pi_list, y):
        scores_list = []
        classes = np.unique(y)
        for p in range(len(classes)):
            score = self.get_lda_score(X.reshape(-1,1),MU_list[p].reshape(-1,1),SIGMA,pi_list[0]) 
            scores_list.append(score)
        self.FDA_List=scores_list
        return classes[np.argmax(scores_list)]

    def PlayFDA(self,x,y):
        X1 = x.iloc[120, :].values.reshape(2, -1)
        pi_k = [50/len(y) for x in range(3)]
        sigma = x.cov().values
        mean_vectors = []
        for c in np.unique(y):
            mean_vectors.append(x[y==c].mean().values)
        top_rank=self.predict_lda_class(X1, mean_vectors, sigma, pi_k, y)
        return top_rank

    def Error_Rate(self):
        pass