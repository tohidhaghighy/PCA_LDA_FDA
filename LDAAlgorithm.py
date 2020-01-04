import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

class LDA:
    def __init__(self,n_components=2):
        self.n_components=n_components
        self.LDA_Matrix=[]

    def LoadFile(self,name):
        return pd.read_csv(name, header=None)

    def OnlyData(self,data):
        x = data.loc[:, 0:3]
        y = data.loc[:, [4]]
        return x

    def Label_Encoder(self,y):
        label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}
        enc = LabelEncoder()
        label_encoder = enc.fit(y.values.ravel())
        y = label_encoder.transform(y) + 1
        return y

    
    def fit_transform(self,data):
        mean_vector = self.Get_Mean_Vector(data)
        data -= mean_vector
        return data

    def Mean_Vector(self,x,y):
        mean_vectors = []
        for c in np.unique(y):
            mean_vectors.append(np.mean(x[y==c], axis=0))
        return mean_vectors

    def Scatter_Matrix(self,x,y,mean_vector):
        s_w = np.zeros((x.shape[1], x.shape[1]))
        for c, mv in zip(range(1, x.shape[1]), mean_vector):
            class_sc_mat = np.zeros((x.shape[1], x.shape[1]))
            for index, row in x[y == c].iterrows():
                row1, mv1 = (row.values.reshape(x.shape[1], 1), mv.values.reshape(x.shape[1], 1)) 
                class_sc_mat += (row1 - mv1).dot((row1 - mv1).T)
            s_w += class_sc_mat 
        return s_w  

    def BetweenClass_Scatter(self,x,y,mean_vector):
        overall_mean = np.mean(x, axis=0)
        s_b = np.zeros((x.shape[1], x.shape[1]))
        for i, mean_vec in enumerate(mean_vector):  
            n = x.iloc[y==i+1,:].values.shape[0]
            mean_vec = mean_vec.values.reshape(x.shape[1], 1)
            overall_mean1 = overall_mean.values.reshape(x.shape[1], 1) 
            s_b += n * (mean_vec - overall_mean1).dot((mean_vec - overall_mean1).T)

        return s_b

    def EigenVector(self,s_w,s_b):
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
        return eig_vecs

    def EigenValue(self,s_w,s_b):
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
        return eig_vals

    def Component_Sort(self,eigen_val,count):
        return np.argsort(np.abs(eigen_val))[::-1][0:count]

    def W_Matrix(self,eig_vecs,sorted_vec,x):
        matrix_w = eig_vecs[:, sorted_vec[0]].reshape(x.shape[1], -1)
        for i in range(len(sorted_vec)-1):
            matrix_w = np.hstack((matrix_w, eig_vecs[:,sorted_vec[i+1]].reshape(x.shape[1], -1)))
        return matrix_w

    def LDA_Matrix(self,x,matrix_w):
        final_matrix= x.values @ matrix_w.real
        sns.scatterplot(final_matrix[:,0], final_matrix[:,1])
        self.LDA_Matrix=final_matrix

    def Play_LDA(self,x,y):
        pass

    
    