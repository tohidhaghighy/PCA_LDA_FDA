import pandas as pd
import numpy as np

class PCA:
    def __init__(self,n_components=2):
        self.n_components=n_components

    def LoadFile(self,name):
        return pd.read_csv(name, header=None)

    def OnlyData(self,data):
        x = data.loc[:, 0:3]
        y = data.loc[:, [4]]
        return x
    
    def fit_transform(self,data):
        mean_vector = self.Get_Mean_Vector(data)
        data -= mean_vector
        return data

    def Get_Mean_Vector(self,data):
        return data.mean().values

    def Scatter_Matrix(self,data,mean_vector):
        scatter_matrix = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[0]):
            scatter_matrix += (data.iloc[i, :].values.reshape(data.shape[1], 1) - mean_vector).dot((data.iloc[i, :].values.reshape(data.shape[1], 1) - mean_vector).T)
        # eigenvectors and eigenvalues for the from the scatter matrix
        # eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
        return scatter_matrix


    def CoVarianse_Matrix(self,data):
        return data.cov()

    def EigenVector(self,cov_matrix):
        # eigenvectors and eigenvalues for the from the covariance matrix
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_matrix)
        return eig_vec_cov

    def EigenValue(self,cov_matrix):
        # eigenvectors and eigenvalues for the from the covariance matrix
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_matrix)
        return eig_val_cov

    def Sort_EigenValue(self,eigen_value,count):
        # Make a list of (eigenvalue, eigenvector) tuples
        # eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

        # # Sort the (eigenvalue, eigenvector) tuples from high to low
        # eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # # Visually confirm that the list is correctly sorted by decreasing eigenvalues
        # for i in eig_pairs:
        #     print(i[0])
        return np.argsort(np.abs(eigen_value))[::-1][0:count]

    def W_Matrix(self,data,eig_vec_cov,sorted_eigen):
        matrix_w = eig_vec_cov[:, sorted_eigen[0]].reshape(data.shape[1], -1)
        print("------------------------------------")
        print(eig_vec_cov)
        for i in range(len(sorted_eigen)-1):
            matrix_w = np.hstack((matrix_w, eig_vec_cov[:,sorted_eigen[i+1]].reshape(data.shape[1], -1)))
        return matrix_w

    

    

