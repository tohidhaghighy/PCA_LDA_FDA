import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class PCA:
    def __init__(self,n_components=2):
        self.n_components=n_components
        self.pca_matrix=[]

    def LoadFile(self,name):
        return pd.read_csv(name, header=None)

    # فقط داده ها را بر میگرداند
    def OnlyData(self,data):
        x = data.loc[:, 0:3]
        y = data.loc[:, [4]]
        return x
    
    # داده ها را به مرکز منتقل میکند
    def fit_transform(self,data):
        mean_vector = self.Get_Mean_Vector(data)
        data -= mean_vector
        return data

    # میانگین داده ها را پیدا میکند
    def Get_Mean_Vector(self,data):
        return data.mean().values

    # Scatter matrix
    def Scatter_Matrix(self,data,mean_vector):
        scatter_matrix = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[0]):
            scatter_matrix += (data.iloc[i, :].values.reshape(data.shape[1], 1) - mean_vector).dot((data.iloc[i, :].values.reshape(data.shape[1], 1) - mean_vector).T)
        # eigenvectors and eigenvalues for the from the scatter matrix
        # eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
        return scatter_matrix

    # Co varianse matrix
    def CoVarianse_Matrix(self,data):
        return data.cov()

    # eigen vector
    def EigenVector(self,cov_matrix):
        # eigenvectors and eigenvalues for the from the covariance matrix
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_matrix)
        return eig_vec_cov

    # eigen value
    def EigenValue(self,cov_matrix):
        # eigenvectors and eigenvalues for the from the covariance matrix
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_matrix)
        return eig_val_cov
    
    #مرتب کردن داده ها
    def Sort_EigenValue(self,eigen_value,count):
        # Make a list of (eigenvalue, eigenvector) tuples
        #eig_pairs = [(np.abs(eigen_value[i]), eigen_vec[:,i]) for i in range(len(eigen_value))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        #eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # # Visually confirm that the list is correctly sorted by decreasing eigenvalues
        # for i in eig_pairs:
        #     print(i[0])

        return np.argsort(np.abs(eigen_value))[::-1][0:2]

    # ماتریسی که با ان کاهش بعد را محاسبه میکنیم
    def W_Matrix(self,data,eig_vec_cov,sorted_eigen):
        matrix_w = eig_vec_cov[:, sorted_eigen[0]].reshape(data.shape[1], -1)
        for i in range(len(sorted_eigen)-1):
            matrix_w = np.hstack((matrix_w, eig_vec_cov[:,sorted_eigen[i+1]].reshape(data.shape[1], -1)))
        return matrix_w
    
    #ماتریس کاهش بعد داده شده
    def reduction_array(self,x,w_matrix):
        return x.values @ w_matrix

    # draw a chart 
    def Draw_Chart(self,x):
        fig, ax = plt.subplots(1, 1)
        sns.scatterplot(x=x[:,0], y=x[:,1], ax=ax)
        # show chart with new data
        plt.show()

    def PlayPCA(self,datafram):
        #از داخل دیتا فقط ویژگی ها را بر میدارد
        x=self.OnlyData(datafram)
        # داده ها را به مرکز میبرد
        x=self.fit_transform(x)
        # میانگین داده ها را پیدا میکند
        mean=self.Get_Mean_Vector(x)
        # 2 روش برای اجرای این الگوریتم داریم
        # اولی روش اسکتر ماتریکس است
        # دومی روش کو واریانس است 
        print(self.Scatter_Matrix(x,mean))
        #-----------------روش کو واریانس را پیلده کردیم رو داده ها-----------------------
        covarianse=self.CoVarianse_Matrix(x)
        # find eigen value
        eigenval=self.EigenValue(covarianse)
        # find eigen vector
        eigenvec=self.EigenVector(covarianse)
        # sort eigen val
        sortedeigenval=self.Sort_EigenValue(eigenval,2)
        # make new matrix with eigen vector
        w_matrix=self.W_Matrix(x,eigenvec,sortedeigenval)
        # demention reduction data
        final_matrix=self.reduction_array(x,w_matrix)
        self.pca_matrix=final_matrix
        # draw chart with data
        self.Draw_Chart(final_matrix)


    

    

    

