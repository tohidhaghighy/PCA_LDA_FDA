import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

class LDA:
    def __init__(self,n_components=2):
        self.n_components=n_components
        self.LDA_Matrix_list=[]

    def LoadFile_vowel(self):
        return pd.read_csv("voweltrain.csv", header=None)

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
    # میانگین داده ها را پیدا میکند
    def Get_Mean_Vector(self,data):
        return data.mean().values

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
        self.LDA_Matrix_list=final_matrix

    def Draw_Chart(self,x):
        fig, ax = plt.subplots(1, 1)
        sns.scatterplot(x=x[:,0], y=x[:,1], ax=ax)
        # show chart with new data
        plt.show()


    def Play_LDA(self,x,y):
        #مقادیر تکست را به عدد تبدیل میکنیم
        y=self.Label_Encoder(y)
        #داده ها را به مرکز میبریم
        x=self.fit_transform(x)
        #Find Mean Vector
        mean_vector=self.Mean_Vector(x,y)
        
        #Find Scatter Matrix
        scatter_matrix=self.Scatter_Matrix(x,y,mean_vector)
        
        #Find Between Scatter Matrix
        between_scatter=self.BetweenClass_Scatter(x,y,mean_vector)
        
        #Find Eigen Vector
        eigen_vec=self.EigenVector(scatter_matrix,between_scatter)
        
        #Find Eigen Value
        eigen_value=self.EigenValue(scatter_matrix,between_scatter)
        
        #Sort Eigen Value for find Main Column
        sorted_component=self.Component_Sort(eigen_value,2)
        
        #Make W Matrix
        w_Matrix=self.W_Matrix(eigen_vec,sorted_component,x)
        
        #Return Final List in Lda Matrix
        self.LDA_Matrix(x,w_Matrix)
        # #Draw Chart
        self.Draw_Chart(self.LDA_Matrix_list)
        

    
    