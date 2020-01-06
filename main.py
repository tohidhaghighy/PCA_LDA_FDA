from PCAAlgorithm import PCA
from LDAAlgorithm import LDA
from FDAAlgorithm import FDA
import pandas as pd
import numpy as np


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


#------------------run pca --------------------------
pca=PCA()
df=pca.LoadFile(url)
pca.PlayPCA(df)

# pca.pca_matrix

pcalist=pca.pca_matrix

#----------------------run lda ------------------------
lda=LDA()
df=lda.LoadFile(url)
x = df.loc[:, 0:3]
y = df.loc[:, [4]]
lda.Play_LDA(x,y)

ldalist=lda.LDA_Matrix_list

#lda.LDA_Matrix_list

# xpca=pca.loc[:,0:1]
# xlda=lda.loc[:,0:1]
xpca=pd.DataFrame(pcalist)
xlda=pd.DataFrame(ldalist)
print(xpca)
print(xlda)
fda=FDA()
fda.PlayFDA(xpca,y)
