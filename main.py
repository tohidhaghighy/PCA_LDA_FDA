from PCAAlgorithm import PCA
from LDAAlgorithm import LDA
from FDAAlgorithm import FDA


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


#------------------run pca --------------------------
pca=PCA()
df=pca.LoadFile(url)
pca.PlayPCA(df)

# pca.pca_matrix


#----------------------run lda ------------------------
lda=LDA()
df=lda.LoadFile(url)
x = df.loc[:, 0:3]
y = df.loc[:, [4]]
lda.Play_LDA(x,y)

#lda.LDA_Matrix_list
