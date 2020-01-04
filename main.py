from PCAAlgorithm import PCA
from LDAAlgorithm import LDA
from FDAAlgorithm import FDA


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


#------------------run pca --------------------------
pca=PCA()
df=pca.LoadFile(url)
pca.PlayPCA(df)

print(pca.pca_matrix)


#----------------------run lda ------------------------
lda=LDA()
df=lda.LoadFile(url)

