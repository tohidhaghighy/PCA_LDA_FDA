from PCAAlgorithm import PCA
from LDAAlgorithm import LDA
from FDAAlgorithm import FDA


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

pca=PCA()
df=pca.LoadFile(url)
print(df.head(5))

x=pca.OnlyData(df)
print(pca.fit_transform(x))

mean=pca.Get_Mean_Vector(df)
print(pca.Scatter_Matrix(x,mean))

covarianse=pca.CoVarianse_Matrix(x)
eigenvec=pca.EigenValue(covarianse)
eigenval=pca.EigenVector(covarianse)
sortedeigenval=pca.Sort_EigenValue(eigenval,2)
print(sortedeigenval)

print(pca.W_Matrix(x,eigenvec,sortedeigenval))