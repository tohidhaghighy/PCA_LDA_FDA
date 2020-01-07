from PCAAlgorithm import PCA
from LDAAlgorithm import LDA
from FDAAlgorithm import LDAClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def cross_validate(x, y, k=5):
    # Stacking x and y horiontally
    m = np.hstack((x, y.reshape(x.shape[0],-1)))
    # Shuffling data to randomize their order
    np.random.shuffle(m)
    # Splitting x and y
    x = m[:, :-1]
    y = m[:, -1].reshape(x.shape[0],-1)
    dl = len(y)
    fl = int(dl/k)
    folds_indices = [(i*fl, (i+1)*fl) for i in range(0, k)]
    scores = []
    for i in range(0, k):
        i, j = folds_indices[i]
        test_x = x[i:j, :]
        test_y = y[i:j, :]
        train_x = np.vstack((x[0:i, :], x[j:, :]))
        train_y = np.vstack((y[0:i, :], y[j:, :]))
        train_x = pd.DataFrame(train_x)
        test_x = pd.DataFrame(test_x)
        lda_classifier = LDAClassifier()
        lda_classifier.fit(train_x, train_y)
        s = lda_classifier.score(test_x, test_y)
        scores.append(s)
    return sum(scores) / len(scores)


if __name__=="__main__":
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
    enc = LabelEncoder()
    label_encoder = enc.fit(y.values.ravel())
    y = label_encoder.transform(y) + 1
    
    lda_classifier = LDAClassifier()
    scores = cross_val_score(lda_classifier, ldalist, y, cv=5)
    print(scores.mean())
