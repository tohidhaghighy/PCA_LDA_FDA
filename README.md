# PCA_LDA_FDA
Machine learning dimension reduction algorithms

The main purposes of a principal component analysis are the analysis of data to identify patterns and finding patterns to reduce the dimensions of the dataset with minimal loss of information.

Here, our desired outcome of the principal component analysis is to project a feature space (our dataset consisting of n d-dimensional samples) onto a smaller subspace that represents our data “well”. A possible application would be a pattern classification task, where we want to reduce the computational costs and the error of parameter estimation by reducing the number of dimensions of our feature space by extracting a subspace that describes our data “best”.

## PCA

Summarizing the PCA approach
Listed below are the 6 general steps for performing a principal component analysis, which we will investigate in the following sections.

 * Take the whole dataset consisting of d-dimensional samples ignoring the class labels
 
 * Compute the d-dimensional mean vector (i.e., the means for every dimension of the whole dataset)
 
 * Compute the scatter matrix (alternatively, the covariance matrix) of the whole data set
 
 * Compute eigenvectors (ee1,ee2,...,eed) and corresponding eigenvalues (λλ1,λλ2,...,λλd)
   Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k dimensional matrix    WW(where every column represents an eigenvector)
   
 * Use this d×k eigenvector matrix to transform the samples onto the new subspace.
 
