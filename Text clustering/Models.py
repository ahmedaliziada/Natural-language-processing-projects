from Transform_Data import TrasForm
from Read_Data import TextDataPartitioning
import nltk,re,pandas as pd,random,numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


#Read Books
Books =TextDataPartitioning.GetBooks() 

#Encode Your Labels
Y =TrasForm.LabelEncoder(Books["Authors"]) 

# TRansform Your Featrues
X_Bow , Transformer =  TrasForm.TransForm("BOW",Books["paragraph"],0,0)

# TF_IDF
X_Tf_IDF , Transformer =  TrasForm.TransForm("TF_IDF",Books["paragraph"],0,0)

# TF_IDF 2_Gram
X_2Gram , Transformer =  TrasForm.TransForm("TFIDF_NGram",Books["paragraph"],2,2)


#Perform SVD On the Data
#BOW
TrasForm.Test_Best_PCA_com(X_Bow,[x for x in range(1,11)],100,"TF_IDf")
X_SVD_Bow , Bow_SVD = TrasForm.SVD(X_Bow,600)


#Perform SVD On the Data
#Tf_idf
TrasForm.Test_Best_PCA_com(X_Tf_IDF,[x for x in range(1,11)],100,"TF_IDf")
X_SVD_Tf_idf  , TF_idf_SVD = TrasForm.SVD(X_Tf_IDF,600)


#Perform SVD On the Data
#2Gram
TrasForm.Test_Best_PCA_com(X_2Gram,[x for x in range(1,11)],100,"TF_IDf")
X_SVD_2Gram , Gram2_SVD = TrasForm.SVD(X_2Gram,700)



def DBScan(X):
    from sklearn.cluster import DBSCAN
    
   
    dbscan = DBSCAN(eps=0.2, min_samples=5 ,metric='euclidean')
    dbscan.fit_predict(X)
    return dbscan


def Visulaize_Data(X,Labels ,Title):
    
    from sklearn.manifold import TSNE
    from sklearn.decomposition import TruncatedSVD
    colors=['red','blue','green','brown','grey','orange','yellow','beige','red','maroon']
    # tfs_reduced = TruncatedSVD(n_components=650, random_state=42).fit_transform(X)
    tfs_embedded = TSNE(n_components=2, perplexity=50, verbose=2 , random_state=1 , n_iter=2000).fit_transform(X)
    
    fig = plt.figure(figsize = (5, 5))
    ax = plt.axes()

    sns.scatterplot(x=tfs_embedded[:, 0], y=tfs_embedded[:, 1],hue=Labels )
    plt.title(Title)

    plt.show()

def Kmeans(X):
    from sklearn.cluster import KMeans
    k = 6
    km = KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=10,verbose=1 ,random_state=1 , algorithm='elkan' )
    km.fit_predict(preprocessing.normalize(X))
    return km



Visulaize_Data(X_Bow,Y,"Bow")
Visulaize_Data(X_Tf_IDF,Y,"TF_idf")
Visulaize_Data(X_2Gram,Y,"2Gram")



Visulaize_Data(X_SVD_Bow,Y,"SVD_Bow")
Visulaize_Data(X_SVD_Tf_idf,Y,"SVD_TF_idf")
Visulaize_Data(X_SVD_2Gram,Y,"SVD_2Gram")
















