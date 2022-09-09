import nltk,re,pandas as pd,random,numpy as np

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import  word_tokenize
from nltk import ngrams
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from Read_Data import TextDataPartitioning
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Doc2Vec
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

class TrasForm:
    
    
    @staticmethod
    def LabelEncoder(Labels):
        le = preprocessing.LabelEncoder()
        Result = le.fit_transform(Labels)
        return Result
    
    
    def BoW_Encoder(Paragaphs):
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(Paragaphs)
        return X , vectorizer
    
    def TFIDF_Encoder(Paragaphs):
        
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(Paragaphs)
        return X , vectorizer
    
    def TFIDF_NGram_Encoder(Paragaphs , start_Ngram , End_Ngram):
        
        vectorizer = TfidfVectorizer(ngram_range=(start_Ngram,End_Ngram))
        X = vectorizer.fit_transform(Paragaphs)
        return X , vectorizer

    @staticmethod
    def W2V_Encoder(Paragaphs):
        from gensim.models import KeyedVectors
        import gensim.downloader as gensim_api
        nlp = gensim_api.load("glove-wiki-gigaword-300")
        
        w2v_model = gensim.models.Word2Vec(Paragaphs, vector_size=300, min_count = 0, window = 5)
        All_Vectores= []
        for par in Paragaphs:
            vectors = [w2v_model.wv[x] for x in par]
            All_Vectores.append(vectors)
        # ALLDoc= [] 
        # for  i in range(0,len(Paragaphs)):
        #     DocVec= [] 
        #     for word in Paragaphs.iloc[i]:
        #         DocVec.append(w2v_model.wv[word])
                
        #     ALLDoc.append(np.mean( np.array(DocVec)))
        
        return All_Vectores , w2v_model
    
    @staticmethod
    def SVD(XTrain,n):
        SVD = TruncatedSVD(n_components = n,random_state=1)
        XTrain = SVD.fit_transform(XTrain)
        return XTrain ,SVD
    
    @staticmethod
    def Test_Best_PCA_com(X,Test_Range,Test_Inrease,plt_title):
        
        List_Variance = [] 
        Tests=[ x*Test_Inrease for x in Test_Range ]
        for i in Tests:
            print(i)
            SVD  =  TruncatedSVD(n_components=i, random_state=42)
            X_trainBOW_SVD= SVD.fit_transform(X)
            var_explained = SVD.explained_variance_ratio_.sum()
            List_Variance.append(var_explained)
        plt.plot( Tests ,List_Variance, linewidth=2)
        plt.title(f'Scree Plot ({plt_title}) Transform')
        plt.xlabel('Principal Component')
        plt.ylabel('Proportion of Variance Explained')
        plt.show()
            
   
    @staticmethod
    def TransForm(Type , Paragaphs ,  start_Ngram , End_Ngram):
        
        if(Type=="BOW"):
            return TrasForm.BoW_Encoder(Paragaphs)
        
        elif(Type=="TF_IDF"):
            return TrasForm.TFIDF_Encoder(Paragaphs)
        
        elif(Type=="TFIDF_NGram"):
            return TrasForm.TFIDF_NGram_Encoder(Paragaphs ,  start_Ngram , End_Ngram)
        
        elif(Type=="W2V_Encoder"):
            return TrasForm.W2V_Encoder(Paragaphs)
        
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        