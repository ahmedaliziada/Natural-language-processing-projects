from tracemalloc import Statistic
import nltk,re,pandas as pd,random,numpy as np
from nltk.corpus import stopwords 
import matplotlib.pyplot as plt

class TextDataPartitioning:  # Create Class to generalize the program

    Partitioning_DataFrame2 = pd.DataFrame()
    
    def ReadBooks(Book_Name):  # This function will take Book_Name as parameter for example: austen-emma.txt
        return nltk.corpus.gutenberg.raw(Book_Name)  # Convert NLTK books to string datatype
    
    def GetTitle(Book_Name):  # This function will take Book_Name as parameter for example: austen-emma.txt
        Book = TextDataPartitioning.ReadBooks(Book_Name)  # Call ReadBooks function
        TextDataPartitioning.Book_Title = re.findall('^\[(.*)\]', Book)  # Use re.findall to extract and assign the title of the book to list datatype variable
        TextDataPartitioning.Book_Title = TextDataPartitioning.Book_Title[0] # Convert list datatype to string datatype
        return TextDataPartitioning.Book_Title # Return book title as string datatype
    
    def BookPreProcessing(Book_Name):   # This function will take Book_Name as parameter for example: austen-emma.txt
        Book = TextDataPartitioning.ReadBooks(Book_Name)  # Call ReadBooks function
        TextDataPartitioning.Book_Title = '[' + TextDataPartitioning.Book_Title + ']'  # Add [] to the book title to be able to remove it with the square brackets from the book
        Book = Book.replace(TextDataPartitioning.Book_Title,'')  # Remove book title from the book
        Book = re.sub('(CHAPTER(.*))|(Chapter(.*))', '', Book)  # Remove chapter title from the book
        Book = re.sub('(VOLUME(.*))|(Volume(.*))', '', Book)  # Remove Volume title from the book  
        Book = re.sub('^$\n', '', Book, flags = re.MULTILINE)  # Remove empty lines
        Book = re.sub('\. *(\W)','.\n\n', Book)  # Create a new line after each fullstop
        Book = re.sub('[^\w\s]','', Book)  # Remove punctuation marks from the book
        Book = re.sub(r'\b\w{1,2}\b', '',Book)
        Book = re.sub(r'(http|https|ftp)://[a-zA-Z0-9\\./]+', '', Book)
        Book = re.sub("[^a-zA-Z]", " ",Book)
        Book = Book.lower()  # Convert the book to small text
        Book = Book.split()  # Convert the book to list datatype
        StopWords = set(stopwords.words('english'))
        Book = [word for word in Book if not word in StopWords]  # Remove stopwords from this list
        return Book  # Return the book list datatype

    def BookPartitioning(Text):
        PhargraphList=[]
        TextLen= len(Text)
        
        
        NumberOfParagraph = TextLen // 150
        
        for i in range(1,NumberOfParagraph):
            startIndex = i*150 - 150
            endIndex = (i-1)*150+150
            
            if(endIndex >= len(Text)):
                endIndex = len(Text) - endIndex
            
            PhargraphList.append(Text[startIndex : endIndex])
        return PhargraphList
 

    def CreateDataFrame1(Partitions_list, Book_Title):
        DFparagraphs = pd.DataFrame(columns =["paragraph","Authors"])
     
        random.seed(41)
        #choose 200 random paragaph from all paragaphs
        Partitions_list=random.choices(Partitions_list, k=200)
        
        data = {'paragraph':Partitions_list,'Authors':Book_Title}
        
        DFparagraphs = DFparagraphs.append( pd.DataFrame(data))

        return DFparagraphs


    def ConvertToString(DataFrame):
        for i in range(len(DataFrame)):
            DataFrame.iloc[i,0] = ' '.join([str(element) for element in DataFrame.iloc[i,0]])
        return DataFrame
    
    
    def Plot_Commaon_words(Paragaphs, Book_title,N_gram):
        parag=" ".join(Paragaphs)
        lst_tokens = nltk.tokenize.word_tokenize(parag)
        List = []
        for i in range(1,N_gram+1):
            dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, i))
            dtf_uni = pd.DataFrame(dic_words_freq.most_common(),  columns=["Word","Freq"])
            dtf_uni.sort_values(by="Freq" ,inplace=True ,ascending=False )
            dtf_uni.iloc[0:15 , :  ].set_index("Word").plot(kind="barh", title= f" {Book_title} {i}grams",legend=False).grid(axis='x')
            List.append(dtf_uni)
            plt.show()
        return List

    @staticmethod
    def GetBooks():
        ListOfBooks = ['austen-emma.txt','edgeworth-parents.txt','melville-moby_dick.txt','chesterton-thursday.txt','carroll-alice.txt']
        Authors = ['Jane Austen','Maria Edgeworth','Herman Melville','G. K. Chesterton','Lewis Carroll']
        DFparagraphs = pd.DataFrame(columns =["paragraph","Authors"])

        Book_Title = [None] * 5
        Book = [None] * 5
        Partitions = [None] * 5
        for i in range(5):
            
            Book_Title[i] = TextDataPartitioning.GetTitle(ListOfBooks[i])
            
            Book[i] = TextDataPartitioning.BookPreProcessing(ListOfBooks[i])
            
            Partitions[i] = TextDataPartitioning.BookPartitioning(Book[i])
            DFparagraphs=  DFparagraphs.append( TextDataPartitioning.CreateDataFrame1(Partitions[i], Authors[i]))
            

        del Partitions


        return TextDataPartitioning.ConvertToString(DFparagraphs)

    @staticmethod
    def PotCommonWords(DFparagraphs):
        Authers = DFparagraphs["Authors"].unique()

        for Auther in Authers:
            Paragaphs = DFparagraphs[DFparagraphs["Authors"] ==Auther ]["paragraph"]
            TextDataPartitioning.Plot_Commaon_words(Paragaphs, Auther,2)


# Books = TextDataPartitioning.GetBooks()
# TextDataPartitioning.PotCommonWords(Books)