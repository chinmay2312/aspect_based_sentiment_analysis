import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize.moses import MosesDetokenizer
from nltk.corpus import stopwords
# import enchant
import re
import scipy

re_tokenize = RegexpTokenizer("[\w']+")
wnl = WordNetLemmatizer()
# checker = enchant.Dict("en_US")
stop_words = set(stopwords.words('english'))
in_data_file = "data2_train.csv"


def load_data(in_data_file):
    data = pd.read_csv(in_data_file, skipinitialspace=True)
    data.text = data.text.str.replace("\[comma\]", ",")
    data.aspect_term = data.aspect_term.str.replace("\[comma\]", ",")
    # data = in_data.copy(deep=True)
    #data.text = data["text"].apply(remove_tags)
    data = stemming_and_lemmatization(data)
    print("Data loaded and preprocessed")
    print(data)
    import time
    # d_x = data.text
    d_y = data["class"]
    x_vect = myFunc(data)#[["text","aspect_term","term_location"]])
    x_vect = calc_adj_feature(x_vect)
    x_vect = scipy.sparse.csr_matrix(x_vect)
    # x_vector = tfidf_vectorize(d_x)
    # print(x_vect.shape)
    #k_neighbor(x_vect, d_y)
    svm(x_vect, d_y)
    #decision_tree(x_vect, d_y)
    # cp_in_data = remove_stopwords(cp_in_data)


'''
def remove_stopwords(cp_in_data):
    # tokenize, remove stopwords and detokenize
    cp_in_data = tokenize(cp_in_data)
    for i in range(len(cp_in_data["text"])):
        for word in cp_in_data["text"][i]:
            if word in stop_words:
                cp_in_data["text"][i].remove(word)

    detokenizer = MosesDetokenizer()
    cp_in_data["text"] = cp_in_data["text"].apply(lambda row: detokenizer.detokenize(row, return_str=True))
    return cp_in_data
'''

def calc_adj_feature(myData):
    text_file = open("pos_words.txt", "r")
    posWords = text_file.read().split('\n')
    text_file = open("neg_words.txt", "r")
    negWords = text_file.read().split('\n')
    
    conjunctList = ['but', 'nor', 'yet', 'although', 'before', 'if', 'though', 'till', 'unless', 'until', 'what', 'whether', 'while']
    
    adjClass = []
    adj_dist = []
    nearest_adj = []
    posCount = []
    negCount = []
    subSentCounts = []
    
    for id,row in myData.iterrows():
        minDist = len(row['text'])
        
        
        myText = row['text']
        myText2 = myText
        
        for word in word_tokenize(myText2):
            if word in conjunctList:
                myText2 = myText2.replace(word,"~")
                
        subSentCount = len(myText2.split('~'))
                
        for subText in myText2.split('~'):
            if row['aspect_term'] in subText:
                #print("Full text: ",myText)
                #print("subText: ",subText)
                #print("aspect:",row['aspect_term'])
                myText = subText
                break
        
        #print("text: ",row['text'])
        #print("aspect: ",row['aspect_term'])
        try:
            term_start = myText.index(row['aspect_term'])#int(re.split('--',row['term_location'])[0])
            term_end = term_start + len(row['aspect_term'])#int(re.split('--',row['term_location'])[1])
        except:
            term_start = int(re.split('--',row['term_location'])[0])
            term_end = int(re.split('--',row['term_location'])[1])
        #print("start: ",term_start, " end: ",term_end)
        neg_count =0
        pos_count =0
        for str in word_tokenize(myText):
            
            #print("str: ",str,"\t index:",row['text'].index(str))
            if str in posWords:
                pos_count = pos_count +1
            elif str in negWords:
                neg_count = neg_count +1
            
            try:
                if myText.index(str) >= term_start and myText.index(str) < term_end:
                    continue
            except:
                print("Error line 94")
                print("text: ",myText)
                print("str: ",str)
                #print("index: ",row['text'].index(str))
            
            try:
                dist = abs(myText.index(str) - myText.index(row['aspect_term']))
            except ValueError:
                print("Aspect Term missing in Text")
                print("id: ",id)
                print("text: ",myText)
                print("aspect: ",row['aspect_term'])
              #  print("start: ",term_start, " end: ",term_end)
            #finally:
                adj_class =0
                minDist = len(myText)
                near_adj = str
                break
            
            if dist < minDist:
                #minDist = dist
                near_adj = str
                if str in posWords:
                    adj_class= +1 #Positive
                    minDist = dist
                    #near_adj = str
                elif str in negWords:
                    adj_class= -1 #Negative
                    minDist = dist
                    #near_adj = str
                else:
                    adj_class= 0  #Neutral
        adjClass.append(adj_class)
        nearest_adj.append(near_adj)
        adj_dist.append(minDist)
        posCount.append(pos_count)
        negCount.append(neg_count)
        subSentCounts.append(subSentCount)
        
    #myData["nearest_adj"] = nearest_adj
    myData["adj_class"] = adjClass
    myData["adj_dist"] = adj_dist
    myData["posCount"] = posCount
    myData["negCount"] = negCount
    myData["subSentCounts"] = subSentCounts
    
    return myData[["aspect_start","aspect_end","text_len","adj_class","adj_dist","posCount","negCount", "idf_score","idf_aspect"]]


def myFunc(myData):
    
    tfidf = TfidfVectorizer(use_idf=True)
    #x_vec = tfidf.fit_transform(myData.text)
    #print(x_vec.shape)
    #print("Features:",tfidf.get_feature_names())
    #idf_sum = [sum(x_vec[i]) for i in range(3602)]
    #idf_sum = x_vec.sum(axis=1)
    #print(idf_sum.shape)
    
    #split term_location into start and end
    #print(data.sample(n=5))
    aspect_st = []
    aspect_end = []
    textLen = []
    senseText = []
    #print(data["term_location"])
    for id,row in myData.iterrows():
        aspect_locs = re.split('--',row['term_location'])
        #print(aspect_locs)
        text_len = len(row['text'])
        textLen.append(text_len)
        aspect_st.append(int(aspect_locs[0])/text_len)
        aspect_end.append(int(aspect_locs[1])/text_len)
        
        word_tokens = word_tokenize(row['text'])
        filtered_sentence = ''
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence = filtered_sentence + ' ' + w
        senseText.append(filtered_sentence.strip())
    
    myData["aspect_start"] = aspect_st
    myData["aspect_end"] = aspect_end
    myData["text_len"] = textLen
    
    myData["sense_text"] = senseText
    #print(senseText[:5])
    
    x_vec = tfidf.fit_transform(myData.sense_text)
    #print(x_vec[0])
    #print(tfidf.inverse_transform(x_vec)[0])
    #print(np.where(tfidf.inverse_transform(x_vec)[0] == 'staff')[0])
    
    #for id,row in myData.iterrows():    
    
    #print("Features:",tfidf.get_stop_words())
    #print(tfidf['abbys'])
    idf_sum = x_vec.sum(axis=1)
    myData["idf_score"] = idf_sum
    
    #print(data.sample(n=5))
    return myData[["text","aspect_term","term_location","aspect_start","aspect_end","text_len", "idf_score"]]
        

def k_neighbor(x, y):
    knn = KNeighborsClassifier(n_neighbors=100)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []
    precision_list = []
    recall_list = []
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy_list.append(knn.score(x_test, y_test))
        cm = confusion_matrix(y_test, y_pred)
        tp = cm[0][0]
        tn = cm[2][2]
        fp = cm[0][2]
        fn = cm[2][0]
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        precision_list.append(prec)
        recall_list.append(rec)
        print(cm)
        # precision = precision_score(y_test, y_pred, labels=[0,1])
        # print(precision)
        # precision_score()

    accuracy = np.mean(accuracy_list)
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)

    print("Accuracy for kNN is : " + str(accuracy))
    print("Precision for kNN is : " + str(precision))
    print("Recall for kNN is : " + str(recall))


def decision_tree(x, y):
    dtree = DecisionTreeClassifier(max_depth=3)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtree.fit(x_train, y_train)
        y_pred = dtree.predict(x_test)
        accuracy_list.append(dtree.score(x_test, y_test))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    accuracy = np.mean(accuracy_list)
    # precision = np.mean(precision_list)
    # recall = np.mean(recall_list)
    print("Accuracy for Decision Tree is : " + str(accuracy))
    # print("Precision for Decision Tree is : " + str(precision))
    # print("Recall for Decision Tree is : " + str(recall))


def svm(x, y):
    svc = LinearSVC()
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        accuracy_list.append(svc.score(x_test, y_test))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    accuracy = np.mean(accuracy_list)
    # precision = np.mean(precision_list)
    # recall = np.mean(recall_list)
    print("Accuracy for svc is : " + str(accuracy))
    # print("Precision for svc is : " + str(precision))
    # print("Recall for svc is : " + str(recall))


def tfidf_vectorize(d_x):
    tfidf = TfidfVectorizer()
    x_vec = tfidf.fit_transform(d_x)
    # print("Features:",tfidf.get_feature_names())
    # idf_sum = [sum(x_vec[i]) for i in range(3602)]
    # x_traincv_array = x_vec.toarray()
    return x_vec


def tokenize(cp_in_data):
    cp_in_data["text"] = cp_in_data["text"].apply(lambda row: word_tokenize(row))
    return cp_in_data


def do_re_tokenize(row):
    #for x in data['text']:
    x = re_tokenize.tokenize(row)
    return x

def remove_tags(row):
    row = str(row)
    cleanr = re.compile('(</?[a-zA-Z]+>|https?:\/\/[^\s]*|(^|\s)RT(\s|$)|@[^\s]+|\d+)')
    cleantext = re.sub(cleanr, ' ', row)
    cleantext = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)', ' ', cleantext)
    cleantext = re.sub('[^\sa-zA-Z]+', '', cleantext)
    cleantext = re.sub('\s+', ' ', cleantext)
    cleantext = cleantext[0:].strip()
    return cleantext


def stemming_and_lemmatization(data_stem):
    # PORTER STEMMER
    print("Porter Stemmer")
    porter_stemmer = PorterStemmer()

    data_stem["text"] = data_stem["text"].apply(do_re_tokenize)
    #data_stem['text'] = data_stem['text'].apply(lambda x: [porter_stemmer.stem(y) for y in x])
    data_stem["text"] = data_stem["text"].apply(lambda x: [wnl.lemmatize(y) for y in x])
    data_stem["text"] = data_stem["text"].apply(lambda x: " ".join(x))

    data_stem["aspect_term"] = data_stem["aspect_term"].apply(do_re_tokenize)
    #data_stem['aspect_term'] = data_stem['aspect_term'].apply(lambda x: [porter_stemmer.stem(y) for y in x])
    data_stem["aspect_term"] = data_stem["aspect_term"].apply(lambda x: [wnl.lemmatize(y) for y in x])
    data_stem["aspect_term"] = data_stem["aspect_term"].apply(lambda x: " ".join(x))

    return data_stem


load_data(in_data_file)
