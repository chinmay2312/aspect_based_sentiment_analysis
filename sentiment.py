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
# from nltk.tokenize.moses import MosesDetokenizer
from nltk.corpus import stopwords
#import enchant
import re
import scipy

re_tokenize = RegexpTokenizer("[\w']+")
#checker = enchant.Dict("en_US")
stop_words = set(stopwords.words('english'))
in_data_file = "data2_train.csv"


def load_data(in_data_file):
    data = pd.read_csv(in_data_file, skipinitialspace=True)
    data.text = data.text.str.replace("\[comma\]", ",")
    # data = in_data.copy(deep=True)
    data.text = data["text"].apply(remove_tags)
    #data = stemming(data)
    print("12323456789")
    d_x = data.text
    d_y = data["class"]
    x_vect = myFunc(data[["text","aspect_term","term_location"]])
    #print(x_vect.sample(n=5))
    #print(type(x_vect))
    x_vect = scipy.sparse.csr_matrix(x_vect)
    x_vector = tfidf_vectorize(d_x)
    #print(type(x_vector))
    print("1111111")
    k_neighbor(x_vector, d_y)
    #svm(x_vect, d_y)
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
def myFunc(myData):
    
    #split term_location into start and end
    #print(data.sample(n=5))
    aspect_st = []
    aspect_end = []
    textLen = []
    #print(data["term_location"])
    for id,row in myData.iterrows():
        aspect_locs = re.split('--',row['term_location'])
        #print(aspect_locs)
        text_len = len(row['text'])
        textLen.append(text_len)
        aspect_st.append(int(aspect_locs[0])/text_len)
        aspect_end.append(int(aspect_locs[1])/text_len)
    myData["aspect_start"] = aspect_st
    myData["aspect_end"] = aspect_end
    myData["text_len"] = textLen
    #print(data.sample(n=5))
    return myData[["aspect_start","aspect_end","text_len"]]
    

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
        print("#######################")
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

    print("*********************************")
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
        print("#######################")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    print("*********************************")
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
        print("#######################")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    print("*********************************")
    accuracy = np.mean(accuracy_list)
    # precision = np.mean(precision_list)
    # recall = np.mean(recall_list)
    print("Accuracy for svc is : " + str(accuracy))
    # print("Precision for svc is : " + str(precision))
    # print("Recall for svc is : " + str(recall))


def tfidf_vectorize(d_x):
    tfidf = TfidfVectorizer()
    x_vec = tfidf.fit_transform(d_x)
    #print("Features:",tfidf.get_feature_names())
    # x_traincv_array = x_vec.toarray()
    return x_vec


def tokenize(cp_in_data):
    cp_in_data["text"] = cp_in_data["text"].apply(lambda row: word_tokenize(row))
    return cp_in_data


def remove_tags(row):
    row = str(row)
    cleanr = re.compile('(</?[a-zA-Z]+>|https?:\/\/[^\s]*|(^|\s)RT(\s|$)|@[^\s]+|\d+)')
    cleantext = re.sub(cleanr, ' ', row)
    cleantext = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)', ' ', cleantext)
    cleantext = re.sub('[^\sa-zA-Z]+', '', cleantext)
    cleantext = re.sub('\s+', ' ', cleantext)
    cleantext = cleantext[0:].strip()
    return cleantext


def stemming(temp_df):
    # PORTER STEMMER
    print("Porter Stemmer")
    porter_stemmer = nltk.stem.PorterStemmer()
    for i, text in enumerate(temp_df.text):
        words = word_tokenize(text)
        for j, word in enumerate(words):
            words[j] = porter_stemmer.stem(word)
        temp_df.loc[i, "text"] = ' '.join(words)
    return temp_df


load_data(in_data_file)
