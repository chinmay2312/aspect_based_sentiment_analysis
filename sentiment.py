import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import scipy

re_tokenize = RegexpTokenizer("[\w']+")
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
in_data_file = "data2_train.csv"
test_data_file = "Data-1_test.csv"


def load_data(in_data_file):
    data = pd.read_csv(in_data_file, skipinitialspace=True)
    dataTest = pd.read_csv(test_data_file, skipinitialspace=True)
	
    '''PreProcessing'''
    data = preprocess(data)
    dataTest = preprocess(dataTest)
    print("Data loaded and preprocessed")
	
    #data["class"] = data["class"].replace(-1, 2)
    d_y = data["class"]

    '''Feature Extraction'''
    x_vect = extract_features(data)
    #print(type(X_vect))
    #print(X_vect.shape)
    #chi2_selector = SelectKBest(chi2, k=4000)
    #x_vect = chi2_selector.fit_transform(X_vect, data['class'])
	
    relevantCols = []
    #for col in chi2_selector.get_support(indices=True):
    #    relevantCols.append(X_vect.columns[col])

    x_vect_test = extract_features(dataTest)#.toarray())
    #x_vect_test = x_vect_test[relevantCols]
    # x_vect = myFunc(data)#[["text","aspect_term","term_location"]])
    # print(x_vect)
    # x_vector = tfidf_vectorize(d_x)
    # print(x_vect.shape)
    print("Features extracted")
	
    '''Learn & Predict from ML model'''
    #k_neighbor(x_vect, d_y)
    #svm(x_vect, d_y)
    preds = finalSVM(x_vect, x_vect_test,d_y)#, dataTest['class'])
	#naive_bayes(x_vect, d_y)
    #decision_tree(x_vect, d_y)
    # cp_in_data = remove_stopwords(cp_in_data)
    print("Model trained and class predicted")
	
	#Write Output
    f=open("Chinmay_Gangak_Kislaya_Singh_Data-1.txt","w+")
    for i in range(len(preds)):
        strRec = str(i+1) + ";;" + str(preds[i]) + "\n"
        f.write(strRec)
    f.close()

def preprocess(data):
    data.text = data.text.str.replace("\[comma\]", ",")
    data.aspect_term = data.aspect_term.str.replace("\[comma\]", ",")
    data.text = data.text.str.replace("_", "")
    data.aspect_term = data.aspect_term.str.replace("_", "")
    data.text = data.text.str.replace(" '", " ")
    data.aspect_term = data.aspect_term.str.replace(" '", " ")
    data.text = data.text.str.replace("' ", " ")
    data.aspect_term = data.aspect_term.str.replace("' ", " ")
    data.text = data.text.str.replace("'s", " ")
    data.aspect_term = data.aspect_term.str.replace("'s", " ")
    #data.text = re.sub("'$", " ",data.text)
    data.text = data.text.apply(lambda row: re.sub("'$", " ",row))
    data.aspect_term = data.aspect_term.apply(lambda row: re.sub("'$", " ",row))
    data.text = data.text.apply(lambda row: re.sub("^'", " ",row))
    data.aspect_term = data.aspect_term.apply(lambda row: re.sub("^'", " ",row))

    data = to_lower(data)

    # data = in_data.copy(deep=True)
    # data.text = data["text"].apply(remove_tags)
    data = do_lemmatization(data)
    return data
    #print(list(data))
	
def extract_features(data):
    x_vect = get_vectorized_ngram_data(data[['text','aspect_term']])
    x_vect = scipy.sparse.csr_matrix(x_vect)
    x_vect = calc_adj_feature(x_vect, data)
    # dimensionality reduction
    
	
    return x_vect

	
def to_lower(data):
    data["text"] = data["text"].str.lower()
    data["aspect_term"] = data["aspect_term"].str.lower()
    return data


def apply_chi2(x, y):
    #print("###########")
    #print(x)
    # chi2_val= chi2(x, y)
    chi2_selector = SelectKBest(chi2, k=4000)
    X_kbest = chi2_selector.fit_transform(x, y)
    #print(X_kbest)
    #import time
    #time.sleep(100)
    return X_kbest


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


def get_vectorized_ngram_data(txt_asp_data):
    
    textList = []
    for id,row in txt_asp_data.iterrows():
        textWords = do_re_tokenize(row['text'])
        try:
            aspIndex = textWords.index(do_re_tokenize(row['aspect_term'])[0])
            xtrWords = 5
            startIndex = max(aspIndex - xtrWords,0)
            endIndex = min(aspIndex + xtrWords,len(textWords))
            textList.append(' '.join(textWords[startIndex:endIndex+1]))
        except:
            #print(row['text'])
            #print("textWords:",textWords)
            #print("aspect:",row['aspect_term'])
            #print("aspect 1st word:",do_re_tokenize(row['aspect_term'])[0])
            textList.append(row['text'])
    
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    x_vec = tfidf.fit_transform(textList)
    # print("Features:",tfidf.get_feature_names())
    # print(x_vec.toarray())
    # idf_sum = [sum(x_vec[i]) for i in range(3602)]
    return x_vec

def finalSVM(X_train, X_test, Y_train):#, Y_test):
    svc = LinearSVC(dual=False)
    svc.fit(X_train, Y_train)
    preds = svc.predict(X_test)
    #print(svc.score(X_test, Y_test))
    return preds
	
def svm(x, y):
    svc = LinearSVC(dual=False)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []
    precision_list_pos = []
    precision_list_neg = []
    precision_list_neutral = []
    recall_list_pos = []
    recall_list_neg = []
    recall_list_neutral = []

    for train_index, test_index in skf.split(x, y):
        # print()
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        accuracy_list.append(svc.score(x_test, y_test))
        #cm = confusion_matrix(y_test, y_pred, labels=[1, 0, 2])
        # print(cm)

        #tp_pos = cm[0][0]
        # tn = cm[2][2]
        #fp = cm[0][2]
        #fn = cm[2][0]

        #tp_neg = cm[2][2]

        #tp_neutral = cm[1][1]

        #prec_pos = tp_pos / (cm[0][0] + cm[1][0] + cm[2][0])
        #prec_neg = tp_neg / (cm[0][2] + cm[1][2] + cm[2][2])
        # print(tp_neg,cm[0][2],cm[1][2],cm[2][2])
        #prec_neutral = tp_neutral / (cm[0][1] + cm[1][1] + cm[2][1])
        #rec_pos = tp_pos / (cm[0][0] + cm[0][1] + cm[0][2])
        #rec_neg = tp_neg / (cm[2][0] + cm[2][1] + cm[2][2])
        #rec_neutral = tp_neutral / (cm[1][0] + cm[1][1] + cm[1][2])
        #precision_list_pos.append(prec_pos)
        #precision_list_neg.append(prec_neg)
        #precision_list_neutral.append(prec_neutral)
        #recall_list_pos.append(rec_pos)
        #recall_list_neg.append(rec_neg)
        #recall_list_neutral.append(rec_neutral)

    accuracy = np.mean(accuracy_list)
    precision_pos = np.mean(precision_list_pos)
    precision_neg = np.mean(precision_list_neg)
    precision_neutral = np.mean(precision_list_neutral)
    recall_pos = np.mean(recall_list_pos)
    recall_neg = np.mean(recall_list_neg)
    recall_neutral = np.mean(recall_list_neutral)
    # print("Precision for svc (class 1) is : " + str(precision_pos))
    # print("Precision for svc (class -1) is : " + str(precision_neg))
    # print("Precision for svc (class 0) is : " + str(precision_neutral))
    # print("Recall for svc (class 1) is : " + str(recall_pos))
    # print("Recall for svc (class -1) is : " + str(recall_neg))
    # print("Recall for svc (class 0) is : " + str(recall_neutral))
    print("Accuracy for svc is : " + str(accuracy))


def calc_adj_feature(x_vect, myData):
    text_file = open("pos_words.txt", "r")
    pos_words = text_file.read().split('\n')
    text_file = open("neg_words.txt", "r")
    neg_words = text_file.read().split('\n')

    conjunct_list = ['but', 'nor', 'yet', 'although', 'before', 'if', 'though', 'till', 'unless', 'until', 'what',
                     'whether', 'while','.']

    adj_class = []
    adj_dist = []
    nearest_adj = []
    pos_count = []
    neg_count = []
    sub_sent_counts = []

    for id, row in myData.iterrows():
        min_dist = len(row['text'])

        my_text = row['text']
        my_text2 = my_text

        for word in do_re_tokenize(my_text2):
            if word in conjunct_list:
                my_text2 = my_text2.replace(word, "~")

        sub_sent_count = len(my_text2.split('~'))

        for subText in my_text2.split('~'):
            if row['aspect_term'] in subText:
                # print("Full text: ",my_text)
                # print("subText: ",subText)
                # print("aspect:",row['aspect_term'])
                my_text = subText
                break

        # print("text: ",row['text'])
        # print("aspect: ",row['aspect_term'])
        try:
            term_start = my_text.index(row['aspect_term'])  # int(re.split('--',row['term_location'])[0])
            term_end = term_start + len(row['aspect_term'])  # int(re.split('--',row['term_location'])[1])
        except:
            term_start = int(re.split('--', row['term_location'])[0])
            term_end = int(re.split('--', row['term_location'])[1])
        # print("start: ",term_start, " end: ",term_end)
        negCount = 0
        posCount = 0
        for str in do_re_tokenize(my_text):

            # print("str: ",str,"\t index:",row['text'].index(str))
            if str in pos_words:
                posCount = posCount + 1
            elif str in neg_words:
                negCount = negCount + 1
            try:
                # if my_text.index(str) >= term_start and my_text.index(str) < term_end:
                if term_start <= my_text.index(str) < term_end:
                    continue
            except:
                print()
				#print("Error line 94")
                #print("text: ", my_text)
                #print("str: ", str)'''
                # print("index: ",row['text'].index(str))

            try:
                dist = abs(my_text.index(str) - my_text.index(row['aspect_term']))
            except ValueError:
                #print("Aspect Term missing in Text")
                #print("id: ", id)
                #print("text: ", my_text)
                #print("aspect: ", row['aspect_term'])'''
                # print("start: ",term_start, " end: ",term_end)
                # finally:
                adjClass = 0
                min_dist = len(my_text)
                near_adj = str
                break

            if dist < min_dist:
                # min_dist = dist
                near_adj = str
                if str in pos_words:
                    adjClass = +1  # Positive
                    min_dist = dist
                    # near_adj = str
                elif str in neg_words:
                    adjClass = 2  # Negative
                    min_dist = dist
                    # near_adj = str
                else:
                    adjClass = 0  # Neutral
        adj_class.append(adjClass)
        nearest_adj.append(near_adj)
        adj_dist.append(min_dist)
        pos_count.append(posCount)
        neg_count.append(negCount)
        sub_sent_counts.append(sub_sent_count)

    # myData["nearest_adj"] = nearest_adj
    myData["adj_class"] = adj_class
    myData["adj_dist"] = adj_dist
    myData["pos_count"] = pos_count
    myData["neg_count"] = neg_count
    myData["sub_sent_counts"] = sub_sent_counts
    
    myData2 = myData[["adj_class", "adj_dist", "pos_count", "neg_count"]]
    myData2 = scipy.sparse.csr_matrix(myData2)
    #print('x_vect shape:',x_vect.shape)
    #print('myData2 shape:',myData2.shape)

    return scipy.sparse.hstack((x_vect,myData2)).tocsr()
    # return myData[["aspect_start", "aspect_end", "text_len", "adj_class", "adj_dist", "pos_count", "neg_count", "idf_score", "idf_aspect"]]

'''
def myFunc(myData):
    tfidf = TfidfVectorizer(use_idf=True)
    # x_vec = tfidf.fit_transform(myData.text)
    # print(x_vec.shape)
    # print("Features:",tfidf.get_feature_names())
    # idf_sum = [sum(x_vec[i]) for i in range(3602)]
    # idf_sum = x_vec.sum(axis=1)
    # print(idf_sum.shape)
    
    # split term_location into start and end
    # print(data.sample(n=5))
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
        
        word_tokens = do_re_tokenize(row['text'])
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
    x_vec_aspect = tfidf.fit_transform(myData.aspect_term)
    #print(x_vec[0])
    #print(tfidf.inverse_transform(x_vec)[0])
    #print(np.where(tfidf.inverse_transform(x_vec)[0] == 'staff')[0])
    
    #for id,row in myData.iterrows():    
    
    #print("Features:",tfidf.get_stop_words())
    #print(tfidf['abbys'])
    idf_sum_aspect = x_vec_aspect.sum(axis=1)
    idf_sum = x_vec.sum(axis=1)
    myData["idf_score"] = idf_sum
    myData["idf_aspect"] = idf_sum_aspect
    
    #print(data.sample(n=5))
    return myData[["text","aspect_term","term_location","aspect_start","aspect_end","text_len", "idf_score","idf_aspect", "sense_text"]]
'''


def k_neighbor(x, y):
    knn = KNeighborsClassifier(n_neighbors=100)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []
    precision_list_pos = []
    precision_list_neg = []
    precision_list_neutral = []
    recall_list_pos = []
    recall_list_neg = []
    recall_list_neutral = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy_list.append(knn.score(x_test, y_test))
        cm = confusion_matrix(y_test, y_pred,labels=[1,0,-1])
        tp_pos = cm[0][0]
        # tn = cm[2][2]
        fp = cm[0][2]
        fn = cm[2][0]
        
        tp_neg = cm[2][2]
        
        tp_neutral = cm[1][1]
        
        prec_pos = tp_pos/(cm[0][0] + cm[1][0] + cm[2][0])
        prec_neg = tp_neg/(cm[0][2] + cm[1][2] + cm[2][2])
        # print(tp_neg,cm[0][2],cm[1][2],cm[2][2])
        prec_neutral = tp_neutral/(cm[0][1] + cm[1][1] + cm[2][1])
        rec_pos = tp_pos/(cm[0][0] + cm[0][1] + cm[0][2])
        rec_neg = tp_neg/(cm[2][0] + cm[2][1] + cm[2][2])
        rec_neutral = tp_neutral/(cm[1][0] + cm[1][1] + cm[1][2])
        precision_list_pos.append(prec_pos)
        precision_list_neg.append(prec_neg)
        precision_list_neutral.append(prec_neutral)
        recall_list_pos.append(rec_pos)
        recall_list_neg.append(rec_neg)
        recall_list_neutral.append(rec_neutral)

    accuracy = np.mean(accuracy_list)
    precision_pos = np.mean(precision_list_pos)
    precision_neg = np.mean(precision_list_neg)
    precision_neutral = np.mean(precision_list_neutral)
    recall_pos = np.mean(recall_list_pos)
    recall_neg = np.mean(recall_list_neg)
    recall_neutral = np.mean(recall_list_neutral)
    print("Accuracy for knn is : " + str(accuracy))
    print("Precision for knn (class 1) is : " + str(precision_pos))
    print("Precision for knn (class -1) is : " + str(precision_neg))
    print("Precision for knn (class 0) is : " + str(precision_neutral))
    print("Recall for knn (class 1) is : " + str(recall_pos))
    print("Recall for knn (class -1) is : " + str(recall_neg))
    print("Recall for knn (class 0) is : " + str(recall_neutral))


def decision_tree(x, y):
    dtree = DecisionTreeClassifier(max_depth=3)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []
    precision_list_pos = []
    precision_list_neg = []
    precision_list_neutral = []
    recall_list_pos = []
    recall_list_neg = []
    recall_list_neutral = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtree.fit(x_train, y_train)
        y_pred = dtree.predict(x_test)
        accuracy_list.append(dtree.score(x_test, y_test))
        cm = confusion_matrix(y_test, y_pred)
        tp_pos = cm[0][0]
        # tn = cm[2][2]
        fp = cm[0][2]
        fn = cm[2][0]
        
        tp_neg = cm[2][2]
        
        tp_neutral = cm[1][1]
        
        prec_pos = tp_pos/(cm[0][0] + cm[1][0] + cm[2][0])
        prec_neg = tp_neg/(cm[0][2] + cm[1][2] + cm[2][2])
        # print(tp_neg,cm[0][2],cm[1][2],cm[2][2])
        prec_neutral = tp_neutral/(cm[0][1] + cm[1][1] + cm[2][1])
        rec_pos = tp_pos/(cm[0][0] + cm[0][1] + cm[0][2])
        rec_neg = tp_neg/(cm[2][0] + cm[2][1] + cm[2][2])
        rec_neutral = tp_neutral/(cm[1][0] + cm[1][1] + cm[1][2])
        precision_list_pos.append(prec_pos)
        precision_list_neg.append(prec_neg)
        precision_list_neutral.append(prec_neutral)
        recall_list_pos.append(rec_pos)
        recall_list_neg.append(rec_neg)
        recall_list_neutral.append(rec_neutral)

    accuracy = np.mean(accuracy_list)
    precision_pos = np.mean(precision_list_pos)
    precision_neg = np.mean(precision_list_neg)
    precision_neutral = np.mean(precision_list_neutral)
    recall_pos = np.mean(recall_list_pos)
    recall_neg = np.mean(recall_list_neg)
    recall_neutral = np.mean(recall_list_neutral)
    print("Accuracy for dtree is : " + str(accuracy))
    print("Precision for dtree (class 1) is : " + str(precision_pos))
    print("Precision for dtree (class -1) is : " + str(precision_neg))
    print("Precision for dtree (class 0) is : " + str(precision_neutral))
    print("Recall for dtree (class 1) is : " + str(recall_pos))
    print("Recall for dtree (class -1) is : " + str(recall_neg))
    print("Recall for dtree (class 0) is : " + str(recall_neutral))


def naive_bayes(x, y):
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []
    precision_list_pos = []
    precision_list_neg = []
    precision_list_neutral = []
    recall_list_pos = []
    recall_list_neg = []
    recall_list_neutral = []
    
    for train_index, test_index in skf.split(x, y):
        gnb = MultinomialNB()
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        accuracy_list.append(gnb.score(x_test, y_test))
        cm = confusion_matrix(y_test, y_pred)
        tp_pos = cm[0][0]
        # tn = cm[2][2]
        fp = cm[0][2]
        fn = cm[2][0]
        
        tp_neg = cm[2][2]
        
        tp_neutral = cm[1][1]
        
        prec_pos = tp_pos/(cm[0][0] + cm[1][0] + cm[2][0])
        prec_neg = tp_neg/(cm[0][2] + cm[1][2] + cm[2][2])
        # print(tp_neg,cm[0][2],cm[1][2],cm[2][2])
        prec_neutral = tp_neutral/(cm[0][1] + cm[1][1] + cm[2][1])
        rec_pos = tp_pos/(cm[0][0] + cm[0][1] + cm[0][2])
        rec_neg = tp_neg/(cm[2][0] + cm[2][1] + cm[2][2])
        rec_neutral = tp_neutral/(cm[1][0] + cm[1][1] + cm[1][2])
        precision_list_pos.append(prec_pos)
        precision_list_neg.append(prec_neg)
        precision_list_neutral.append(prec_neutral)
        recall_list_pos.append(rec_pos)
        recall_list_neg.append(rec_neg)
        recall_list_neutral.append(rec_neutral)

    accuracy = np.mean(accuracy_list)
    precision_pos = np.mean(precision_list_pos)
    precision_neg = np.mean(precision_list_neg)
    precision_neutral = np.mean(precision_list_neutral)
    recall_pos = np.mean(recall_list_pos)
    recall_neg = np.mean(recall_list_neg)
    recall_neutral = np.mean(recall_list_neutral)
    print("Accuracy for nb is : " + str(accuracy))
    print("Precision for nb (class 1) is : " + str(precision_pos))
    print("Precision for nb (class -1) is : " + str(precision_neg))
    print("Precision for nb (class 0) is : " + str(precision_neutral))
    print("Recall for nb (class 1) is : " + str(recall_pos))
    print("Recall for nb (class -1) is : " + str(recall_neg))
    print("Recall for nb (class 0) is : " + str(recall_neutral))


def tokenize(cp_in_data):
    cp_in_data["text"] = cp_in_data["text"].apply(lambda row: word_tokenize(row))
    return cp_in_data


def do_re_tokenize(row):
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


def do_lemmatization(data_stem):
    # PORTER STEMMER
    #print("Porter Stemmer")
    porter_stemmer = PorterStemmer()

    data_stem["text"] = data_stem["text"].apply(do_re_tokenize)
    # data_stem['text'] = data_stem['text'].apply(lambda x: [porter_stemmer.stem(y) for y in x])
    data_stem["text"] = data_stem["text"].apply(lambda x: [wnl.lemmatize(y) for y in x])
    data_stem["text"] = data_stem["text"].apply(lambda x: " ".join(x))

    data_stem["aspect_term"] = data_stem["aspect_term"].apply(do_re_tokenize)
    # data_stem['aspect_term'] = data_stem['aspect_term'].apply(lambda x: [porter_stemmer.stem(y) for y in x])
    data_stem["aspect_term"] = data_stem["aspect_term"].apply(lambda x: [wnl.lemmatize(y) for y in x])
    data_stem["aspect_term"] = data_stem["aspect_term"].apply(lambda x: " ".join(x))

    return data_stem


load_data(in_data_file)
