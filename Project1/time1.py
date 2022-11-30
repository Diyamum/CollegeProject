
import re
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st = time.time()
def read_data(file):
    data = []
    with open(file, 'r')as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

file = "C:\\Users\\dipay\\Desktop\\data.txt"
data = read_data(file)
print("Number of instances: {}".format(len(data)))

def ngram(token, n):
    output = []
    for i in range(n-1, len(token)):
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram)
    return output

def create_feature(text, nrange=(1, 1)):
    text_features = []
    text = text.lower()
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1):
        text_features += ngram(text_alphanum.split(), n)
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)


def convert_label(item, name):
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)):
        if items[idx] == 1:
            label += name[idx] + " "

    return label.strip()


emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))


X_train, X_rem, y_train, y_rem = train_test_split(X_all,y_all, train_size=0.8)


test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

def train_test(clf, X_train,X_valid, X_test, y_train,y_valid, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    valid_acc=accuracy_score(y_valid, clf.predict(X_valid))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, valid_acc, test_acc

from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse = True)
X_train = vectorizer.fit_transform(X_train)
X_valid= vectorizer.transform(X_valid)
X_test = vectorizer.transform(X_test)

svc = SVC()
lsvc = LinearSVC(random_state=123)
rforest = RandomForestClassifier(random_state=124)
dtree = DecisionTreeClassifier()

clifs = [svc ,lsvc, rforest, dtree]

# train and test them
print("| {:40} | {} |  {} | {} |".format("Classifier", "Training Accuracy", "Validation Accuracy","Test Accuracy"))
print("| {} | {} |  {} | {} |".format("-"*40, "-"*17, "-"*19,"-"*13))


l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
l.sort()
label_freq = {}
for label, _ in data:
    label_freq[label] = label_freq.get(label, 0) + 1

t1 = "This looks so impressive"
t2 = "I have a fear of dogs"
t3 = "My dog died yesterday"
t4 = "I don't love you anymore..!"

texts = [t1, t2, t3, t4]
# print the labels and their counts in sorted order
for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))
    emoji_dict = {"joy": "😂", "fear": "😱", "anger": "😠", "sadness": "😢", "disgust": "😒", "shame": "😳", "guilt": "😳"}



for clf in clifs:
    clf_name = clf.__class__.__name__
    train_acc, valid_acc, test_acc = train_test(clf, X_train, X_valid, X_test, y_train, y_valid, y_test)
    print("| {:40} | {:17f} | {:19f} | {:13f} |\n".format(clf_name, train_acc,valid_acc, test_acc))
    for text in texts:
        features = create_feature(text, nrange=(1, 4))
        features = vectorizer.transform(features)
        prediction = clf.predict(features)[0]
        print(text, emoji_dict[prediction])
et = time.time()
elapsed_time = et - st
el_time= elapsed_time/60
print('\nTime Elapsed:', "%.2f" %el_time, 'Mins')