import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from collections import Counter
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import time
import re
import spacy

nlp = spacy.load('en')


def loading_data(file_path):
    return_me = []
    temp = []
    with open(file_path) as file:
        line = file.readline()
        while line:
            item = tuple(line.replace('\n', '').split('\t'))
            if item != ('',):
                temp.append(item)
            line = file.readline()
    first = True
    for item in temp:
        if item[0] == '0' and first:
            temp_sents = []
            temp_sents.append(item[1:4])
            first = False
        elif item[0] == '0':
            return_me.append(temp_sents)
            temp_sents = []
            temp_sents.append(item[1:4])
        else:
            temp_sents.append(item[1:4])
    return return_me


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-5:]': word[-5:],
        'word[-4:]': word[-4:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'facility_found': word in lda_facility and not word.islower(),
        'person_found': word in lda_person and not word.islower(),
        'location_found': word in lda_location and not word.islower(),
        'company_found': word in lda_company and not word.islower(),
        'shape': shape_dict[word]
    }
    if word.lower() in twitter_cluster_dict:
        features['twitter_cluster'] = twitter_cluster_dict[word.lower()]
    else:
        features['twitter_cluster'] = 'OOV'
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:shape': shape_dict[word1]

        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:shape': shape_dict[word1]
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def token2doc():
    doc = []
    for sent in train_sents:
        st = ' '.join(sent2tokens(sent))
        doc.append(st + "\n")

    fh = open("train.doc", "w")
    fh.writelines(doc)
    fh.close()


def crf():
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


def result_output():
    output = []
    flat_list = [item for sublist in y_pred for item in sublist]

    with open('test.raw') as file:
        line = file.readline()
        count = 0
        while line:
            if line != '\n':
                if count < len(flat_list):
                    output.append(line.replace('\n', '') + "\t" + flat_list[count] + "\n")
                else:
                    output.append(line.replace('\n', '') + "\t" + "O" + "\n")  # Misc
                line = file.readline()
                count += 1
            else:
                output.append("\n")
                line = file.readline()

    fh = open("test.gen", "w")
    fh.writelines(output)
    fh.close()


# reference: http://www.cs.cmu.edu/~ark/TweetNLP/cluster_viewer.html
def twitter_cluster():
    return_me = {}
    with open("twitter_cluster.txt") as file:
        line = file.readline()
        while line:
            temp = line.replace('\n', '').split('\t')
            cluster_num = temp[0].replace('^', '').split(' ')[0]
            for word in temp[1].split(' '):
                return_me[word] = cluster_num
            line = file.readline()
    return return_me


def lda_data(file_path):
    return_me = []
    with open(file_path) as file:
        line = file.readline()
        while line:
            trim_length = re.sub(r'\b\w{1,3}\b', '', line.replace('\n', ''))
            remove_symbol = re.sub(r'[^\w]', ' ', trim_length)
            temp = remove_symbol.split(' ')
            return_me.extend(temp)
            line = file.readline()
    return set(return_me)


def shape_data(file_path):
    return_me = {}
    with open(file_path) as file:
        line = file.readline()
        while line:
            temp = line.replace('\n', '').split('\t')
            return_me[temp[0]] = temp[1]
            line = file.readline()
    return return_me


train_sents = loading_data('train.gold')
dev_sents = loading_data('dev.gold')
test_sents = loading_data('test.gold')

shape_dict = {}
shape_dict.update(shape_data('train.shape'))
shape_dict.update(shape_data('dev.shape'))
shape_dict.update(shape_data('test.shape'))

print("Loading Twitter Brown Cluster..")
twitter_cluster_dict = twitter_cluster()

print("Loading NE data..")
lda_facility = lda_data("lda_facility.txt")
lda_person = lda_data("lda_person.txt")
lda_location = lda_data("lda_location.txt")
lda_company = lda_data("lda_company.txt")

print("Feature representation example:", sent2features(train_sents[0])[0])

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

print(X_train)
print(y_train)

X_dev = [sent2features(s) for s in dev_sents]
y_dev = [sent2labels(s) for s in dev_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

train_label_list = set([item for sublist in y_train for item in sublist])
dev_label_list = set([item for sublist in y_dev for item in sublist])
test_label_list = set([item for sublist in y_test for item in sublist])

start_time = time.time()
print("Start training...")
crf = crf()
print("Finished. Time: {0:.2g}s".format(time.time() - start_time))

'''
FAC: facility
ORG: organization
PER: person
LOC: location
GPE: geopolitical entity
VEH: vehicle
'''
# labels = list(crf.classes_)
# labels.remove('O')
labels = list(test_label_list)
labels.remove('O')
print("Labels for test:", labels)

y_pred = crf.predict(X_test)

sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
print(sorted_labels)
print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
print(y_test[0:10])
print(y_pred[0:10])

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(reversed(Counter(crf.state_features_).most_common()[-30:]))


# For output purpose
# result_output()
