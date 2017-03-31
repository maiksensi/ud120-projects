#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

from sklearn.svm import SVC
from pprint import pprint

t0 = time()
clf = SVC(kernel="rbf", C=10000)
X = features_train
Y = labels_train
clf.fit(X,Y)
print "training time:", round(time()-t0, 3), "s"

t0 = time()

count = 0
for event in clf.predict(features_test):
    if event == 1:
        count += 1

pprint(count)

print "prediciton time:", round(time()-t0, 3), "s"

#from sklearn.metrics import accuracy_score
#acc = accuracy_score(pred, labels_test)


#pprint(acc)

def submitAccuracy():
    return acc

