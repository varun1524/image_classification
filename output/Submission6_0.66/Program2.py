
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix


# In[2]:


lines_train = []
lines_test = []
classes = []
categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

with open("data/train.dat", "r") as fh:
    lines = fh.readlines()
    for l in lines:
        lines_train.append(l.rstrip().split(' '))
    
with open("data/train.labels", "r") as fh:
    lines = fh.readlines()
    for l in lines:
        classes.append(l.rstrip())
    
with open("data/test.dat", "r") as fh:
    lines = fh.readlines()
    for l in lines:
        lines_test.append(l.rstrip().split(' '))


# In[3]:


print(len(lines_train))
print(len(lines_test))
print(len(classes))


# In[4]:


print(lines_train[:1])
# print(lines_test[:1])
# print(classes[:1])


# In[5]:


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

    
from time import time


# In[6]:


from sklearn.decomposition import PCA
# from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler


# In[7]:


X_train_std = StandardScaler().fit_transform(lines_train)
X_test_std = StandardScaler().fit_transform(lines_test)


# In[8]:


print(type(X_train_std))
print(X_train_std.shape)
print(type(X_test_std))
print(X_test_std.shape)


# In[9]:


pca = PCA(n_components=100)
pca.fit(X_train_std)
print(type(pca))


# In[10]:


X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


# In[11]:


print(X_train_pca.shape)
print(X_train_pca)


# In[12]:


# vectorizer = HashingVectorizer(stop_words='english')
# X_train = vectorizer.transform(lines_train)
# X_test = vectorizer.transform(lines_test)
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
# X_train = vectorizer.fit_transform(docs_train)


y_train = np.asarray(classes)


# In[13]:


# target_names = ["1", "2", "3", "4", "5"]
def classify(clf, X_train, Y_train, X_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, Y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    print(type(pred))
    return pred


# In[20]:


# pred1 = classify(KNeighborsClassifier(n_neighbors=200), X_train_pca, y_train, X_test_pca)
pred1 = classify(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False), X_train_pca, y_train, X_test_pca)


# In[21]:


def writeToFile(pred):
    with open('output/output_KNeighbor.dat','w+') as f:
        for p in pred:
            f.write(str(p)+"\n")


# In[22]:


print(type(pred1))
writeToFile(pred1)

