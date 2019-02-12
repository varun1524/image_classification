
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix


# In[54]:


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


# In[65]:


print(len(lines_train))
print(len(lines_test))
print(len(classes))


# In[66]:


print(lines_train[:1])
# print(lines_test[:1])
# print(classes[:1])


# In[67]:


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

    
from time import time


# In[68]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[69]:


X_train_std = StandardScaler().fit_transform(lines_train)
X_test_std = StandardScaler().fit_transform(lines_test)


# In[70]:


print(type(X_train_std))
print(X_train_std.shape)
print(type(X_test_std))
print(X_test_std.shape)


# In[72]:


pca = PCA(n_components=150)
pca.fit(X_train_std)
print(type(pca))


# In[73]:


X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


# In[74]:


print(X_train_pca.shape)
print(X_train_pca)


# In[75]:


# vectorizer = HashingVectorizer(stop_words='english')
# X_train = vectorizer.transform(lines_train)
# X_test = vectorizer.transform(lines_test)
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
# X_train = vectorizer.fit_transform(docs_train)


y_train = np.asarray(classes)


# In[76]:


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


# In[78]:


pred1 = classify(KNeighborsClassifier(n_neighbors=3), X_train_pca, y_train, X_test_pca)


# In[79]:


def writeToFile(pred):
    with open('output/output_KNeighbor.dat','w+') as f:
        for p in pred:
            f.write(str(p)+"\n")


# In[81]:


writeToFile(pred1)

