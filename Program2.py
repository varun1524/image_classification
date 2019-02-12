
# coding: utf-8

# In[187]:


import numpy as np
import pandas as pd
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix


# In[188]:


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


# In[189]:


print(len(lines_train[0]))
print(len(lines_test))
print(len(classes))


# In[190]:


# print(lines_train[:1])
# print(lines_test[:1])
# print(classes[:1])


# In[191]:


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from time import time


# In[192]:


from sklearn.preprocessing import StandardScaler
X_train_std = StandardScaler().fit_transform(lines_train)
X_test_std = StandardScaler().fit_transform(lines_test)
# vectorizer = HashingVectorizer()
# X_train_std = vectorizer.transform(lines_train)
# X_test_std = vectorizer.transform(lines_test)


# In[193]:


print(type(X_train_std))
print(X_train_std.shape)
print(type(X_test_std))
print(X_test_std.shape)


# In[194]:


# from sklearn.decomposition import PCA
# pca = PCA(n_components=48)
# pca.fit(X_train_std)
# X_train_svd = pca.transform(X_train_std)
# X_test_svd = pca.transform(X_test_std)


# In[195]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=48, n_iter=10, random_state=42)
svd.fit(X_train_std)
X_train_svd = svd.transform(X_train_std)
X_test_svd = svd.transform(X_test_std)


# In[196]:


print(X_train_svd)
print(X_test_svd)


# In[197]:


# vectorizer = HashingVectorizer(stop_words='english')
# X_train = vectorizer.transform(lines_train)
# X_test = vectorizer.transform(lines_test)
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
# X_train = vectorizer.fit_transform(docs_train)


y_train = np.asarray(classes)


# In[198]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(k_neighbors=2, ratio="minority")
X_resampled, y_resampled = smote.fit_sample(X_train_svd, y_train)
# X_res_vis = pca.transform(X_resampled)
print(type(X_resampled))


# In[199]:


# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(ratio="minority")
# X_resampled, y_resampled = ros.fit_sample(X_train_pca, y_train)


# In[200]:


# from imblearn.over_sampling import ADASYN 
# ada = ADASYN(random_state=10, n_neighbors=2)
# X_resampled, y_resampled = ada.fit_sample(X_train_pca, y_train)



# In[201]:


print(X_resampled.shape)
print(y_resampled.shape)


# In[202]:


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


# In[203]:


# pred1 = classify(KNeighborsClassifier(n_neighbors=4), X_train_pca, y_train, X_test_pca)
pred1 = classify(KNeighborsClassifier(n_neighbors=4), X_resampled, y_resampled, X_test_svd)
# pred1 = classify(SVC(gamma=2), X_resampled, y_resampled, X_test_pca)
# pred1 = classify(AdaBoostClassifier(), X_resampled, y_resampled, X_test_pca)


# In[204]:


def writeToFile(pred):
    with open('output/output_KNeighborSVD48.dat','w+') as f:
        for p in pred:
            f.write(str(p)+"\n")


# In[205]:


writeToFile(pred1)

