
# coding: utf-8

# In[496]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix


# In[497]:


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


# In[498]:


print(len(lines_train))
print(len(lines_test))
print(len(classes))


# In[499]:


# print(lines_train[:1])
# print(lines_test[:1])
# print(classes[:1])


# In[500]:


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from time import time


# In[501]:


from sklearn.preprocessing import StandardScaler
X_train_std = StandardScaler().fit_transform(lines_train)
X_test_std = StandardScaler().fit_transform(lines_test)
# vectorizer = HashingVectorizer()
# X_train_std = vectorizer.transform(lines_train)
# X_test_std = vectorizer.transform(lines_test)


# In[502]:


print(type(X_train_std))
print(X_train_std.shape)
print(type(X_test_std))
print(X_test_std.shape)


# In[503]:


# from sklearn.decomposition import PCA
# pca = PCA(n_components=25)
# pca.fit(X_train_std)
# X_train_pca = pca.transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)


# In[504]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=30, n_iter=10, random_state=42)
svd.fit(X_train_std)
X_train_pca = svd.transform(X_train_std)
X_test_pca = svd.transform(X_test_std)


# In[505]:


print(X_train_pca)
print(X_test_pca)


# In[506]:


# vectorizer = HashingVectorizer(stop_words='english')
# X_train = vectorizer.transform(lines_train)
# X_test = vectorizer.transform(lines_test)
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
# X_train = vectorizer.fit_transform(docs_train)


y_train = np.asarray(classes)


# In[532]:


# from imblearn.over_sampling import SMOTE
# smote = SMOTE(k_neighbors=2)
# X_resampled, y_resampled = smote.fit_sample(X_train_pca, y_train)
# # X_res_vis = pca.transform(X_resampled)
# print(type(X_resampled))


# In[557]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(ratio="minority")
X_resampled, y_resampled = ros.fit_sample(X_train_pca, y_train)


# In[558]:


# from imblearn.over_sampling import ADASYN 
# ada = ADASYN(random_state=10, n_neighbors=2)
# X_resampled, y_resampled = ada.fit_sample(X_train_pca, y_train)



# In[559]:


print(X_resampled.shape)
print(y_resampled.shape)


# In[560]:


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


# In[561]:


# pred1 = classify(KNeighborsClassifier(n_neighbors=4), X_train_pca, y_train, X_test_pca)
pred1 = classify(KNeighborsClassifier(n_neighbors=4), X_resampled, y_resampled, X_test_pca)
# pred1 = classify(SVC(gamma=2), X_resampled, y_resampled, X_test_pca)
# pred1 = classify(AdaBoostClassifier(), X_resampled, y_resampled, X_test_pca)


# In[562]:


def writeToFile(pred):
    with open('output/output_KNeighbor.dat','w+') as f:
        for p in pred:
            f.write(str(p)+"\n")


# In[563]:


print(type(pred1))
writeToFile(pred1)


# In[564]:


def readFile(filename):
    content = []
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip() for x in content] 
    return content


# In[565]:


def compare(p1, p2):
    match = 0
    total = len(p1)
    for i in range(0,total-1):                
        if(p1[i]==p2[i]):
            match = match+1
    per = ((float(match)/float(total))*100)
    return per
    


# In[568]:


# p1 = readFile("output/output_KNeighbor1.dat")
p2 = readFile("submissions/Submission2_0.8168/output_KNeighbor.dat")
print(compare(pred1,p2))
print(compare(pred1,readFile("submissions/Submission10/output_KNeighbor1.dat")))


# In[567]:


print(pred1[:10])
print(p2[:10])
print(len(pred1))
print(len(p2))

