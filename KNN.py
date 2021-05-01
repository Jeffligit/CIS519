from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
from sklearn.metrics import f1_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.model_selection import cross_val_score as cvs
from sklearn.tree import DecisionTreeClassifier

# path
directory = (r"D:\迅雷下载\519_project\genres")

# Define feature and output sets
X1 = []
X2 = []
y = []
i = 0

# Read file and extract data
for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory + "\\" + folder):
        (rate, sig) = wav.read(directory + "\\" + folder + "\\" + file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False, preemph=0.95)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        X1.append(mean_matrix)
        X2.append(covariance)
        y.append(i)

y = np.array(y)
X = np.array(X1)


def knn_1(data, label, loop):
    X_train, X_test, y_train, y_test = train_test_split(data, label)
    for j in range(loop):
        knn_classifier = KNeighborsClassifier(j + 1)
        knn_classifier.fit(X_train, y_train)
        scores = knn_classifier.score(X_test, y_test)
        print(f'k = {j + 1}, accuracy = {scores}')


def svm_1(data, label, kernel):
    X_train, X_test, y_train, y_test = train_test_split(data, label)
    svm_classifier = svm.SVC(C=1.0, kernel=kernel)
    svm_classifier.fit(X_train, y_train)
    scores = svm_classifier.score(X_test, y_test)
    print(f'Kernel = rbf, accuracy = {scores}')


def knn_pca(data, label, n_comp, loop):
    X_train, X_test, y_train, y_test = train_test_split(data, label)
    pca = PCA(n_components=n_comp)
    pca.fit(X_train)
    train_pca = pca.transform(X_train)
    test_pca = pca.transform(X_test)

    # lda = LDA(n_components=0.9)
    # lda.fit(train_pca, y_train)
    # train_lda = lda.transform(train_pca)
    # test_lda = lda.transform(test_pca)

    # mms = MinMaxScaler()
    for j in range(loop):
        knn = KNeighborsClassifier(j+1)

        # train_lda_std = mms.fit(train_lda)
        knn.fit(train_pca, y_train)

        scores = knn.score(test_pca, y_test)
        print(f'KNN with k = {j+1}, PCA_test accuracy = {scores}')


def automatic_dt_pruning(dt_classifier, data, label):
    np.random.seed(42)
    alpha = []
    score = []
    for k in range(0, 100):
        ccp_alpha_test = k / 100
        dt_classifier.set_params(ccp_alpha=ccp_alpha_test)
        alpha.append(ccp_alpha_test)
        score.append(cvs(dt_classifier, data, label, cv=5).mean())

    best_ccp_alpha = alpha[score.index(max(score))]

    return best_ccp_alpha


def test_dt_pruning():
    np.random.seed(42)
    data = pd.DataFrame(X)
    label = pd.Series(y)
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    ccp_alpha = automatic_dt_pruning(clf, data, label)
    clf.set_params(ccp_alpha=ccp_alpha)
    clf.fit(data, label)
    dt_acc = clf.score(data, label)
    print(dt_acc)


print('############## KNN #############################')
knn_1(X, y, 10)
print('############### SVM  ##############################')
svm_1(X, y, 'rbf')
print('###########  KNN with PCA  #########################')
knn_pca(X, y, 0.99, 10)
print('################  Decision Tree  #########################')
test_dt_pruning()
