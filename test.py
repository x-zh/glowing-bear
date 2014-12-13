import scipy.io
import numpy
from sklearn import svm
from sklearn import linear_model

mat = scipy.io.loadmat('finalCS6923.mat')
train_label = mat.get('train_label')
train = mat.get('train')
# for i in range(10):
#     print train[i]
#     print train_label[i]


def lr():
    clf = linear_model.LogisticRegression(C=1e7)
    clf.fit(train, train_label)
    clf.fit(train[:6000], train_label[:6000])
    err = 0
    t = clf.predict(mat.get('train')[9000:])
    for i in range(1000):
        if t[i] != train_label[i + 9000][0]:
            err += 1
    print err

lr()


def clf():
    clf = svm.SVC(
        C=1.0,
        cache_size=200,
        class_weight=None,
        coef0=0.0,
        degree=3,
        gamma=0.0,
        kernel='rbf',
        max_iter=-1,
        probability=False,
        random_state=None,
        shrinking=True,
        tol=0.001,
        verbose=False)  # kernel='linear')
    clf.fit(train[:6000], numpy.ravel(train_label)[:6000])
    err = 0
    t = clf.predict(mat.get('train')[8000:])
    for i in range(2000):
        if t[i] != train_label[i + 8000][0]:
            err += 1
    print err
