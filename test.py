import scipy.io
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics


mat = scipy.io.loadmat('finalCS6923.mat')
train_label = mat.get('train_label')
train = mat.get('train')
# for i in range(10):
#     print train[i]
#     print train_label[i]

SVM_PARAMETERS = [
    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    {'C': [1, 10, 100], 'kernel': ['linear']}
]

LINEAR_REGRESSION_PARAMETERS = [{'C': [0.0001, 0.0003, 0.001, 0.003, 0.1, 0.3, 1, 3, 10, 30, 100]}]


def load_data(N=10000):
    return train[:N], np.ravel(train_label)[:N]

def ml(method='svm', num_iteration=1):    
    X, y = load_data()
    X = np.array(X, dtype=np.float64) # convert to float
    X_scaled = preprocessing.scale(X) # normalize
    clf = linear_model.LogisticRegression()

    # crossvalidation
    # scores = cross_validation.cross_val_score(clf, X_scaled, y, cv=10)
    # print(scores.mean(), scores)

    # grid search Logistic regression
    if method == 'regression':
        logReg = linear_model.LogisticRegression()
        clf = GridSearchCV(logReg, LINEAR_REGRESSION_PARAMETERS, cv=10)
        clf.fit(X_scaled, y)
        print 'best_score:', clf.best_score_
        print 'best_params:', clf.best_params_

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, y, test_size=0.10, random_state=111)
        clf = linear_model.LogisticRegression(C=clf.best_params_.get('C'))
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        print 'accuracy_score:', metrics.accuracy_score(y_test, predicted)

    # grid search SVM
    if method == 'svm':
        # s = svm.SVC()
        # clf = GridSearchCV(s, SVM_PARAMETERS, cv=10)
        # clf.fit(X_scaled, y)
        # print 'best_score:', clf.best_score_
        # print 'best_params:', clf.best_params_
        
        # -- rbf --
        # best_params: {'kernel': 'rbf', 'C': 35, 'gamma': 0.09}
        # accuracy_score: 0.807
        # 
        
        clf = svm.SVC(
            C=35, #clf.best_params_.get('C'),
            cache_size=200,
            class_weight=None,
            coef0=0.0,
            degree=3,
            gamma=0.09,
            kernel='rbf',  # linear / rbf 
            max_iter=-1,
            probability=False,
            random_state=None,
            shrinking=True,
            tol=0.001,
            verbose=False)  # kernel='linear')
        ave_accuracy_score = 0
        for ni in xrange(num_iteration):
            seed = np.random.randint(10000)
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, y, test_size=0.10, random_state=seed)
            clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)
            accuracy_score = metrics.accuracy_score(y_test, predicted)
            print '[%d/%d] accuracy_score:%f\tseed:%d' % (ni + 1, num_iteration, accuracy_score, seed)
            ave_accuracy_score += accuracy_score
        print '== Average accuracy score: %f ==' % (ave_accuracy_score / num_iteration)

# ml('regression')
ml(method='svm', num_iteration=10)
