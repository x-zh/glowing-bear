# import numpy as np, h5py
# f = h5py.File(r'C://Users//Xin//Downloads//finalCS6923.mat','r')
# data = f.get('test')
# data = np.array(data) # For converting to numpy array


import scipy.io
import numpy
from sklearn import svm
mat = scipy.io.loadmat('finalCS6923.mat')

train_label = mat.get('train_label')
train = mat.get('train')
clf = svm.SVC(kernel='linear')
clf.fit(train[:100], numpy.ravel(train_label)[:100])
print clf.predict(mat.get('test')[:100])
print ''
