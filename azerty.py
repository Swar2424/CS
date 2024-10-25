#%%
import scipy.io as sio
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix


mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
N = len(face_data)
M = 100

mean = np.mean(np.stack(face_data, axis=1), axis=0)

#print(mean, mean.shape)

#print(face_data[:,0], face_data[:,0].shape)

A = (face_data.T - mean)

S = 1/N * (np.dot(A,A.T))

eigvals, eigvecs = np.linalg.eig(S)

s_eigvecs = [x for _,x in sorted(zip(eigvals,eigvecs))][M:]
s_eigvals = sorted(eigvals)[M:]

print(s_eigvals)
#print(s_eigvecs)