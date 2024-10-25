#%%
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

X, y = make_blobs(n_samples=250, centers=2,
                  random_state=0, cluster_std=0.60)
y[y == 0] = -1 # labels as -1 or +1
tmp = np.ones(len(X))
y = tmp * y

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


our_svm = LinearSVC()
our_svm.fit(X_train, y_train) # training function

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter');
ax = plt.gca()
xlim = ax.get_xlim()
w = our_svm.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])

yy = a * xx - our_svm.intercept_[0] / w[1]
plt.plot(xx, yy)

yy = a * xx - (our_svm.intercept_[0] - 1) / w[1]
plt.plot(xx, yy, 'k--')

yy = a * xx - (our_svm.intercept_[0] + 1) / w[1]
plt.plot(xx, yy, 'k--')

y_pred = our_svm.predict(X_test)
cm=confusion_matrix(y_test, y_pred)

plt.matshow(cm, cmap = 'Blues')
plt.colorbar()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

plt.show()

