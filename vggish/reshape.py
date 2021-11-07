import numpy as np
from sklearn.preprocessing import StandardScaler
a = r'./trainlabel.npy'
b = r'./devlabel.npy'
label1 = np.load(a)
label1 = label1.reshape((-1,1))
label2 = np.load(b)
label2 = label2.reshape((-1,1))
np.save('trainlabel.npy',label1)
np.save('devlabel.npy',label2)
# a = r'./train_data.npy'
# b = r'./develop_data.npy'
# data1 = np.load(a)
# data2 = np.load(b)
# X_train = StandardScaler().fit_transform(data1)
# X_test = StandardScaler().fit_transform(data2)
# np.save(a,X_train)
# np.save(b,X_test)