
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn import cross_validation 
from sklearn.metrics import accuracy_score
names = ['x','y','class'] 
df = pd.read_csv('C:/Users/SHUBI/Desktop/Applied AI/demo data for cross validation/3.concertriccir2.csv',header=None,names=names)
x = np.array(df.iloc[:,0:4])
y = np.array(df['class'])
df.head(5)


# In[5]:


x_1, x_test,y_1,y_test=cross_validation.train_test_split(x, y, test_size=0.3, random_state=0)
x_tr,x_cv,y_tr,y_cv=cross_validation.train_test_split(x_1,y_1, test_size=0.3)
for i in range(1,30,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_tr,y_tr)
    pred = knn.predict(x_cv)
    acc = accuracy_score(y_cv, pred, normalize=True) * float(100)
    print('\nCV accuracy for k = %d is %d%%' % (i, acc))
    


# In[7]:


x_1, x_test,y_1,y_test=cross_validation.train_test_split(x, y, test_size=0.3, random_state=0)
knn = KNeighborsClassifier(1)
knn.fit(x_tr,y_tr)
pred = knn.predict(x_test)
acc = accuracy_score(y_test, pred, normalize=True) * float(100)
print('\n****Test accuracy for k = 1 is %d%%' % (acc))


# In[8]:


from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn import cross_validation 
from sklearn.metrics import accuracy_score
myList = list(range(0,50))
neighbors = list(filter(lambda x: x % 2 != 0, myList))


cv_scores = []
x_1, x_test,y_1,y_test=cross_validation.train_test_split(x, y, test_size=0.3, random_state=0)
x_tr,x_cv,y_tr,y_cv=cross_validation.train_test_split(x_1,y_1, test_size=0.3)
# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_tr, y_tr, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())


MSE = [1 - x for x in cv_scores]


optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)

 
plt.plot(neighbors, MSE)

for xy in zip(neighbors, np.round(MSE,3)):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

print("the misclassification error for each k value is : ", np.round(MSE,3))

