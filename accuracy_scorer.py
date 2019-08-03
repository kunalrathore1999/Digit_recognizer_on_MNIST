import numpy as np
from sklearn.tree import DecisionTreeClassifier
#from sklearn.preprocessing import StandardScaler 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df=pd.read_csv("train.csv").values

#train_test_split
#random_state=1 specify that data should be shuffle before selection
#stratify=df[:,1:] specifiy that that the ration of classes should be same in train and test case ex: [30 30 30]
#train and [15 15 15] for test
xtrain,xtest,ytrain,ytest=train_test_split(df[:,1:],df[:,0],test_size=0.2,random_state=1,stratify=df[:,0])
#ss=StandardScaler()
#ss.fit(xtrain)
#xtrain=ss.transform(xtrain)
#xtest=ss.transform(xtest)
#Clasifier
clf=DecisionTreeClassifier()

clf.fit(xtrain,ytrain)
y_pred=clf.predict(xtest)
#df3= pd.DataFrame(data={"ImageId": list(range(1,28001)), "Label": y_pred})
#df3.to_csv("./file.csv", sep=',',index=False)  
#d=xtest[50]
#d.shape=(28,28)
#pt.imshow(255-d,cmap='gray')
#print(clf.predict([xtest[50]]),test_label[50])
#pt.show()
print('Accuracy of DecisionTree is ',accuracy_score(ytest,y_pred))
