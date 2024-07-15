import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
dataset = pd.read_csv("Social_Network_Ads.csv")
#print(dataset)
x=dataset.iloc[: ,:-1]
y=dataset.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)
#Feature scaling
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
print(X_train)

classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,Y_train)
SVC(kernel='linear',random_state=0)
print(classifier.predict(sc.transform([[150000,1]]))[0])
y_pred=classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))

cm=confusion_matrix(Y_test,y_pred)
print(cm)
print(accuracy_score(y_pred,Y_test))