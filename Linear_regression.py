import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plc

#reading dataset

dataset=pd.read_csv("Salary_Data.csv")
dataset.head(10)
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, -1].values

#creating model

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
reg= LinearRegression()
reg.fit(X_train,Y_train)

#prediction

print(X_test)
Y_pred=reg.predict(X_test)

# ploting the training set

plc.scatter(X_train,Y_train,color="red")
plc.plot(X_train,reg.predict(X_train),color="blue")
plc.show()

#ploting for the training set

plc.scatter(X_test,Y_test,color="green")
plc.plot(X_test,Y_pred,color="red")
plc.grid(True)
plc.show()