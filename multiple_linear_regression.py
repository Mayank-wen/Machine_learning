import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Load the dataset
dataSet = pd.read_csv("50_Startups.csv")

# Separate the features and the target variable
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

# Apply OneHotEncoder to the categorical column (assuming the 4th column)
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg= LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))


