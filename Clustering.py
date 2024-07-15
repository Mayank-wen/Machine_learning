import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
dataset=pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values
#elbow method
wcss =[]
for i in range(2,11):#for 1 it wont run as minimum number of cluster should be 2
    km=KMeans(n_clusters=i,random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(2,11),wcss)        
plt.show()
kmeans=KMeans(n_clusters=5,init="k-means++",random_state=42)
y_means=kmeans.fit_predict(X)
print(y_means)
plt.scatter(X[y_means == 0,0],X[y_means == 0,1], s=100, c='red', label='Cluster1')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1], s=100, c='blue', label='Cluster2')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1], s=100, c='green', label='Cluster3')
plt.scatter(X[y_means == 3,0],X[y_means == 3,1], s=100, c='yellow', label='Cluster4')
plt.scatter(X[y_means == 4,0],X[y_means == 4,1], s=100, c='brown', label='Cluster5')
plt.show()