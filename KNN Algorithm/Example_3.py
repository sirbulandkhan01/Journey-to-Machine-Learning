from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Data set

x = np.array([
    [22,40],
    [25,50],
    [47,90],
    [52,110],
    [46,95],
    [56,100],
    [23,60]
])

y = np.array(['NO', 'NO', 'Yes', 'Yes', 'Yes', 'Yes', 'NO'])

#New Data Point

new_person = np.array([[30,60]])

#model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x,y)

#Prediction
prediction = knn.predict(new_person)
print("Predicted class", prediction[0])

