import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#Data set of 5 movies

x = np.array([
    [8.8, 148], # Inception
    [9.0, 152], # The Dark Knight
    [9.2, 175], # The God Father
    [8.9, 154], # Pulp Fiction
    [8.9, 201]  # LOTR: Return of The King
])

y = np.array(["sci-fi", "Action", "Drama", "Drama", "Fantasy"])

#Train the KNN Model

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x,y)

#Predict the genre of Interstellar
interstellar = np.array([[8.6, 169]])
predicted_genre = knn.predict(interstellar)

print("Movie: Interstellar (2014)")
print("IMDb Rating: 8.6 | Duration: 169")
print("Predicted Genre is: ", predicted_genre[0])
