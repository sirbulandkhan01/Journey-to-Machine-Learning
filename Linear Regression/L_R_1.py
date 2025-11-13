import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Data
diameter_X = np.array([8, 10, 12]).reshape(-1,1)
price_Y = np.array([10, 13, 16])

#Fit
model = LinearRegression()
model.fit(diameter_X, price_Y)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.4f}")

#predict for 20 inches
pred = model.predict([[20]])
print("Predicted price for @0 inches Pizza is: ", pred[0])

#plot
plt.scatter(diameter_X, price_Y, color="blue",label='Data')
plt.plot(diameter_X, model.predict(diameter_X), color='red', label='Fit')
plt.scatter(13, pred, color='green', s=100, label='Prediction')
plt.xlabel('Diameter (inches)')
plt.ylabel('Price ($)')
plt.legend()
plt.show()