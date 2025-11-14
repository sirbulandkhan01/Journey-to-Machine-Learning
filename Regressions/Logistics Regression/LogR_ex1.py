import numpy as np
from sklearn.linear_model import LogisticRegression

#Data
study_hours = np.array([2,3,4,5,6,7,8]).reshape(-1,1)
Result = np.array([0,0,0,1,1,1,1])

#Fit
model = LogisticRegression()
model.fit(study_hours, Result)

print(f"Intercept: {float(model.intercept_[0]):.2}")
print(f"Coefficient: {float(model.coef_[0,0]):.4f}")

#predict for 4.5 hours
pred = model.predict([[4.5]])
print("Result if a student study 4.5 hours: ", pred[0])
