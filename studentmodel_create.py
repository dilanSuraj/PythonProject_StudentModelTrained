import numpy as np
import sklearn
from sklearn import linear_model
import pandas as pd

# Load data
data = pd.read_csv("student-mat.csv", ";")

# Extract data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Decide the predict column
predict = "G3"

# Extract X and y axis data
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Split the data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

# Train the model
linear.fit(x_train, y_train)

# Measure the accuracy
acc = linear.score(x_test, y_test)

print("Accuracy : \n", acc)
print("Coefficient : \n", linear.coef_)
print("Intercept : \n", linear.intercept_)

# Predict the data
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], " ", x_test[x], " ", y_test[x])