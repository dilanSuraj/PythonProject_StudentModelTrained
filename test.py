import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read data, delimetter in here is ;
data = pd.read_csv("student-mat.csv",sep=";")

# Display first 5 columns
print(data.head())

# Extracted attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Display first 5 columns
print(data.head())

# Predict Column
predict = "G3"

X =  np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Split into arrays
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# # model
# linear = linear_model.LinearRegression()
# linear.fit(x_train, y_train)
#
# # accuracy
# acc = linear.score(x_test, y_test)
#
# # Saving the model
# # Open the file in wb Mode
# with open("studentmodel.pickle", "wb") as f:
#     # Save the file
#     pickle.dump(linear, f)

# Read the saved model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

acc = linear.score(x_test, y_test)

print("Accuracy : \n", acc)
print("Coefficient :\n", linear.coef_)
print("Intercept :\n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
