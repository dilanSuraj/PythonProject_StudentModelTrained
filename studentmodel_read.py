import numpy as np
import sklearn
from sklearn import linear_model
import pandas as pd
import pickle

data = pd.read_csv("student-mat.csv",";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.1)

# linear = linear_model.LinearRegression()
# linear.fit(x_train, y_train)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

acc = linear.score(x_test, y_test)

print("Accuracy : \n", acc)
print("Coefficient : \n", linear.coef_)
print("Intercept : \n", linear.intercept_)

predictions =linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], " ", x_test[x], " ", y_test[x])