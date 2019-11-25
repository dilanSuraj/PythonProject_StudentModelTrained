import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle

data = pd.read_csv("student-mat.csv", ";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best = 0
for i in range(30):
    x_train,x_test, y_train,y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    if acc > best:
         best = acc
         with open("studentmodel.pickle","wb") as f:
             pickle.dump(linear,f)

print("Accuracy : \n", acc)
print("Coefficient : \n", linear.coef_)
print("Intercept : \n", linear.intercept_)

