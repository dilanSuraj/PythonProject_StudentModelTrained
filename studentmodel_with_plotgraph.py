import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", ";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train,x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

acc = linear.score(x_test, y_test)

predictions = linear.predict(x_test)

# Draw plot
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()