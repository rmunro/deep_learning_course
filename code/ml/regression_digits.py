import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
train_data = data[:n_samples//2]
test_data = data[n_samples//2:]
y_train = digits.target[:n_samples//2]
y_test = digits.target[n_samples//2:]

# regr = linear_model.LinearRegression()
logisticRegr = LogisticRegression()   # this is where we say what type of regression to use

# Train the model using the training sets
logisticRegr.fit(train_data, y_train)

# Make predictions using the testing set
score = logisticRegr.score(test_data, y_test)
print(score)

from sklearn import metrics
predictions = logisticRegr.predict(test_data)
cm = metrics.confusion_matrix(y_test, predictions)

print(cm)







