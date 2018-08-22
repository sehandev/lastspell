from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Limit to the two first classes, and split into training and test
X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                    test_size=.5,
                                                    random_state=random_state)


# Create a simple classifier
classifier = svm.LinearSVC(random_state=random_state)
print("\n\n-------------------- A ---------------------")
print(classifier)
classifier.fit(X_train, y_train)
print("\n\n-------------------- B ---------------------")
print(classifier)
y_score = classifier.decision_function(X_test)
print("\n\n-------------------- C ---------------------")
print(y_score)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

print("\n\n y_test ==========================")
print(y_test)
print("\n\n y_score ==========================")
print(y_score)
precision, recall, _ = precision_recall_curve(y_test, y_score)

print("\n\n precision ==========================")
print(precision)
print("\n\n recall ==========================")
print(recall)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

plt.savefig('test.png')