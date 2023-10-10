import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###### IMPLEMENT LOGISTICS REGRESSION MODEL FROM SCRATCH

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterarions=5000, tolerance=1e-4, threshold = 0.05):
        self.alpha = learning_rate
        self.num_iter = num_iterarions
        self.tol = tolerance
        self.thresh = threshold

        self._weight = None
        self._X = None


    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _transform(self, X):
        ones = X.shape[0]
        self._X = np.c_[np.ones((ones, 1)), X]
        return self._X

    def fit(self, X, y):
        self._X = self._transform(X)
        self._weight = np.zeros((self._X.shape[1], 1))

        for _ in range(self.num_iter):
            z = self._X @ self._weight
            y_pred = self._sigmoid(z)
            gradient = (self._X.T @ (y_pred - y))
            self._weight -= self.alpha * gradient

            if np.linalg.norm(gradient) < self.tol:
                break

        return

    def predict_prob(self, X):
        if self._weight is None:
            raise Exception('Model was not fitted before')

        self._X = self._transform(X)

        prob_estimated = self._sigmoid(self._X @ self._weight)
        return prob_estimated

    def predict(self, X):
        prob_estimated = self.predict_prob(X)

        classification_predicted = np.zeros(X.shape[0])
        classification_predicted[np.where(prob_estimated > self.thresh)[0]] = 1

        return classification_predicted

    def accuracy(self, X, y):
        y_predict = self.predict(X)
        num_matches = (y_predict == y.squeeze()).sum()
        total_obs = y.shape[0]
        return num_matches / total_obs

    def plot_data(self, X):
        group = self.predict(X)
        class_0 = group[group == 0]
        class_1 = group[group == 1]

        plt.plot(class_0, 'ro', label='0')
        plt.plot(class_1, 'bo', label='1')
        plt.legend()
        plt.show()

##### TRAINING DATASET #######

# Import train_data
train_data = pd.read_csv(r'D:\NEU\Năm 3\ML\Data\ds1_train.csv')
X_train = train_data[['x_1', 'x_2']].values
y_train = train_data['y'].values.reshape((-1, 1))

model = LogisticRegression()
model.fit(X_train,y_train)

# Test model acuracy on Trainning Dataset
training_score = model.accuracy(X_train,y_train)
print('Accuracy on training dataset : {:.4f}%'.format(training_score*100))

##### TEST DATASET ########
test_data = pd.read_csv(r'D:\NEU\Năm 3\ML\Data\ds1_valid.csv')
X_test = test_data[['x_1','x_2']].values
y_test = test_data['y'].values.reshape((-1, 1))

test_model = LogisticRegression()
test_model.fit(X_test,y_test)

test_score = test_model.accuracy(X_test, y_test)
print('Accuracy on training dataset : {:.4f}%'.format(test_score*100))

########## VISUALIZE CLASSIFICATION RESULT ###############

# Create a mesh grid to cover the feature space
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Use the trained model to make predictions on the mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary as a filled contour plot
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.6)

# Calculate accuracy using the accuracy_score method
accuracy = test_model.accuracy(X_test,y_test)
print("Accuracy on test set by our model: {:.4f}%".format(accuracy * 100))

# Visualize the decision boundary
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_model.predict(X_test), cmap=plt.cm.RdYlBu, marker='o')
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.title("Logistic Regression Decision Boundary")
plt.show()


