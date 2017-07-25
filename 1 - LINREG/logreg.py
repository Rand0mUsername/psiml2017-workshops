import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.datasets import load_iris

print("=== activations ===")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(0))
testArray = np.array([1,5])
print(sigmoid(testArray))
x = np.arange(-10., 10., 0.2)
y = sigmoid(x)
plt.plot(x,y)
plt.show()

def relu(x):
    return np.maximum(0, x)

print(relu(-5))
print(relu(5))
testArray = np.array([3,0,-1,2,5,-2])
print(relu(testArray))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

testArray = np.array([-1,0.1899,0.4449,0.98990])
print(softmax(testArray))
print(np.sum(softmax(testArray)))

print("=== logistic regression ===")

def hypothesis(features, weights):
    return sigmoid(np.matmul(features, weights))

features = np.array([1, 2])
weights = np.array([0.05, 0.2])

print(hypothesis(features, weights))
print(sigmoid(0.45))

def cost_function(features, weights, targets):
    hyp = hypothesis(features, weights)
    cost = np.average(-targets * np.log(hyp) - (1 - targets) * np.log(1 - hyp))
    return cost

# saves and loads the model

def save_params(file_path, weights):
    np.savez(file_path, weights=weights)

def load_params(file_path):
    data = np.load(file_path)
    return data['weights']

weights = np.array([0.05,0.2])
save_params("testmodel.tsv", weights)
data = load_params("testmodel.tsv.npz")
print(data)

# calculates metrics

def train_results(targets, predictions):
    TP = FP = FN = TN = 0
    for i in range(len(targets)):
        if predictions[i]:
            if targets[i]:
                TP += 1
            else:
                FP += 1
        else:
            if targets[i]:
                FN += 1
            else:
                TN += 1

    if TP + FP == 0:
        P = 1
    else:
        P = TP / (TP + FP)
    
    if TP + FN == 0:
        R = 1
    else:
        R = TP / (TP + FN)
    
    return TP, FP, TN, FN, P, R

# prints stats on given dataset with given weights

def check_model(weights, features, targets):
    predictions = hypothesis(features, weights) >= 0.5
    TP, FP, TN, FN, P, R = train_results(targets, predictions)
    print("True positive count: " + str(TP))
    print("False positive count: " + str(FP))
    print("True negative count: " + str(TN))
    print("False negative count: " + str(FN))
    print("Precision: " + str(P))
    print("Recall: " + str(R))

# sklearn built in logistic regression for comparison

iris = load_iris()
x = iris.data
y = iris.target
y = (y == 1).astype(int).T
bias = np.zeros((x.shape[0],1)) + 1
x = np.append(bias, x, axis = 1)
h = .02  # step size in the mesh
LR = linear_model.LogisticRegression(C=1e5)
LR.fit(x, y)
predictions = LR.predict(x)
TP, FP, TN, FN, P, R = train_results(y, predictions)
print("True positive count: " + str(TP))
print("False positive count: " + str(FP))
print("True negative count: " + str(TN))
print("False negative count: " + str(FN))
print("Precision: " + str(P))
print("Recall: " + str(R))

# gradient descent

def logistic_regression(features, targets, epochCount, learning_rate):
    num_examples = features.shape[0]
    num_features = features.shape[1]
    weights = np.zeros(num_features)

    for step in range(epochCount):
        err = hypothesis(features, weights) - targets
        delta = (learning_rate / num_examples) * np.matmul(features.T, err)
        weights -= delta

        # Save cost function value every so often
        if step % 100 == 0:
            costData.append(cost_function(features, weights, targets))

    return weights

# run
costData = []
epochCount = 10000
weights = logistic_regression(x, y, epochCount = epochCount, learning_rate = 0.1)
print(len(costData))
print(weights)

# pick an example
print(x[80])
print(y[80])
print(sigmoid(weights[0] + weights[1] * 5.5 + weights[2] * 2.4 + weights[3] * 3.8 + weights[4] * 1.1))

# plot cost data
a = np.arange(0., epochCount/100, 1.)
plt.plot(a,costData)
plt.show()

# check model
check_model(weights, x, y)

print("=== regularization ===")

def cost_function_reg(features, weights, targets, regularization):
    num_examples = features.shape[0]
    num_features = features.shape[1]
    cost = cost_function(features, weights, targets)
    cost += (regularization / (2 * num_examples)) * np.sum(weights * weights)
    return cost

# gradient descent with regularization

def logistic_regression_reg(features, targets, epochCount, learning_rate, regularization):
    num_examples = features.shape[0]
    num_features = features.shape[1]
    weights = np.zeros(num_features)
    
    for step in range(epochCount):

        err = hypothesis(features, weights) - targets
        reg = (regularization / num_features) * weights
        reg[0] = 0 # don't regularize bias
        delta = (learning_rate / num_examples) * (np.matmul(features.T, err) + reg)
        weights -= delta

        # Save cost function value every so often
        if step % 100 == 0:
            costData.append(cost_function(features, weights, targets))

    return weights

costData = []
epochCount = 10000
weights = logistic_regression_reg(x, y, epochCount = epochCount, learning_rate = 0.1, regularization = 0.1)
print(weights)
a = np.arange(0., epochCount/100, 1.)
plt.plot(a,costData)
plt.show()
check_model(weights, x, y)